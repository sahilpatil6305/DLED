"""
Adversarial fine-tuning of DLED (NOEFS) from a trained checkpoint.

  repro/run.sh at_pgd python repro/adv_train/at_train.py --mode pgd \
      --ckpt <ckpt_best.pth> --log <training/DLED_*.log with the thresholds> --out {RUN}

Modes (all fine-tune the same LoRA + prompt parameters DLED trains; official loss and
optimisers; the attack is L_inf PGD on the 224x224 image, freq map via freq_torch STE):
  pgd     AT-1, Madry PGD-AT. Adversary: untargeted CE on fused class probabilities.
          Loss: official EDL+AvU loss on the clean (augmented) batch + on the adversarial batch.
  evat    AT-2, EV-AT-style (after Sournac et al. 2026; our adaptation, not their code).
          Adversary: maximise ||log alpha_adv - log alpha_clean||^2 (fused Dirichlet).
          Loss: official clean loss + beta * that alignment term.
  reject  AT-3, rejection-aware (ours). Adversary per sample: real -> E2 (push into rejection);
          fake -> E2 or E4 (push to 'real'), chosen at random. Loss: official clean loss +
          official loss on the adversarial batch + gate hinge relu(kappa + l_y - z_y)
          (+ for fakes relu(kappa + z_real - z_y)), with z = log(1+e), l_c = -log tau_c.

Threat model at training time = test time: the attack perturbs the un-augmented resized image
(the image the frequency branch sees in DLED's loader) and both branches see x + delta.

Outputs in --out: at_last.pth (COMPLETE state dict, including the frozen LoRA base weights that
DLED's own checkpoints omit, FINDINGS F13), at_epoch<k>.pth, training.log (contains a
'Retrieved uncertainty dict {...} !' line, recomputed after training exactly like
trainer.calculate_uthre, so repro/attack/*.py can read it), train_log.jsonl, config.json.
Evaluate with: run_attack.py --ckpt <out>/at_last.pth --log <out>/training.log
CPU smoke test: --device cpu --batch 2 --steps_per_epoch 2 --epochs 1 --thr_batches 2 --workers 0
"""
import argparse
import json
import math
import os
import random
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument('--mode', required=True, choices=['pgd', 'evat', 'reject', 'clean'])
ap.add_argument('--ckpt', required=True, help='trained DLED checkpoint to fine-tune')
ap.add_argument('--log', nargs='+', required=True, help='log(s) holding the checkpoint thresholds')
ap.add_argument('--out', default=os.environ.get('DLED_LOG_DIR', '.'))
ap.add_argument('--epochs', type=int, default=5)
ap.add_argument('--steps_per_epoch', type=int, default=0, help='0 = full epoch (1688 at batch 32)')
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--lr', type=float, default=2e-4, help='peak LR (official run used 2e-3); cosine to 0')
ap.add_argument('--clip', type=float, default=5.0, help='grad-norm clip (official ramps 1->5)')
ap.add_argument('--eps', type=float, default=4.0, help='training eps in 1/255')
ap.add_argument('--attack_steps', type=int, default=3)
ap.add_argument('--alpha', type=float, default=0.0, help='step size in 1/255; 0 = 2.5*eps/steps')
ap.add_argument('--w_adv', type=float, default=1.0)
ap.add_argument('--beta', type=float, default=1.0, help='evat alignment weight')
ap.add_argument('--kappa', type=float, default=0.05)
ap.add_argument('--grad_scale', type=float, default=1024.0, help='loss scale for fp16 input gradients')
ap.add_argument('--thr_batches', type=int, default=0, help='0 = full train loader for thresholds')
ap.add_argument('--log_every', type=int, default=50)
ap.add_argument('--seed', type=int, default=1024)
ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()

if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(args.threads)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'attack'))
import dled_attack as da  # noqa: E402  (also puts the repo + training/ on sys.path)
import freq_torch as ft  # noqa: E402
from dataset.frequency_dateset import DeepfakeFrequencyDataset as D  # noqa: E402
from loralib.layers import LinearLoRA  # noqa: E402

K = 3
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
LOGF = open(os.path.join(args.out, 'training.log'), 'a')
JSONL = open(os.path.join(args.out, 'train_log.jsonl'), 'a')


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    LOGF.write(time.strftime('%Y-%m-%d %H:%M:%S') + ' - ' + msg + '\n'); LOGF.flush()


class ATDataset(D):
    """Official training dataset + the un-augmented resized image (the attacked input)."""

    def __getitem__(self, index, no_norm=False):
        item = super().__getitem__(index, no_norm)
        if len(item) == 8:                       # official error path already returned ours
            return item
        p = item[-1][0] if isinstance(item[-1], (list, tuple)) else item[-1]
        return item + (ft.load_resized(p, self.config['resolution']),)

    @staticmethod
    def collate_fn(batch):
        dd = D.collate_fn([b[:7] for b in batch])
        dd['x_raw'] = torch.stack([b[7] for b in batch])
        return dd


def save_full(det, path):
    """state_dict + the frozen LoRA base weights (plain tensors after step 1, not in state_dict)."""
    sd = {k: v.detach().cpu() for k, v in det.state_dict().items()}
    n = 0
    for name, m in det.named_modules():
        if isinstance(m, LinearLoRA):
            sd[name + '.weight'] = m.weight.detach().cpu().clone()
            n += 1
    torch.save(sd, path)
    return n


def official_loss(det, data, epoch, total):
    pred = det(data, 'fusion')
    pred.update({'epoch': epoch, 'total_epoch': total, 'training': True})
    return det.get_losses(data, pred, mode='train')['overall'], pred


def adv_objective(mode, out, y, ref, objE2, thr_f):
    e = out['evidence']
    if mode == 'pgd':
        return da.ce_loss('E1', e, y, K)                          # minimise log p_y
    if mode == 'evat':
        return -(torch.log(out['pred_alpha'].float()) - ref).pow(2).sum(1)
    if mode == 'reject':
        l2 = da.attack_loss('E2', e, y, thr_f, K, args.kappa)
        l4 = da.attack_loss('E4', e, y, thr_f, K, args.kappa)
        return torch.where(objE2, l2, l4)
    raise ValueError(mode)


def make_adv(det, x0, y, mode, thr_f, gen):
    eps = args.eps / 255.0
    alpha = (args.alpha / 255.0) if args.alpha else 2.5 * eps / args.attack_steps
    delta = (torch.rand(x0.shape, generator=gen).to(x0) * 2 - 1) * eps
    delta = (x0 + delta).clamp(0, 1) - x0
    ref, objE2 = None, None
    if mode == 'evat':
        with torch.no_grad():
            ref = torch.log(da.forward(det, x0, 'floor')['pred_alpha'].float())
    if mode == 'reject':
        coin = torch.rand(y.shape, generator=gen).to(y.device) < 0.5
        objE2 = (y == 0) | coin
    for _ in range(args.attack_steps):
        delta.requires_grad_(True)
        out = da.forward(det, x0 + delta, 'ste')
        L = adv_objective(mode, out, y, ref, objE2, thr_f)
        g, = torch.autograd.grad(L.sum() * args.grad_scale, delta)   # params get no .grad
        with torch.no_grad():
            delta = (delta - alpha * g.sign()).clamp(-eps, eps)
            delta = (x0 + delta).clamp(0, 1) - x0
    return (x0 + delta).detach(), objE2, ref


@torch.no_grad()
def thresholds(det, loader, max_batches=0):
    """Same procedure as trainer.calculate_uthre (augmented train loader, top-5% vacuity per class)."""
    un = {c: {'spatial': [], 'frequency': [], 'fused': []} for c in ['overall'] + list(range(K))}
    for b, dd in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        data = {'image': dd['image'].to(dev), 'freq': dd['freq'].to(dev)}
        p = det.model_full_fp(data)
        lab = dd['label'].to(dev)
        for br, key in (('spatial', 'spatial_vacuity'), ('frequency', 'frequency_vacuity'), ('fused', 'vacuity')):
            v = p[key].float()
            un['overall'][br].extend(v.tolist())
            for c in range(K):
                un[c][br].extend(v[lab == c].tolist())
    thr = {'spatial': {}, 'frequency': {}, 'fused': {}}
    for c, d in un.items():
        for br, vals in d.items():
            vals = sorted(vals, reverse=True)
            thr[br][c] = vals[int(len(vals) * 0.05)] if vals else 1.0
    return thr


def main():
    json.dump(vars(args), open(os.path.join(args.out, 'config.json'), 'w'), indent=1)
    thr0, src = da.load_thresholds(args.log)
    thr_f = thr0['fused']
    log('mode', args.mode, '| ckpt', args.ckpt, '| thresholds from', src, '| fused tau', thr_f)

    cfg = da.dled_config(prec='fp32' if args.device == 'cpu' else None)
    cfg.update(train_batchSize=args.batch, workers=args.workers)
    ds = ATDataset(config=cfg, mode='train')
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                                         collate_fn=ATDataset.collate_fn, drop_last=True)
    det = da.build_detector(args.ckpt, args.device, mode='train', prec='fp32' if args.device == 'cpu' else None)
    trainable = [p for p in det.parameters() if p.requires_grad]
    log(f'train images/epoch {len(ds)} | batches/epoch {len(loader)} | trainable params '
        f'{sum(p.numel() for p in trainable):,} | precision {det.model.spatial_image_encoder.conv1.weight.dtype}')

    spe = args.steps_per_epoch or len(loader)
    total = args.epochs * spe
    gen = torch.Generator().manual_seed(args.seed)
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        det.train()
        for it, dd in enumerate(loader):
            if it >= spe:
                break
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / total))
            for opt in det._optims.values():
                if opt is not None:
                    for g in opt.param_groups:
                        g['lr'] = lr
            y = dd['label'].to(dev)
            clean = {'image': dd['image'].to(dev), 'freq': dd['freq'].to(dev), 'label': y}
            det.model_zero_grad()
            # clean (official) loss, backpropagated first to free its graph
            loss_c, _ = official_loss(det, clean, epoch, args.epochs)
            loss_c.backward()
            rec = {'step': step, 'epoch': epoch, 'lr': lr, 'loss_clean': float(loss_c)}
            if args.mode != 'clean':
                x0 = dd['x_raw'].to(dev)
                x_adv, objE2, ref = make_adv(det, x0, y, args.mode, thr_f, gen)
                adv = {'image': ft.spatial_map(x_adv), 'freq': ft.freq_map(x_adv, 'floor'), 'label': y}
                if args.mode == 'evat':                  # ref = clean log-alpha from make_adv
                    pred_a = det(adv, 'fusion')
                    loss_a = args.beta * (torch.log(pred_a['pred_alpha'].float()) - ref).pow(2).sum(1).mean()
                else:
                    loss_a, pred_a = official_loss(det, adv, epoch, args.epochs)
                    if args.mode == 'reject':
                        z = torch.log1p(pred_a['evidence'].float())
                        lg = -torch.log(torch.tensor([thr_f[c] for c in range(K)], device=z.device))
                        zy = z.gather(1, y[:, None])[:, 0]
                        gate = torch.relu(args.kappa + lg[y] - zy)
                        anti_real = torch.relu(args.kappa + z[:, 0] - zy) * (y > 0).float()
                        loss_a = loss_a + (gate + anti_real).mean()
                (args.w_adv * loss_a).backward()
                with torch.no_grad():
                    d = da.decide(pred_a['evidence'].float(), thr_f, K)
                rec.update(loss_adv=float(loss_a), adv_correct=float((d == y).float().mean()),
                           adv_rejected=float((d == K).float().mean()),
                           adv_fake_as_real=float(((d == 0) & (y > 0)).float().sum() / max(1, int((y > 0).sum()))))
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=args.clip)
            det.model_update()
            if not all(math.isfinite(v) for k, v in rec.items() if k.startswith('loss')):
                log('non-finite loss, stopping', rec); sys.exit(3)
            JSONL.write(json.dumps(rec) + '\n')
            if step % args.log_every == 0:
                JSONL.flush()
                el = time.time() - t0
                log(f'step {step}/{total} ep {epoch} lr {lr:.2e} ' +
                    ' '.join(f'{k} {v:.4f}' for k, v in rec.items() if k not in ('step', 'epoch', 'lr')) +
                    f' | {el / max(step, 1):.2f}s/step, eta {el / max(step, 1) * (total - step) / 3600:.1f}h')
            step += 1
        n = save_full(det, os.path.join(args.out, f'at_epoch{epoch}.pth'))
        log(f'epoch {epoch} done, saved at_epoch{epoch}.pth ({n} frozen base weights included)')

    save_full(det, os.path.join(args.out, 'at_last.pth'))
    log('computing rejection thresholds on the augmented train loader (as trainer.calculate_uthre)')
    det.train()                                  # official computes them in train mode too
    thr = thresholds(det, loader, args.thr_batches)
    log('Retrieved uncertainty dict {} !'.format(thr))
    log(f'AT TRAINING DONE in {(time.time() - t0) / 3600:.2f} h -> {args.out}')


if __name__ == '__main__':
    main()
