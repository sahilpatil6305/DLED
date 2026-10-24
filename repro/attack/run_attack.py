"""
Rejection-aware PGD against DLED, with official re-evaluation.

  repro/run.sh attack_pilot python repro/attack/run_attack.py \
      --run ~/DLED/runs/<baseline run> --n_per_class 50 --eps 1 2 4 8

Pipeline
  1. sample n test images per label from DF40_FF_test_NOFE_all (official
     collection code, fixed seed; 0 real, 1 FS, 2 FR, 3 EFS = unknown)
  2. load checkpoint + the thresholds the official test used (training log)
  3. clean check: harness decision (fp32 + torch freq map) vs official
     model_inference on official-loader tensors -> agreement rate
  4. for each attack (E1 E2 E3 E3R E4) and eps: PGD on the fp32 copy (STE for
     the uint8 step); round to 8-bit; SAVE AS PNG; re-read with the OFFICIAL
     loader and decide with the OFFICIAL model_inference (native precision).
     Reported numbers come from that last step only.
Outputs (in $DLED_LOG_DIR, i.e. the run folder under run.sh):
  samples.csv, thresholds.json, clean.csv, per_sample.csv, summary.json, summary.txt,
  examples/ (a few adversarial PNGs per attack/eps).
CPU smoke test: add --device cpu --n_per_class 2 --steps 5 --eps 8.
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import tempfile
import time

ap = argparse.ArgumentParser()
ap.add_argument('--run', help='baseline run folder (finds ckpt + training log)')
ap.add_argument('--ckpt')
ap.add_argument('--log', nargs='*', help='log file(s) holding "Retrieved uncertainty dict"')
ap.add_argument('--json_dir', default=os.path.join(os.environ.get('DLED_DATA', os.path.expanduser('~/DLED/data')), 'DF40/json_file'))
ap.add_argument('--n_per_class', type=int, default=50)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--attacks', nargs='+', default=['E1', 'E2', 'E3', 'E3R', 'E4'])
ap.add_argument('--eps', nargs='+', type=float, default=[1, 2, 4, 8], help='in 1/255 units')
ap.add_argument('--steps', type=int, default=50)
ap.add_argument('--no_random_start', action='store_true')
ap.add_argument('--quantize', default='ste', choices=['ste', 'none', 'floor'])
ap.add_argument('--kappa', type=float, default=0.05)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
ap.add_argument('--threads', type=int, default=4)
ap.add_argument('--keep_examples', type=int, default=4)
ap.add_argument('--out', default=os.environ.get('DLED_LOG_DIR', '.'))
args = ap.parse_args()

if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(args.threads)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dled_attack as da  # noqa: E402
import freq_torch as ft  # noqa: E402

K = 3
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)
t_start = time.time()


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------ locate ckpt + log
ckpt, logs = args.ckpt, args.log or []
if args.run:
    import glob
    from datetime import datetime
    r = os.path.expanduser(args.run)
    if not ckpt:
        # DLED writes two folders: train.py's logger (training.log) and the trainer's
        # (test/<key>/ckpt_best.pth), with timestamps seconds apart. run.sh moves both into
        # the run folder at the end; while training only the first is linked, so also look
        # for its sibling in ./logs by timestamp.
        cands = glob.glob(os.path.join(r, 'dled_logs*', 'test', 'DF40_FF_test_NOFE_all', 'ckpt_best.pth'))
        link = os.path.join(r, 'dled_logs')
        if not cands and os.path.islink(link):
            tgt = os.path.realpath(link)

            def ts(d):
                try:
                    return datetime.strptime(os.path.basename(d)[-19:], '%Y-%m-%d-%H-%M-%S')
                except ValueError:
                    return None
            t0 = ts(tgt)
            for d in glob.glob(os.path.join(os.path.dirname(tgt), '*')):
                t = ts(d)
                if t0 and t and abs((t - t0).total_seconds()) <= 120:
                    cands += glob.glob(os.path.join(d, 'test', 'DF40_FF_test_NOFE_all', 'ckpt_best.pth'))
        if len(cands) == 1:
            ckpt = cands[0]
        elif len(cands) > 1:
            sys.exit(f'several checkpoints found, pass --ckpt: {cands}')
    if not logs:
        logs = [os.path.join(r, 'dled_logs', 'training.log')] + \
               sorted(os.path.join(r, f) for f in os.listdir(r) if f.startswith('DLED_') and f.endswith('.log'))
assert ckpt and os.path.isfile(ckpt), f'checkpoint not found ({ckpt}); pass --ckpt'
thr, thr_src = da.load_thresholds(logs)
thr_f = thr['fused']
log('checkpoint :', ckpt)
log('thresholds :', thr_src)
log('fused tau  :', {k: round(v, 5) for k, v in thr_f.items()})
with open(os.path.join(args.out, 'thresholds.json'), 'w') as f:
    json.dump({'source': thr_src, 'ckpt': ckpt, 'thresholds': {b: {str(k): v for k, v in d.items()} for b, d in thr.items()}}, f, indent=1)

# ------------------------------------------------------------ sample test images
from dataset.frequency_dateset import DeepfakeFrequencyDataset as D  # noqa: E402

cfg = da.dled_config()
cfg.update(dataset_json_folder=args.json_dir)
ds = D.__new__(D)
ds.config, ds.mode = cfg, 'test'
ds.compression, ds.frame_num = cfg['compression'], cfg['frame_num']['test']
ds.video_level, ds.clip_size, ds.lmdb = False, None, False
paths, labels, _ = ds.collect_img_and_label_for_one_dataset('DF40_FF_test_NOFE_all')
rng = random.Random(args.seed)
samples = []
for lab in range(K + 1):
    idx = [i for i, l in enumerate(labels) if l == lab]
    for i in sorted(rng.sample(idx, min(args.n_per_class, len(idx)))):
        p = paths[i]
        method = p.split('/DF40/')[1].split('/')[0] if '/DF40/' in p else 'FF++_real'
        samples.append((p, lab, method))
with open(os.path.join(args.out, 'samples.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['idx', 'path', 'label', 'method'])
    for i, (p, l, m) in enumerate(samples):
        w.writerow([i, p, l, m])
log(f'samples    : {len(samples)} ({args.n_per_class} per label, seed {args.seed})')

# ------------------------------------------------------------ models
det_off = da.build_detector(ckpt, dev)                      # official, native precision
if args.device == 'cpu':                                    # fp16 matmuls are not reliable on CPU
    det_off = da.to_fp32(det_off)
det_atk = da.to_fp32(da.build_detector(ckpt, dev))          # attack copy
log('official precision:', next(det_off.model.spatial_image_encoder.parameters()).dtype,
    '| attack precision: float32')


def official_decide(img_paths):
    """Official loader tensors -> official model_inference -> fused decision."""
    sp, fr = [], []
    for p in img_paths:
        img, freq, _ = ds.load_rgb(p)
        sp.append(ds.normalize(ds.to_tensor(np.array(img))))
        fr.append(ds.normalize(ds.to_tensor(np.array(freq))))
    dd = {'image': torch.stack(sp).to(dev), 'freq': torch.stack(fr).to(dev)}
    with torch.no_grad():
        pred = det_off.model_inference(dd, thr)
    return pred['prob']['fused'].cpu()


def batches(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


# ------------------------------------------------------------ clean check
clean = {}
agree = 0
with open(os.path.join(args.out, 'clean.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['idx', 'label', 'method', 'official', 'harness'])
    for b in batches(list(range(len(samples))), args.batch):
        d_off = official_decide([samples[i][0] for i in b])
        x = torch.stack([ft.load_resized(samples[i][0]) for i in b]).to(dev)
        with torch.no_grad():
            d_h = da.decide(da.forward(det_atk, x, 'floor')['evidence'], thr_f, K).cpu()
        for j, i in enumerate(b):
            clean[i] = int(d_off[j])
            agree += int(d_off[j] == d_h[j])
            w.writerow([i, samples[i][1], samples[i][2], int(d_off[j]), int(d_h[j])])
lab_arr = np.array([s[1] for s in samples])
cl_arr = np.array([clean[i] for i in range(len(samples))])
log(f'clean      : (K+1)-way acc {np.mean(cl_arr == lab_arr):.3f} | unknown DR '
    f'{np.mean(cl_arr[lab_arr == K] == K):.3f} | harness/official agreement {agree}/{len(samples)}')

# ------------------------------------------------------------ attacks
tmp = tempfile.mkdtemp(prefix='dled_adv_')
gen = torch.Generator().manual_seed(args.seed)
rows, summary = [], {'args': vars(args), 'clean_agreement': agree / len(samples),
                     'clean_acc': float(np.mean(cl_arr == lab_arr)),
                     'clean_unknown_dr': float(np.mean(cl_arr[lab_arr == K] == K)), 'results': {}}
for attack in args.attacks:
    idx = [i for i in range(len(samples)) if da.ATTACKS[attack](samples[i][1], K)]
    for eps_px in args.eps:
        eps = eps_px / 255.0
        t0 = time.time()
        n_ok_h = n_ok_off = n_clean_ok = 0
        for b in batches(idx, args.batch):
            x0 = torch.stack([ft.load_resized(samples[i][0]) for i in b]).to(dev)
            y = torch.tensor([samples[i][1] for i in b], device=dev)
            x_adv, hit, loss = da.pgd(det_atk, x0, y, attack, thr_f, K, eps, args.steps,
                                      random_start=not args.no_random_start,
                                      quantize=args.quantize, kappa=args.kappa, generator=gen)
            with torch.no_grad():
                d_h = da.decide(da.forward(det_atk, x_adv, 'floor')['evidence'], thr_f, K)
            ok_h = da.success(attack, d_h, y, K).cpu()
            pngs = []
            for j, i in enumerate(b):
                pp = os.path.join(tmp, f'{i}.png')
                ft.save_png(x_adv[j], pp)
                pngs.append(pp)
            d_off = official_decide(pngs)
            ok_off = da.success(attack, d_off, y.cpu(), K)
            ok_clean = da.success(attack, torch.tensor([clean[i] for i in b]), y.cpu(), K)
            linf = ((x_adv - x0).abs().amax((1, 2, 3)) * 255).round().cpu()
            for j, i in enumerate(b):
                rows.append([attack, eps_px, i, samples[i][1], samples[i][2], clean[i], int(d_h[j]),
                             int(d_off[j]), int(ok_clean[j]), int(ok_off[j]), int(hit[j]),
                             int(linf[j]), float(loss[j])])
                ex_dir = os.path.join(args.out, 'examples', f'{attack}_eps{eps_px:g}')
                if i in idx[:args.keep_examples]:
                    os.makedirs(ex_dir, exist_ok=True)
                    shutil.copy(pngs[j], os.path.join(ex_dir, f'{i}_y{samples[i][1]}_clean{clean[i]}_adv{int(d_off[j])}.png'))
            n_ok_h += int(ok_h.sum()); n_ok_off += int(ok_off.sum()); n_clean_ok += int(ok_clean.sum())
        n = len(idx)
        res = {'n': n, 'clean_rate': n_clean_ok / max(n, 1), 'harness_rate': n_ok_h / max(n, 1),
               'official_rate': n_ok_off / max(n, 1), 'seconds': round(time.time() - t0, 1)}
        summary['results'][f'{attack}@{eps_px:g}'] = res
        log(f'{attack:4s} eps={eps_px:g}/255  n={n:4d}  success clean={res["clean_rate"]:.3f}  '
            f'harness={res["harness_rate"]:.3f}  OFFICIAL={res["official_rate"]:.3f}  ({res["seconds"]}s)')

with open(os.path.join(args.out, 'per_sample.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['attack', 'eps', 'idx', 'label', 'method', 'clean_dec', 'adv_dec_harness', 'adv_dec_official',
                'success_clean', 'success_official', 'first_success_step', 'linf_px', 'final_loss'])
    w.writerows(rows)

# ------------------------------------------------------------ masking checks (Carlini et al. 2019 style)
checks = {}
for attack in args.attacks:
    rates = [summary['results'][f'{attack}@{e:g}']['official_rate'] for e in sorted(args.eps)]
    checks[f'{attack}_monotone_in_eps'] = all(b >= a - 0.02 for a, b in zip(rates, rates[1:]))
    checks[f'{attack}_beats_clean'] = rates[-1] >= summary['results'][f'{attack}@{max(args.eps):g}']['clean_rate']
mism = sum(1 for r in rows if r[6] != r[7])
checks['harness_vs_official_mismatch_rate'] = mism / max(len(rows), 1)
summary['checks'] = checks
summary['minutes'] = round((time.time() - t_start) / 60, 1)
with open(os.path.join(args.out, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(args.out, 'summary.txt'), 'w') as f:
    f.write(f"clean (K+1)-acc {summary['clean_acc']:.3f}  unknown DR {summary['clean_unknown_dr']:.3f}  "
            f"harness/official clean agreement {summary['clean_agreement']:.3f}\n")
    f.write('attack  eps  n  clean_success  official_success\n')
    for k, v in summary['results'].items():
        a, e = k.split('@')
        f.write(f"{a:5s} {e:>4s} {v['n']:4d}  {v['clean_rate']:.3f}  {v['official_rate']:.3f}\n")
    for k, v in checks.items():
        f.write(f'check {k}: {v}\n')
shutil.rmtree(tmp, ignore_errors=True)
log('checks     :', checks)
log(f'ATTACK RUN DONE in {summary["minutes"]} min -> {args.out}')
