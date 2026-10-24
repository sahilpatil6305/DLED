"""
Rejection-aware white-box attacks on DLED (library). CLI: run_attack.py.

DLED's decision rule (detectors/dled.py, model_inference, fused branch), with
e = fused evidence (B, K), K known classes (NOEFS: 0 real, 1 FS, 2 FR):
    c   = argmax e
    u   = 1 / (1 + max e)                      (fused vacuity)
    out = K ("unknown") if u > tau_c else c    (tau_c: per-class threshold)
Equivalently: accepted as c  <=>  c = argmax e  and  z_c >= l_c,
with z = log(1 + e) and l_c = -log(tau_c). All attack losses below are hinge
margins on z, so the attack optimises exactly the quantity the rule compares.

Error types (decision d, true label y; y == K means unknown / EFS):
    E1    known y,   d known and d != y    misclassified while accepted
    E2    known y,   d == K                falsely rejected (denial of service)
    E3    unknown,   d known               unknown accepted ("confidently wrong")
    E3R   unknown,   d == 0                unknown fake accepted as REAL
    E4    any fake,  d == 0                fake (known or unknown) accepted as real
"""
import ast
import os
import re
import sys

import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in (REPO, os.path.join(REPO, 'training'), os.path.dirname(os.path.abspath(__file__))):
    if p not in sys.path:
        sys.path.insert(0, p)
import freq_torch as ft  # noqa: E402

ATTACKS = {   # name -> (labels it applies to, given K)
    'E1': lambda y, K: y < K,
    'E2': lambda y, K: y < K,
    'E3': lambda y, K: y == K,
    'E3R': lambda y, K: y == K,
    'E4': lambda y, K: y > 0,
}


# ---------------------------------------------------------------- model
def dled_config(num_classes=3):
    """Same YAML merge and overrides as train.py / repro/smoke_model.py (NOEFS)."""
    import yaml
    with open(os.path.join(REPO, 'training/config/detector/dled.yaml')) as f:
        config = yaml.safe_load(f)
    with open(os.path.join(REPO, 'training/config/train_config.yaml')) as f:
        train_cfg = yaml.safe_load(f)
    if 'label_dict' in config:
        train_cfg['label_dict'] = config['label_dict']
    config.update(train_cfg)
    config.update(dict(
        local_rank=0, track_his=False, kl_reg=False, avu_reg=True, save_model=False,
        device_choice='old', save_path='/tmp', training_data_ratio=1.0,
        max_samples_per_class=13504, detect_novel=False, training_mode='multi',
        ddp=False, test_fre=50, nEpochs=50, task_target='test',
        train_dataset=['DF40_FF_train_NOEFS'], test_dataset=['DF40_FF_test_NOFE_all'],
        num_classes=num_classes))
    config.setdefault('backbone_config', {})['num_classes'] = num_classes
    return config


def build_detector(ckpt, device):
    """Official detector (native fp16 CLIP weights) with the checkpoint loaded, eval mode."""
    from detectors import DETECTOR
    det = DETECTOR['dled'](dled_config())
    sd = torch.load(ckpt, map_location='cpu')
    if not isinstance(sd, dict):
        sd = sd.state_dict()
    missing, unexpected = det.load_state_dict(sd, strict=False)
    # LinearLoRA.forward (train mode) replaces each LoRA'd frozen weight by a plain
    # tensor (merge_lora_param), so after step 1 it is no longer in state_dict and the
    # checkpoint lacks it. Those weights are frozen CLIP: keep the pretrained ones.
    # (FINDINGS F13: the in-training copy drifts slightly through fp16 merge/unmerge.)
    frozen = [k for k in missing if k.endswith('.weight') and k[:-len('weight')] + 'w_lora_A' in sd]
    missing = [k for k in missing if k not in frozen and not k.startswith('loss_func')]
    if missing or unexpected:
        raise RuntimeError(f'checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}')
    n_lora = sum(1 for k in sd if '_lora_' in k)
    print(f'checkpoint : {len(sd)} tensors, {n_lora} LoRA, {len(frozen)} frozen LoRA base weights '
          'taken from pretrained CLIP')
    det.model.device = torch.device(device)
    return det.to(device).eval()


def to_fp32(det):
    """float32 copy for attacking. CustomCLIP/TextEncoder cache dtype at init -> reset it."""
    det = det.float().eval()
    m = det.model
    m.dtype = torch.float32
    m.spatial_text_encoder.dtype = torch.float32
    m.frequency_text_encoder.dtype = torch.float32
    for p in det.parameters():
        p.requires_grad_(False)
    return det


# ---------------------------------------------------------------- thresholds
_THR = re.compile(r"Retrieved uncertainty dict (\{.*\}) !")


def load_thresholds(log_paths):
    """Last 'Retrieved uncertainty dict' in the training log(s): the thresholds the
    official test used for the checkpoint saved at that test epoch."""
    found = None
    for p in log_paths:
        if not os.path.isfile(p):
            continue
        with open(p, errors='replace') as f:
            for line in f:
                m = _THR.search(line)
                if m:
                    found = (ast.literal_eval(m.group(1)), p)
    if found is None:
        raise FileNotFoundError(f'no "Retrieved uncertainty dict" in {log_paths}')
    return found


# ---------------------------------------------------------------- forward / decision
def forward(det, x, quantize='ste'):
    """x: (B,3,224,224) in [0,1] -> fused/branch evidence dict (official forward)."""
    out, _, _ = det.model(ft.dled_inputs(x, quantize))
    return out


def decide(evidence, thr_fused, K):
    """Vectorised copy of model_inference's fused rule. Returns (B,) in {0..K}."""
    tau = torch.tensor([thr_fused[c] for c in range(K)], device=evidence.device, dtype=evidence.dtype)
    e_max, pred = torch.max(evidence, 1)
    u = 1.0 / (1.0 + e_max)
    return torch.where(u > tau[pred], torch.full_like(pred, K), pred)


def success(attack, d, y, K):
    if attack == 'E1':
        return (d < K) & (d != y)
    if attack == 'E2':
        return d == K
    if attack == 'E3':
        return d < K
    if attack in ('E3R', 'E4'):
        return d == 0
    raise ValueError(attack)


# ---------------------------------------------------------------- losses (minimise)
def _accept_loss(z, c, lgate, kappa):
    zc = z.gather(1, c[:, None])[:, 0]
    other = z.scatter(1, c[:, None], float('-inf')).amax(1)
    return F.relu(kappa + other - zc) + F.relu(kappa + lgate[c] - zc)


def attack_loss(attack, evidence, y, thr_fused, K, kappa=0.05):
    z = torch.log1p(evidence.float())
    lgate = -torch.log(torch.tensor([thr_fused[c] for c in range(K)], device=z.device))
    if attack == 'E2':                                  # every class below its own gate
        return F.relu(kappa + z - lgate).sum(1)
    if attack == 'E1':                                  # best wrong known class
        c = z.scatter(1, y.clamp(max=K - 1)[:, None], float('-inf')).argmax(1)
    elif attack == 'E3':                                # easiest known class to pass
        c = (z - lgate).argmax(1)
    else:                                               # E3R, E4: accepted as real
        c = torch.zeros_like(y)
    return _accept_loss(z, c, lgate, kappa)


# ---------------------------------------------------------------- PGD
def pgd(det, x0, y, attack, thr_fused, K, eps, steps=50, alpha=None, random_start=True,
        quantize='ste', kappa=0.05, generator=None):
    """L_inf PGD in [0,1] pixel space of the 224x224 image. eps in [0,1] units.
    Keeps the first successful iterate per sample (else the last), then rounds
    to the 8-bit grid (stays within eps when eps is a multiple of 1/255).
    Returns x_adv (on the grid), first-success step (-1 = never), final loss."""
    alpha = alpha or 2.5 * eps / steps
    # the unperturbed image is always a candidate: an attacker would not perturb an
    # image that already produces the target error (the random start can undo it,
    # because uniform noise changes the FFT spectrum strongly)
    with torch.no_grad():
        ok0 = success(attack, decide(forward(det, x0, quantize)['evidence'], thr_fused, K), y, K)
    if random_start:
        delta = (torch.rand(x0.shape, generator=generator, device='cpu').to(x0) * 2 - 1) * eps
    else:
        delta = torch.zeros_like(x0)
    delta = (x0 + delta).clamp(0, 1) - x0
    best = delta.clone()
    hit = torch.full((x0.shape[0],), -1, dtype=torch.long, device=x0.device)
    hit[ok0] = 0
    best[ok0] = 0.0
    loss = None
    for t in range(steps + 1):
        delta.requires_grad_(True)
        out = forward(det, x0 + delta, quantize)
        loss = attack_loss(attack, out['evidence'], y, thr_fused, K, kappa)
        with torch.no_grad():
            ok = success(attack, decide(out['evidence'], thr_fused, K), y, K)
            new = ok & (hit < 0)
            hit[new] = t
            best[new] = delta.detach()[new]
        if t == steps or bool((hit >= 0).all()):
            break
        g, = torch.autograd.grad(loss.sum(), delta)
        with torch.no_grad():
            delta = delta - alpha * g.sign()
            delta = delta.clamp(-eps, eps)
            delta = (x0 + delta).clamp(0, 1) - x0
    with torch.no_grad():
        fin = torch.where((hit >= 0)[:, None, None, None], best, delta.detach())
        x_adv = torch.round((x0 + fin) * 255.0) / 255.0
    return x_adv, hit, loss.detach()
