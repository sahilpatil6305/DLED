"""
CPU check that our differentiable input path for DiCoME (ICML 2026) matches its official one,
on the same DF40 test images used for DLED. Run in venv_dicome, safe next to GPU jobs:

  source ~/DLED/venv_dicome/bin/activate && export HF_HOME=~/DLED/cache/hf
  python repro/detectors/dicome/check.py --samples <attack run>/samples.csv --per_label 4 --out <folder>

Official DiCoME test path (src/dataset/deepfake.py): PIL RGB frame -> CLIPProcessor (resize
shortest edge 224 bicubic, centre crop 224, /255, CLIP mean/std) -> EvidentialFusionModel
-> fused evidence; probability = alpha/S, uncertainty = K/S (K = 2).
Our attack path: x = the 224x224 processor image in [0,1] (8-bit grid) -> CLIP normalisation
(differentiable) -> the same model. Checks:
  1. x recovered from the processor output lies on the 8-bit grid; our normalisation reproduces
     the processor tensor exactly
  2. same fused evidence through both paths
  3. a 224x224 PNG of x re-read by the official processor gives the identical tensor
  4. gradients of uncertainty w.r.t. x are finite and non-zero
Also prints the clean decision for each image (label 0 real, 1 FS, 2 FR, 3 unknown EFS).
"""
import argparse
import csv
import json
import os
import random
import sys
import tempfile

ap = argparse.ArgumentParser()
ap.add_argument('--samples', required=True, help='samples.csv from a run_attack.py run (path,label)')
ap.add_argument('--per_label', type=int, default=4)
ap.add_argument('--repo', default=os.path.expanduser('~/DLED/ext/DiCoME'))
ap.add_argument('--out', default='.')
ap.add_argument('--threads', type=int, default=4)
args = ap.parse_args()
args.samples, args.out = os.path.abspath(args.samples), os.path.abspath(args.out)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

torch.set_num_threads(args.threads)
sys.path.insert(0, args.repo)
os.chdir(args.repo)                                   # DiCoME resolves its config paths from the repo root
from src.config import load_config  # noqa: E402
from src.model.dicome_module import DiCoMEModule  # noqa: E402
from transformers import CLIPProcessor  # noqa: E402

cfg = load_config('src/config/dicome_default.yaml')
module = DiCoMEModule.load_from_checkpoint('weights/dicome-best.ckpt', config=cfg, map_location='cpu', strict=True)
model = module.model.eval().float()
proc = CLIPProcessor.from_pretrained(cfg.backbone)
MEAN = torch.tensor(proc.image_processor.image_mean).view(1, 3, 1, 1)
STD = torch.tensor(proc.image_processor.image_std).view(1, 3, 1, 1)
K = cfg.num_classes


def official(img):
    return proc(images=img, return_tensors='pt')['pixel_values']


def ours(x01):
    return (x01 - MEAN) / STD


def head(pv):
    ev = model(pv)[0].float()
    alpha = ev + 1
    S = alpha.sum(1, keepdim=True)
    return ev, alpha / S, (K / S).squeeze(1)


rows = list(csv.DictReader(open(args.samples)))
random.Random(0).shuffle(rows)
pick = []
for lab in '0123':
    pick += [r for r in rows if r['label'] == lab][:args.per_label]

rep = {'images': [], 'max_pixel_diff': 0.0, 'max_evidence_diff': 0.0, 'grid_err': 0.0,
       'roundtrip_failures': 0}
tmp = tempfile.mkdtemp()
for r in pick:
    img = Image.open(r['path']).convert('RGB')
    pv = official(img)
    x = (pv * STD + MEAN).clamp(0, 1)
    grid_err = float((x * 255 - (x * 255).round()).abs().max())
    x = (x * 255).round() / 255
    with torch.no_grad():
        e1, p1, u1 = head(pv)
        e2, p2, u2 = head(ours(x))
    pd_ = float((ours(x) - pv).abs().max())
    ed = float((e1 - e2).abs().max())
    png = os.path.join(tmp, 'rt.png')
    Image.fromarray((x[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(png)
    rt_ok = bool(torch.equal(official(Image.open(png).convert('RGB')), official(img)) or
                 float((official(Image.open(png).convert('RGB')) - ours(x)).abs().max()) < 1e-5)
    rep['roundtrip_failures'] += int(not rt_ok)
    rep['max_pixel_diff'] = max(rep['max_pixel_diff'], pd_)
    rep['max_evidence_diff'] = max(rep['max_evidence_diff'], ed)
    rep['grid_err'] = max(rep['grid_err'], grid_err)
    rep['images'].append({'label': int(r['label']), 'method': r.get('method'), 'p_fake': float(p1[0, 1]),
                          'uncertainty': float(u1[0]), 'roundtrip_ok': rt_ok})
    print(f"label {r['label']} {r.get('method', ''):14s} p_fake {float(p1[0, 1]):.3f} "
          f"u {float(u1[0]):.4f} | pixel diff {pd_:.1e} evidence diff {ed:.1e} round-trip {rt_ok}")

x.requires_grad_(True)
_, _, u = head(ours(x))
g, = torch.autograd.grad(u.sum(), x)
rep['grad_finite'] = bool(torch.isfinite(g).all())
rep['grad_nonzero_frac'] = float((g != 0).float().mean())
ok = (rep['grid_err'] < 1e-3 and rep['max_pixel_diff'] < 1e-5 and rep['max_evidence_diff'] < 1e-3
      and rep['roundtrip_failures'] == 0 and rep['grad_finite'] and rep['grad_nonzero_frac'] > 0.5)
rep['passed'] = ok
os.makedirs(args.out, exist_ok=True)
json.dump(rep, open(os.path.join(args.out, 'dicome_check.json'), 'w'), indent=1)
print({k: v for k, v in rep.items() if k != 'images'})
print('DICOME CHECK PASSED' if ok else 'DICOME CHECK FAILED')
sys.exit(0 if ok else 1)
