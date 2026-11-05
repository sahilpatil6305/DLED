"""
Mechanism analysis: why does DLED's reject option break before its classification?

For each clean image (the same paired list as the attack runs) we measure, in DLED's own decision
space z = log(1 + fused evidence) with gate l_c = -log tau_c (accept as c <=> c = argmax z and
z_c >= l_c, FINDINGS F12):
  m_rej = z_c - l_c                 how far the top evidence sits above its rejection gate
  m_cls = z_c - max_{j != c} z_j    how far the top class sits above the runner-up
and the first-order L-inf distance to each boundary, d = m / ||grad_x m||_1, in 1/255 units
(the L-inf budget a linearised attacker needs to cross it). Hypothesis H1: d_rej < d_cls for most
correctly accepted known images, which predicts E2 (known -> rejected) succeeding at smaller eps
than E1 (known -> wrong known class). With --attack_run, the minimal eps at which E1 / E2 actually
succeeded (OFFICIAL, per image) is merged in, so the prediction can be checked per image.

Analysis only: no attack, no change to any decision. Safe on CPU next to GPU jobs:
  python repro/attack/margins.py --ckpt <ckpt> --log <log with thresholds> \
      --samples_csv ~/DLED/runs/20261026-001437_paper_margin/samples.csv --n_per_class 100 \
      --device cpu --threads 2 --out <folder> [--attack_run <final margin run> <final ce run>]
Outputs: margins.csv (per image), margins_summary.txt
"""
import argparse
import csv
import os
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True)
ap.add_argument('--log', nargs='+', required=True, help='log file(s) with "Retrieved uncertainty dict"')
ap.add_argument('--samples_csv', required=True)
ap.add_argument('--n_per_class', type=int, default=100)
ap.add_argument('--labels', nargs='+', type=int, default=[0, 1, 2], help='known labels to analyse')
ap.add_argument('--attack_run', nargs='*', default=[], help='run folders whose per_sample.csv gives min success eps')
ap.add_argument('--batch', type=int, default=4)
ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
ap.add_argument('--threads', type=int, default=2)
ap.add_argument('--out', required=True)
args = ap.parse_args()
if args.device == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch  # noqa: E402

torch.set_num_threads(args.threads)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dled_attack as da  # noqa: E402
import freq_torch as ft  # noqa: E402

K = 3
dev = torch.device(args.device)
os.makedirs(args.out, exist_ok=True)

thr, src = da.load_thresholds(args.log)
thr_f = thr['fused']
lgate = -torch.log(torch.tensor([thr_f[c] for c in range(K)], dtype=torch.float32, device=dev))
det = da.to_fp32(da.build_detector(os.path.expanduser(args.ckpt), args.device,
                                   prec='fp32' if args.device == 'cpu' else None))
det.model.eval()

# the same image list and idx layout as run_attack.py --samples_csv (label-major, first n per label)
per = {}
for r in csv.DictReader(open(os.path.expanduser(args.samples_csv))):
    lab = int(r['label'])
    if len(per.setdefault(lab, [])) < args.n_per_class:
        per[lab].append((r['path'], lab, r['method']))
samples = []
for lab in range(K + 1):
    samples += per.get(lab, [])
todo = [i for i, s in enumerate(samples) if s[1] in args.labels]


def margins(x):
    ev = da.forward(det, x, 'ste')['evidence'].float()
    z = torch.log1p(ev)
    c = z.argmax(1)
    zc = z.gather(1, c[:, None])[:, 0]
    second = z.scatter(1, c[:, None], float('-inf')).amax(1)
    return z, c, zc - lgate[c], zc - second


rows = []
for k in range(0, len(todo), args.batch):
    b = todo[k:k + args.batch]
    x = torch.stack([ft.load_resized(samples[i][0]) for i in b]).to(dev).requires_grad_(True)
    z, c, m_rej, m_cls = margins(x)
    g_rej, = torch.autograd.grad(m_rej.sum(), x, retain_graph=True)
    g_cls, = torch.autograd.grad(m_cls.sum(), x)
    with torch.no_grad():
        dec = da.decide(torch.expm1(z), thr_f, K)
        n_rej, n_cls = g_rej.abs().flatten(1).sum(1), g_cls.abs().flatten(1).sum(1)
        d_rej = 255.0 * m_rej / n_rej.clamp_min(1e-12)
        d_cls = 255.0 * m_cls / n_cls.clamp_min(1e-12)
    for j, i in enumerate(b):
        rows.append({'idx': i, 'label': samples[i][1], 'method': samples[i][2], 'clean_dec': int(dec[j]),
                     'top_class': int(c[j]), 'z_top': float(z[j, c[j]]), 'gate': float(lgate[c[j]]),
                     'm_rej': float(m_rej[j]), 'm_cls': float(m_cls[j]),
                     'grad_l1_rej': float(n_rej[j]), 'grad_l1_cls': float(n_cls[j]),
                     'd_rej_255': float(d_rej[j]), 'd_cls_255': float(d_cls[j])})
    print(f'{min(k + args.batch, len(todo))}/{len(todo)}', flush=True)

# minimal OFFICIAL success eps per image and attack, over all given runs (worst case)
mins = {}
for run in args.attack_run:
    for r in csv.DictReader(open(os.path.join(os.path.expanduser(run), 'per_sample.csv'))):
        if int(float(r['success_official'])):
            key = (r['attack'], int(r['idx']))
            mins[key] = min(mins.get(key, 1e9), float(r['eps']))
for r in rows:
    for a in ('E1', 'E2'):
        r[f'min_eps_{a}'] = mins.get((a, r['idx']), '') if args.attack_run else ''

with open(os.path.join(args.out, 'margins.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

# ------------------------------------------------------------------ summary
lines = [f'checkpoint {args.ckpt}', f'thresholds {src}  fused tau {thr_f}',
         f'images analysed {len(rows)} (labels {args.labels})']
ok = [r for r in rows if r['clean_dec'] == r['label']]          # correctly accepted known images
lines.append(f'correctly accepted (fp32 copy decision; official agrees on >=99% of clean images): {len(ok)}')


def q(v):
    v = sorted(v)
    return f'median {statistics.median(v):.3f}  [q25 {v[len(v) // 4]:.3f}, q75 {v[(3 * len(v)) // 4]:.3f}]' if v else 'n/a'


for name in ('m_rej', 'm_cls', 'd_rej_255', 'd_cls_255'):
    lines.append(f'{name:10s} {q([r[name] for r in ok])}')
closer = sum(1 for r in ok if r['d_rej_255'] < r['d_cls_255'])
lines.append(f'H1: rejection boundary closer than class boundary (d_rej < d_cls): {closer}/{len(ok)}'
             f' = {100 * closer / max(len(ok), 1):.1f}%')
for lab in args.labels:
    sub = [r for r in ok if r['label'] == lab]
    if sub:
        cl = sum(1 for r in sub if r['d_rej_255'] < r['d_cls_255'])
        lines.append(f'  label {lab}: {cl}/{len(sub)}  d_rej {q([r["d_rej_255"] for r in sub])}'
                     f'  d_cls {q([r["d_cls_255"] for r in sub])}')
if args.attack_run:
    both = [r for r in ok if r['min_eps_E1'] != '' and r['min_eps_E2'] != '']
    e2_first = sum(1 for r in both if r['min_eps_E2'] < r['min_eps_E1'])
    tie = sum(1 for r in both if r['min_eps_E2'] == r['min_eps_E1'])
    lines.append(f'attacks (OFFICIAL, min eps over {len(args.attack_run)} run(s)): both E1 and E2 succeeded for '
                 f'{len(both)}; E2 at smaller eps {e2_first}, same eps {tie}, E1 smaller {len(both) - e2_first - tie}')
    pred = [r for r in both if r['min_eps_E2'] != r['min_eps_E1']]
    agree = sum(1 for r in pred if (r['d_rej_255'] < r['d_cls_255']) == (r['min_eps_E2'] < r['min_eps_E1']))
    lines.append(f'per-image agreement of the d_rej<d_cls prediction with which attack succeeded first: '
                 f'{agree}/{len(pred)}')
open(os.path.join(args.out, 'margins_summary.txt'), 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
print('MARGINS DONE ->', args.out)
