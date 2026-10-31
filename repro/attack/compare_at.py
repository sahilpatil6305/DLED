"""Paired comparison of the adversarially fine-tuned DLED models (CPU only, standard library).

All AT attack runs used --samples_csv <paper_margin>/samples.csv --n_per_class 100, so they attacked the
SAME images in the same order: row idx i is the same image in every run. This script
  * reads per_sample.csv of the margin run and the CE run of each model,
  * reports OFFICIAL success per attack/eps for margin, CE and the WORST CASE (a sample counts as
    broken if either attack broke it; the honest robustness number),
  * adds a baseline row from paper_margin restricted to the same images (margin only: paper_ce used
    other images, so the baseline has no worst case),
  * runs exact two-sided McNemar tests (paired) on the worst case: at_reject vs every other model.

Usage (VM):  python repro/attack/compare_at.py --runs ~/DLED/runs --out ~/DLED/runs/at_compare.txt
"""
import argparse, csv, glob, math, os, sys
from collections import defaultdict

p = argparse.ArgumentParser()
p.add_argument('--runs', default='~/DLED/runs')
p.add_argument('--models', nargs='+', default=['ft_clean', 'at_pgd', 'at_evat', 'at_reject'])
p.add_argument('--ref', default='at_reject', help='model compared against all others with McNemar')
p.add_argument('--baseline', default='paper_margin')
p.add_argument('--n_per_class', type=int, default=100)
p.add_argument('--out', default=None)
args = p.parse_args()
R = os.path.expanduser(args.runs)
out_lines = []


def say(s=''):
    print(s); out_lines.append(s)


def latest(pattern):
    c = sorted(d for d in glob.glob(os.path.join(R, pattern)) if os.path.isdir(d))
    return c[-1] if c else None


def load(run):
    """{(attack, eps, idx): (success_official, (label, method))}"""
    d = {}
    for r in csv.DictReader(open(os.path.join(run, 'per_sample.csv'))):
        d[(r['attack'], float(r['eps']), int(r['idx']))] = (int(float(r['success_official'])), (int(r['label']), r['method']))
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 2
    ph = k / n; den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den; h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def mcnemar(b, c):
    """exact two-sided binomial p for discordant counts b, c"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


# ---------------------------------------------------------------- load runs
data, src = {}, {}
for m in args.models:
    for loss, suf in (('margin', 'attack'), ('ce', 'attack_ce')):
        run = latest(f'*_{m}_{suf}')
        if run is None:
            sys.exit(f'missing run *_{m}_{suf} in {R}')
        data[(m, loss)] = load(run); src[(m, loss)] = os.path.basename(run)

base_run = latest(f'*_{args.baseline}')
if base_run:
    # paper_margin: per label 300 images, samples.csv order, idx = label*300 + rank. The AT runs took the
    # first n_per_class of each label: AT idx = label*n + rank. Build the map and verify the labels match.
    raw = load(base_run)
    n_base = max(k[2] for k in raw) + 1
    per_label = n_base // 4
    bmap = {}
    for (a, e, i), (s, lab) in raw.items():
        lab_i, rank = divmod(i, per_label)
        assert lab_i == lab[0], 'paper_margin idx layout is not label-major'
        if rank < args.n_per_class:
            bmap[(a, e, lab_i * args.n_per_class + rank)] = (s, lab)
    data[('baseline', 'margin')] = bmap; src[('baseline', 'margin')] = os.path.basename(base_run)

say('sources:')
for k, v in src.items():
    say(f'  {k[0]:10s} {k[1]:6s} {v}')

keys = sorted({k for k in data[(args.models[0], 'margin')]})
attacks = sorted({k[0] for k in keys}, key=lambda a: ['E1', 'E2', 'E3', 'E3R', 'E4'].index(a)
                 if a in ['E1', 'E2', 'E3', 'E3R', 'E4'] else 99)
epss = sorted({k[1] for k in keys})

# sanity: same images (labels per idx identical across all runs)
bad = 0
for k in keys:
    labs = {d[k][1] for d in data.values() if k in d}
    bad += len(labs) > 1
say(f'pairing check: {len(keys)} (attack,eps,idx) cells, label/generator mismatches across runs = {bad}')
if bad:
    say('  WARNING: runs are NOT paired; do not trust the McNemar tests below')


def worst(m, k):
    return max(data[(m, 'margin')][k][0], data[(m, 'ce')][k][0])


rows_models = (['baseline'] if ('baseline', 'margin') in data else []) + args.models
for a in attacks:
    say(); say(f'=== {a}   OFFICIAL success %  (margin / CE / WORST [95% CI])')
    say(f'{"model":10s} ' + ' '.join(f'{"eps "+format(e, "g"):>27s}' for e in epss))
    for m in rows_models:
        cells = []
        for e in epss:
            ks = [k for k in keys if k[0] == a and k[1] == e]
            n = len(ks)
            if m == 'baseline':
                km = sum(data[(m, 'margin')][k][0] for k in ks if k in data[(m, 'margin')])
                cells.append(f'{100*km/n:5.1f} /   -  /   -          ')
            else:
                km = sum(data[(m, 'margin')][k][0] for k in ks)
                kc = sum(data[(m, 'ce')][k][0] for k in ks)
                kw = sum(worst(m, k) for k in ks)
                lo, hi = wilson(kw, n)
                cells.append(f'{100*km/n:5.1f}/{100*kc/n:5.1f}/{100*kw/n:5.1f}[{100*lo:3.0f},{100*hi:3.0f}]')
        say(f'{m:10s} ' + ' '.join(f'{c:>27s}' for c in cells))
    say(f'  n per cell = {len([k for k in keys if k[0] == a and k[1] == epss[0]])}')

say(); say(f'=== McNemar (exact, paired, WORST case): {args.ref} vs others')
say('  d = ref% - other%  (negative = ref is MORE robust). b = only ref broken, c = only other broken.')
for other in [m for m in args.models if m != args.ref]:
    say(f'-- {args.ref} vs {other}')
    for a in attacks:
        parts = []
        for e in epss:
            ks = [k for k in keys if k[0] == a and k[1] == e]
            b = sum(1 for k in ks if worst(args.ref, k) and not worst(other, k))
            c = sum(1 for k in ks if worst(other, k) and not worst(args.ref, k))
            d = 100 * (b - c) / len(ks)
            pv = mcnemar(b, c)
            star = '**' if pv < 0.01 else '*' if pv < 0.05 else '  '
            parts.append(f'eps{e:g}: d={d:+5.1f} p={pv:.3f}{star}')
        say(f'   {a:4s} ' + ' | '.join(parts))

if args.out:
    with open(os.path.expanduser(args.out), 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f'\nwritten {args.out}')
