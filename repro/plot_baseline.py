"""
Figures for the DLED NOEFS baseline, drawn from the run's own logs.

  python repro/plot_baseline.py --run ~/DLED/runs/<baseline run> \
      [--reload ~/DLED/runs/<reload_check run>/reload_check.json] [--out <folder>]

Writes PNG (300 dpi) + PDF into --out (default: <run>/figures):
  fig1_paper_vs_ours      Acc and DR: paper Table I vs ours (logged in training; reloaded checkpoint)
  fig2_per_class          per-class decision rates, logged vs reloaded
  fig3_training_loss      training losses vs iteration (logged every 300 iterations)
  fig4_training_metrics   training metrics vs iteration
  fig5_learning_rate      learning rate vs step
  fig6_confusion          confusion matrix of the reloaded checkpoint (needs --reload with a confusion matrix)
Each figure is skipped (with a note) when its input is missing. Numbers are only read
from the logs / JSON, never typed in.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PAPER = {'Acc': 75.52, 'DR': 34.71}          # DLED, FG 2026, Table I, NOEFS (EFS unseen)
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'  # categorical slots 1-3 (validated palette)
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e6e5e1'
CLASSES = ['Real', 'Face swap (FS)', 'Face reenact. (FR)', 'Unknown EFS (DR)']

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID,
                     'axes.axisbelow': True, 'figure.dpi': 110, 'savefig.bbox': 'tight'})


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'{name}.{ext}'), dpi=300)
    plt.close(fig)
    print('wrote', os.path.join(out, name + '.png'))


def bars(ax, groups, series, colors, ylim=100):
    """groups: x labels; series: [(name, [values or None])]."""
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, ((name, vals), col) in enumerate(zip(series, colors)):
        pos = x - 0.4 + w / 2 + i * w
        v = [np.nan if a is None else a for a in vals]
        b = ax.bar(pos, v, w * 0.92, color=col, label=name, zorder=2)
        for rect, val in zip(b, v):
            if not np.isnan(val):
                ax.text(rect.get_x() + rect.get_width() / 2, val + ylim * 0.012, f'{val:.1f}',
                        ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, ylim)
    ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, loc='upper right', fontsize=9)


def parse_log(path):
    txt = open(path, errors='replace').read()
    res = {}
    for c in range(4):
        m = re.findall(rf'Class {c} fused accuracy: ([0-9.]+)', txt)
        if m:
            res[c] = 100 * float(m[-1])
    m = re.findall(r'testing-metric, fused_acc: ([0-9.]+)', txt)
    if m:
        res['acc'] = 100 * float(m[-1])
    loss, metric = defaultdict(list), defaultdict(list)
    for line in txt.splitlines():
        it = re.search(r'Iter: (\d+)', line)
        if not it:
            continue
        i = int(it.group(1))
        for k, v in re.findall(r'training-loss, ([\w\-]+): ([-0-9.eE]+)', line):
            loss[k].append((i, float(v)))
        for k, v in re.findall(r'training-metric, ([\w\-]+): ([-0-9.eE]+)', line):
            metric[k].append((i, float(v)))
    lr = [(int(a), float(b)) for a, b in re.findall(r'Fusion Step (\d+) - lr: ([0-9.eE\-]+)', txt)]
    return res, loss, metric, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--reload')
    ap.add_argument('--out')
    a = ap.parse_args()
    run = os.path.expanduser(a.run)
    out = os.path.expanduser(a.out or os.path.join(run, 'figures'))
    os.makedirs(out, exist_ok=True)
    logs = sorted(glob.glob(os.path.join(run, 'DLED_*.log')))
    assert logs, f'no DLED_*.log in {run}'
    res, loss, metric, lr = parse_log(logs[0])
    rel = json.load(open(os.path.expanduser(a.reload))) if a.reload else None
    rr = rel['reload'] if rel else None
    tag = os.path.basename(run.rstrip('/'))

    # fig1 paper vs ours
    if 'acc' in res and 3 in res:
        series = [('Paper (Table I)', [PAPER['Acc'], PAPER['DR']]),
                  ('Ours: logged during training', [res['acc'], res[3]])]
        cols = [C1, C2]
        if rr:
            series.append(('Ours: reloaded checkpoint', [100 * rr['fused_acc'], 100 * rr['class3']]))
            cols.append(C3)
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        bars(ax, ['(K+1)-way accuracy', 'Unknown detection rate (DR)'], series, cols)
        ax.set_ylabel('%')
        ax.set_title('DLED on DF40, NOEFS protocol: accuracy reproduces, unknown detection does not',
                     fontsize=10, color=INK, loc='left')
        fig.text(0.0, -0.04, f'Final epoch (49), single seed 1024. Run {tag}.', fontsize=8, color=INK2)
        save(fig, out, 'fig1_paper_vs_ours')
    else:
        print('fig1 skipped: final test numbers not in log')

    # fig2 per class
    if all(c in res for c in range(4)):
        series = [('Logged during training', [res[c] for c in range(4)])]
        cols = [C2]
        if rr:
            series.append(('Reloaded checkpoint', [100 * rr[f'class{c}'] for c in range(4)]))
            cols.append(C3)
        fig, ax = plt.subplots(figsize=(6.6, 3.6))
        bars(ax, CLASSES, series, cols, ylim=110)
        ax.set_ylabel('% decided as the correct class')
        ax.set_title('Per-class decision rates (unknown EFS: % rejected as unknown)', fontsize=10,
                     color=INK, loc='left')
        ax.tick_params(axis='x', labelsize=8.5)
        save(fig, out, 'fig2_per_class')

    # fig3/4 training curves
    epoch_len = 54016 // 32 + (1 if 54016 % 32 else 0)
    for name, d, ttl in (('fig3_training_loss', loss, 'Training losses'),
                         ('fig4_training_metrics', metric, 'Training metrics')):
        # drop placeholders: DLED logs -1 for metrics it does not compute (e.g. ap/auc in 3-class training)
        keys = [k for k in d if len(d[k]) > 2 and len({v for _, v in d[k]}) > 1]
        if not keys:
            print(f'{name} skipped: no training lines in log')
            continue
        keys = keys[:6]
        fig, axs = plt.subplots(len(keys), 1, figsize=(6.6, 1.7 * len(keys)), sharex=True)
        axs = np.atleast_1d(axs)
        for ax, k in zip(axs, keys):
            it, v = zip(*d[k])
            ax.plot(np.array(it) / epoch_len, v, color=C1, lw=1.5)
            ax.set_ylabel(k, fontsize=8.5)
        axs[-1].set_xlabel('epoch')
        axs[0].set_title(f'{ttl}, mean of each 300-iteration window', fontsize=10, color=INK, loc='left')
        save(fig, out, name)

    # fig5 lr
    if lr:
        s, v = zip(*lr)
        fig, ax = plt.subplots(figsize=(6.2, 2.6))
        ax.plot(np.array(s) / epoch_len, v, color=C1, lw=1.5)
        ax.set_xlabel('epoch'); ax.set_ylabel('learning rate')
        ax.set_title('Learning-rate schedule', fontsize=10, color=INK, loc='left')
        save(fig, out, 'fig5_learning_rate')
    else:
        print('fig5 skipped: no "Fusion Step ... lr" lines')

    # fig6 confusion
    if rel and 'confusion_rows_true_cols_decision' in rel:
        cm = np.array(rel['confusion_rows_true_cols_decision'], dtype=float)
        pct = 100 * cm / cm.sum(1, keepdims=True)
        labels = ['Real', 'FS', 'FR', 'Unknown']
        fig, ax = plt.subplots(figsize=(4.6, 4.0))
        ax.imshow(pct, cmap='Blues', vmin=0, vmax=100)
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f'{pct[i, j]:.1f}%\n({int(cm[i, j])})', ha='center', va='center',
                        fontsize=8, color='white' if pct[i, j] > 55 else INK)
        ax.set_xticks(range(4), labels); ax.set_yticks(range(4), ['Real', 'FS', 'FR', 'EFS (unknown)'])
        ax.set_xlabel('DLED decision'); ax.set_ylabel('true label')
        ax.set_title('Reloaded checkpoint, NOFE_all test (row %)', fontsize=10, color=INK, loc='left')
        save(fig, out, 'fig6_confusion')
    else:
        print('fig6 skipped: pass --reload with a reload_check.json that has a confusion matrix')


if __name__ == '__main__':
    main()
