"""
Reload fidelity check (FINDINGS F13): does the SAVED checkpoint reproduce the
numbers DLED logged during training?

The in-training test ran on frozen CLIP weights that had drifted through fp16
LoRA merge/unmerge; the checkpoint does not contain those weights, so a reload
uses pristine CLIP. This script:
  1. rebuilds the exact in-training test set: same config as train.py with
     train.sh NOEFS args, same seed (1024), train set built first (it consumes
     the RNG before the test set is balanced, as in train.py)
  2. loads the checkpoint, decides with the official model_inference and the
     thresholds logged for that checkpoint
  3. compares (K+1)-way accuracy and per-class rates with the training log, and
     per-sample with the saved metric_dict_best.pickle when the order matches

  repro/run.sh reload_check python repro/attack/eval_reload.py --run ~/DLED/runs/<baseline run>
GPU, about 11 minutes for 81,024 images (batch 100, as in training).
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dled_attack as da  # noqa: E402
import train as T  # noqa: E402  (training/train.py; main() is guarded)

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)
ap.add_argument('--ckpt')
ap.add_argument('--batch', type=int, default=None, help='default: test_batchSize from the config')
ap.add_argument('--out', default=os.environ.get('DLED_LOG_DIR', '.'))
args = ap.parse_args()
run = os.path.expanduser(args.run)
K = 3

# ---- checkpoint, its metrics pickle, logs
ckpts = [args.ckpt] if args.ckpt else glob.glob(os.path.join(run, 'dled_logs*', 'test', 'DF40_FF_test_NOFE_all', 'ckpt_best.pth'))
if not ckpts and os.path.islink(os.path.join(run, 'dled_logs')):
    ckpts = glob.glob(os.path.join(os.path.dirname(os.path.realpath(os.path.join(run, 'dled_logs'))),
                                   '*', 'test', 'DF40_FF_test_NOFE_all', 'ckpt_best.pth'))
    ckpts = [c for c in ckpts if os.path.getmtime(c) >= os.path.getmtime(os.path.join(run, 'meta.txt'))]
assert len(ckpts) == 1, f'need exactly one checkpoint, found {ckpts}; pass --ckpt'
ckpt = ckpts[0]
logs = sorted(glob.glob(os.path.join(run, 'dled_logs*', 'training.log'))) + sorted(glob.glob(os.path.join(run, 'DLED_*.log')))
thr, thr_src = da.load_thresholds(logs)
print('checkpoint :', ckpt)
print('thresholds :', thr_src, '| fused', {k: round(v, 5) for k, v in thr['fused'].items()})

logged = {}
with open([l for l in logs if os.path.basename(l).startswith('DLED_')][-1], errors='replace') as f:
    txt = f.read()
for c in range(K + 1):
    m = re.findall(rf'Class {c} fused accuracy: ([0-9.]+)', txt)
    if m:
        logged[f'class{c}'] = float(m[-1])
m = re.findall(r'testing-metric, fused_acc: ([0-9.]+)', txt)
if m:
    logged['fused_acc'] = float(m[-1])

# ---- config exactly as train.py main() with train.sh NOEFS arguments
config = da.dled_config()
config.update(track_his=False, kl_reg=False, avu_reg=True, save_model=True, device_choice='old',
              training_data_ratio=1.0, max_samples_per_class=13504, detect_novel=False,
              num_classes=K, training_mode='multi', ddp=False, test_fre=50, nEpochs=50,
              task_target='test', local_rank=0)
torch.cuda.set_device(0)
T.init_seed(config)                       # seed 1024, as in training
_ = T.prepare_training_data(config)       # consumes the RNG exactly as in training
test_loader = T.prepare_testing_data(config)['DF40_FF_test_NOFE_all']
if args.batch:
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=args.batch, shuffle=False,
                                              num_workers=test_loader.num_workers,
                                              collate_fn=test_loader.dataset.collate_fn)
print('test images:', len(test_loader.dataset))

det = da.build_detector(ckpt, 'cuda')     # native fp16, as the official test
preds, labels = [], []
with torch.no_grad():
    for dd in tqdm(test_loader, desc='reload test'):
        dd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in dd.items()}
        p = det.model_inference(dd, thr)['prob']['fused']
        preds.append(p.cpu().numpy()); labels.append(dd['label'].cpu().numpy())
preds, labels = np.concatenate(preds), np.concatenate(labels)

os.makedirs(args.out, exist_ok=True)
np.save(os.path.join(args.out, 'reload_preds.npy'), preds)
np.save(os.path.join(args.out, 'reload_labels.npy'), labels)
conf = np.zeros((K + 1, K + 1), dtype=int)                 # rows: true label, cols: decision
for t, p in zip(labels, preds):
    conf[int(t), int(p)] += 1
res = {'ckpt': ckpt, 'thresholds_from': thr_src, 'fused_tau': {str(k): v for k, v in thr['fused'].items()},
       'n': int(len(labels)),
       'reload': {'fused_acc': float(np.mean(preds == labels)),
                  'binary_acc': float(np.mean((preds != 0) == (labels != 0))),
                  **{f'class{c}': float(np.mean(preds[labels == c] == c)) for c in range(K + 1)}},
       'confusion_rows_true_cols_decision': conf.tolist(),
       'clean_E3R_unknown_accepted_as_real': float(np.mean(preds[labels == K] == 0)),
       'clean_E3_unknown_accepted_any_known': float(np.mean(preds[labels == K] < K)),
       'logged': logged}
pk = os.path.join(os.path.dirname(ckpt), 'metric_dict_best.pickle')
if os.path.isfile(pk):
    # the pickle stores BINARY labels and predictions (0 real, 1 fake) in test-loader order
    with open(pk, 'rb') as f:
        md = pickle.load(f)
    lp, ll = np.asarray(md.get('fused_pred')).astype(int), np.asarray(md.get('label')).astype(int)
    res['pickle_keys'] = sorted(md.keys())
    res['pickle_label_values'] = sorted(set(ll.tolist()))[:10]
    if lp.shape == preds.shape and ll.shape == labels.shape:
        bl, bp = (labels != 0).astype(int), (preds != 0).astype(int)
        res['same_binary_label_order_as_training'] = bool(np.array_equal(ll, bl))
        res['binary_label_match_rate'] = float(np.mean(ll == bl))
        res['per_sample_binary_agreement_with_training'] = float(np.mean(lp == bp))
        res['training_binary_acc'] = float(np.mean(lp == ll))
        for c in range(K + 1):                              # per true class: P(predicted fake), training vs reload
            m = labels == c
            if res['same_binary_label_order_as_training']:
                res[f'class{c}_pred_fake_training'] = float(np.mean(lp[m] == 1))
                res[f'class{c}_pred_fake_reload'] = float(np.mean(bp[m] == 1))
    else:
        res['pickle_note'] = f'shapes differ: pickle pred {lp.shape} label {ll.shape} vs reload {preds.shape}'

print(json.dumps(res, indent=1))
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, 'reload_check.json'), 'w') as f:
    json.dump(res, f, indent=1)
gap = {k: round(100 * (res['reload'][k] - v), 2) for k, v in logged.items() if k in res['reload']}
print('reload minus logged (percentage points):', gap)
