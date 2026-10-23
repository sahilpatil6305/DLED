"""
Setup smoke test for official DLED -- NOT an experiment.

Builds the detector exactly as training/train.py does (same YAML merge),
then runs ONE forward + loss + backward on random tensors on the GPU.
Needs no dataset. Checks:
  1. all imports resolve (incl. the reconstructed loralib shim)
  2. 48 LoRA attention modules, 3,932,160 LoRA params (rank16, qkvo, all blocks)
  3. only prompt learners + LoRA params are trainable
  4. forward produces spatial / frequency / fused evidence and vacuity
  5. loss is finite and LoRA grads are non-zero

Run from the repo root (see repro/README.md):
  source repro/env.sh
  python repro/smoke_model.py
"""
import os
import sys
import yaml
import torch

ROOT = os.getcwd()
assert os.path.isfile(os.path.join(ROOT, 'train.sh')), 'run from the DLED repo root'
sys.path[:0] = [ROOT, os.path.join(ROOT, 'training')]

import loralib  # noqa: E402
print('loralib from:', loralib.__file__)
assert 'loralib_shim' in loralib.__file__, 'wrong loralib picked up (PyPI one?)'

from detectors import DETECTOR  # noqa: E402

PROTOCOL = dict(train_dataset=['DF40_FF_train_NOEFS'],
                test_dataset=['DF40_FF_test_NOFE_all'], num_classes=3)

with open('training/config/detector/dled.yaml') as f:
    config = yaml.safe_load(f)
with open('training/config/train_config.yaml') as f:
    train_cfg = yaml.safe_load(f)
if 'label_dict' in config:
    train_cfg['label_dict'] = config['label_dict']
config.update(train_cfg)
config.update(dict(
    local_rank=0, track_his=False, kl_reg=False, avu_reg=True, save_model=False,
    device_choice='old', save_path='/tmp', training_data_ratio=1.0,
    max_samples_per_class=13504, detect_novel=False, training_mode='multi',
    ddp=False, test_fre=50, nEpochs=50, task_target='test', **PROTOCOL))
config.setdefault('backbone_config', {})['num_classes'] = PROTOCOL['num_classes']

assert torch.cuda.is_available(), 'CUDA not available'
torch.cuda.set_device(0)
torch.manual_seed(0)
print('torch', torch.__version__, '| GPU', torch.cuda.get_device_name(0),
      '| PREC', config['TRAINER']['PREC'])

model = DETECTOR['dled'](config).cuda()
clip_model = model.model

# --- 2. LoRA structure
from loralib.layers import PlainMultiheadAttentionLoRA  # noqa: E402
n_mod = sum(isinstance(m, PlainMultiheadAttentionLoRA) for m in clip_model.modules())
n_lora = sum(p.numel() for n, p in clip_model.named_parameters() if 'lora_' in n)
print(f'LoRA modules: {n_mod} (expect 48) | LoRA params: {n_lora:,} (expect 3,932,160)')

# --- 3. trainable set
trainable = [n for n, p in clip_model.named_parameters() if p.requires_grad]
bad = [n for n in trainable if not ('prompt_learner' in n or 'lora_' in n)]
n_train = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
print(f'trainable tensors: {len(trainable)} | trainable params: {n_train:,} | unexpected: {bad[:5]}')

# --- 4/5. forward + backward on random input (shape/flow check only)
B = 2
dummy = {
    'image': torch.randn(B, 3, 224, 224).cuda(),
    'freq': torch.randn(B, 3, 224, 224).cuda(),
    'high': torch.randn(B, 3, 224, 224).cuda(),
    'label': torch.tensor([0, 2]).cuda(),
    'landmark': None, 'mask': None,
}
model.train()
pred = model(dummy, 'fusion')
for k in ['spatial_pred_alpha', 'frequency_pred_alpha', 'pred_alpha',
          'spatial_vacuity', 'frequency_vacuity', 'vacuity']:
    v = pred.get(k)
    print(f'  {k:22s}', tuple(v.shape) if torch.is_tensor(v) else v)
pred.update({'epoch': 0, 'total_epoch': 50, 'training': True})
losses = model.get_losses(dummy, pred, mode='train')
loss = losses['overall']
print('loss:', float(loss), '| finite:', bool(torch.isfinite(loss)))
loss.backward()
gnorm = sum(float(p.grad.float().norm()) for n, p in clip_model.named_parameters()
            if 'lora_B' in n and p.grad is not None)
print('sum |grad| over lora_B:', gnorm, '(should be > 0)')
print('peak GPU memory (batch 2): %.2f GiB' % (torch.cuda.max_memory_allocated() / 2**30))

ok = n_mod == 48 and n_lora == 3_932_160 and not bad and torch.isfinite(loss) and gnorm > 0
print('\nSMOKE TEST', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
