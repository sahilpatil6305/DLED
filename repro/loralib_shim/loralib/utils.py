"""
RECONSTRUCTED loralib.utils for official DLED (zhyczy/DLED @ b8c41ea).

WHY THIS EXISTS
  training/detectors/dled.py line 30 imports:
      from loralib.utils import apply_lora, apply_double_lora,
                                INDEX_POSITIONS_TEXT, INDEX_POSITIONS_VISION
  The official repo ships no `loralib/` package. The PyPI `loralib` (Microsoft)
  has none of these names. The only public match is CLIP-LoRA
  (github.com/MaxZanella/CLIP-LoRA, AGPL-3.0), whose apply_lora has a
  DIFFERENT signature: apply_lora(args, clip_model).

  DLED calls:  apply_lora(cfg, encoder, indices, dtype)
    - encoder: either TextEncoder(clip_model) or clip_model.visual;
               both expose `.transformer.resblocks`
    - indices: block indices, from INDEX_POSITIONS_* below
    - dtype:   CLIP's dtype (fp16 when TRAINER.PREC == "fp16")

  This file adapts CLIP-LoRA's PlainMultiheadAttentionLoRA (layers.py,
  copied unmodified) to that call signature. It is a DOCUMENTED DEVIATION
  from the official release and must be reported as such.

ASSUMPTIONS (cannot be verified without the authors' file)
  A1. Layer implementation = CLIP-LoRA PlainMultiheadAttentionLoRA.
  A2. LoRA scaling = CLIP-LoRA's lora_alpha / sqrt(r)  (= 0.25 for alpha=1,r=16),
      NOT Microsoft loralib's alpha / r (= 0.0625).
  A3. The replaced attention module is cast to `dtype`, so fp16 CLIP keeps
      working without autocast (DLED's fp16 path uses no autocast/GradScaler).
  A4. apply_double_lora is imported but never called by DLED; it is defined
      only so the import succeeds.

SANITY CHECK
  With ENABLE_LORA=[q,k,v,o], RANK=16, POSITION='all', ViT-B/16 on 4 encoders
  (spatial/frequency x text/vision): 48 replaced modules and
  2*12*4*2*16*768 + 2*12*4*2*16*512 = 3,932,160 LoRA parameters.
"""
import torch.nn as nn

from .layers import PlainMultiheadAttentionLoRA

# Copied verbatim from CLIP-LoRA loralib/utils.py
INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-L/14': {
        'half-up': list(range(12, 24)),
        'half-bottom': list(range(0, 12)),
        'all': list(range(0, 24))},
}


def _resblocks(encoder):
    if hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'resblocks'):
        return encoder.transformer.resblocks
    raise AttributeError(
        f'{type(encoder).__name__} has no .transformer.resblocks; '
        'apply_lora expects TextEncoder or CLIP .visual')


def apply_lora(cfg, encoder, indices, dtype=None):
    t = cfg['TRAINER']
    enable = list(t['ENABLE_LORA'])
    r = int(t['RANK'])
    alpha = t['LORA_ALPHA']
    dropout = float(t.get('DROPOUT_RATE', 0) or 0)

    layers = []
    for i, block in enumerate(_resblocks(encoder)):
        if i not in indices:
            continue
        for name, sub in block.named_children():
            if isinstance(sub, nn.MultiheadAttention):
                new = PlainMultiheadAttentionLoRA(
                    sub, enable_lora=enable, r=r,
                    lora_alpha=alpha, dropout_rate=dropout)
                if dtype is not None:
                    new = new.to(dtype)
                setattr(block, name, new)
                layers.append(new)
    return layers


def apply_double_lora(*args, **kwargs):
    raise NotImplementedError(
        'apply_double_lora is imported by dled.py but never called; '
        'its original implementation was not released.')


def count_lora_params(layers):
    return sum(p.numel() for l in layers
               for n, p in l.named_parameters() if 'lora_' in n)
