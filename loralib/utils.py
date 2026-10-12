"""
loralib/utils.py
================
LoRA utility functions for DLED (Dual-Level Evidential face forgery Detection).

This module provides the custom apply_lora() function and the index-position
lookup tables used by training/detectors/dled.py.

The function signature is:
    apply_lora(cfg, encoder, indices, dtype) -> list[nn.Module]

where:
  - cfg:      the full DLED config dict (used for TRAINER.LORA_R, LORA_ALPHA)
  - encoder:  a CLIP TextEncoder or Visual encoder module
  - indices:  list of transformer block indices to add LoRA to
  - dtype:    the parameter dtype from the CLIP model (e.g. torch.float16)

This is NOT the same as the upstream MaxZanella/CLIP-LoRA apply_lora(),
which has a different 2-argument signature.
"""

import torch
import torch.nn as nn

from .layers import MultiheadAttentionLoRA, PlainMultiheadAttentionLoRA


# ---------------------------------------------------------------------------
# Transformer block index tables
# (must match the config keys used in train.sh / train_config.yaml)
# ---------------------------------------------------------------------------

INDEX_POSITIONS_TEXT = {
    'top1':        [11],
    'top2':        [10, 11],
    'top3':        [9, 10, 11],
    'bottom':      [0, 1, 2, 3],
    'mid':         [4, 5, 6, 7],
    'up':          [8, 9, 10, 11],
    'half-up':     [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all':         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
}

INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top':         [11],
        'top3':        [9, 10, 11],
        'bottom':      [0, 1, 2, 3],
        'mid':         [4, 5, 6, 7],
        'up':          [8, 9, 10, 11],
        'half-up':     [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all':         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    'ViT-B/32': {
        'bottom':      [0, 1, 2, 3],
        'mid':         [4, 5, 6, 7],
        'up':          [8, 9, 10, 11],
        'half-up':     [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all':         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    },
    'ViT-L/14': {
        'half-up':     list(range(12, 24)),
        'half-bottom': list(range(0, 12)),
        'all':         list(range(0, 24)),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_lora_r(cfg: dict) -> int:
    """Extract LoRA rank from the config dict."""
    try:
        return int(cfg['TRAINER']['LORA_R'])
    except (KeyError, TypeError):
        return 4   # sensible default matching typical CLIP-LoRA setups


def _get_lora_alpha(cfg: dict) -> int:
    """Extract LoRA alpha from the config dict."""
    try:
        return int(cfg['TRAINER']['LORA_ALPHA'])
    except (KeyError, TypeError):
        return 1   # CLIP-LoRA default


def _iter_transformer_blocks(encoder):
    """
    Yield (index, block) for the transformer blocks inside a CLIP encoder.

    CLIP's TextEncoder wraps clip.model.Transformer whose blocks are in
    .resblocks.  The visual ViT encoder's blocks are also in .transformer.resblocks.
    We support both conventions.
    """
    # TextEncoder (custom class wrapping clip.model.Transformer)
    if hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'resblocks'):
        yield from enumerate(encoder.transformer.resblocks)
        return
    # Visual encoder (clip.model.VisualTransformer)
    if hasattr(encoder, 'transformer') and hasattr(encoder.transformer, 'resblocks'):
        yield from enumerate(encoder.transformer.resblocks)
        return
    # CLIP visual encoder stores blocks directly in .resblocks
    if hasattr(encoder, 'resblocks'):
        yield from enumerate(encoder.resblocks)
        return
    # Fallback: recurse one level
    for attr in ['blocks', 'layers']:
        if hasattr(encoder, attr):
            yield from enumerate(getattr(encoder, attr))
            return
    raise ValueError(
        f"Cannot find transformer blocks in encoder of type {type(encoder).__name__}. "
        "Expected one of: .transformer.resblocks, .resblocks, .blocks, .layers"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_lora(cfg: dict, encoder: nn.Module, indices: list, dtype=None) -> list:
    """
    Replace nn.MultiheadAttention modules in the specified transformer block
    indices with LoRA-augmented versions.

    Parameters
    ----------
    cfg : dict
        DLED training config.  Must contain cfg['TRAINER']['LORA_R'] and
        optionally cfg['TRAINER']['LORA_ALPHA'].
    encoder : nn.Module
        A CLIP TextEncoder or Visual encoder whose transformer blocks will be
        modified in-place at the given indices.
    indices : list[int]
        Transformer block indices at which to inject LoRA layers.
    dtype : torch.dtype or None
        The CLIP model's parameter dtype.  LoRA parameters are cast to this
        dtype so they participate in mixed-precision training correctly.

    Returns
    -------
    list[MultiheadAttentionLoRA]
        The newly created LoRA layers (in block-index order).  The caller
        should extend its lora_layers list with this for parameter tracking.
    """
    r = _get_lora_r(cfg)
    alpha = _get_lora_alpha(cfg)
    lora_layers = []

    for block_idx, block in _iter_transformer_blocks(encoder):
        if block_idx not in indices:
            continue
        print(f"  [apply_lora] block {block_idx}: injecting LoRA (r={r}, alpha={alpha})")

        # Find and replace MultiheadAttention inside this residual block
        for child_name, child_module in list(block.named_children()):
            if isinstance(child_module, nn.MultiheadAttention):
                lora_attn = MultiheadAttentionLoRA(
                    child_module, r=r, lora_alpha=alpha, dtype=dtype
                )
                setattr(block, child_name, lora_attn)
                lora_layers.append(lora_attn)
                print(f"    Replaced {child_name} in block {block_idx} with MultiheadAttentionLoRA")

            # CLIP uses 'attn' attribute inside ResidualAttentionBlock
            elif hasattr(child_module, 'attn') and isinstance(
                    getattr(child_module, 'attn', None), nn.MultiheadAttention):
                orig_attn = child_module.attn
                lora_attn = MultiheadAttentionLoRA(
                    orig_attn, r=r, lora_alpha=alpha, dtype=dtype
                )
                child_module.attn = lora_attn
                lora_layers.append(lora_attn)
                print(f"    Replaced .attn inside {child_name} in block {block_idx}")

        # CLIP ResidualAttentionBlock stores attn directly on the block
        if hasattr(block, 'attn') and isinstance(block.attn, nn.MultiheadAttention):
            lora_attn = MultiheadAttentionLoRA(
                block.attn, r=r, lora_alpha=alpha, dtype=dtype
            )
            block.attn = lora_attn
            lora_layers.append(lora_attn)
            print(f"    Replaced block.attn in block {block_idx}")

    if not lora_layers:
        print(f"  WARNING: apply_lora found no MultiheadAttention modules at indices {indices}. "
              "Check encoder structure.")

    return lora_layers


def apply_double_lora(cfg: dict, encoder_a: nn.Module, encoder_b: nn.Module,
                      indices: list, dtype=None):
    """
    Apply LoRA to two encoders at the same indices.
    Imported by dled.py but not called — kept for API compatibility.
    """
    layers_a = apply_lora(cfg, encoder_a, indices, dtype)
    layers_b = apply_lora(cfg, encoder_b, indices, dtype)
    return layers_a, layers_b


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    """Freeze all non-LoRA parameters."""
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, MultiheadAttentionLoRA):
                if hasattr(m, 'in_proj_bias') and m.in_proj_bias is not None:
                    m.in_proj_bias.requires_grad = True
    else:
        raise NotImplementedError(f"bias mode '{bias}' not supported")


def lora_state_dict(model: nn.Module, bias: str = 'none'):
    """Return only LoRA-related state dict entries."""
    sd = model.state_dict()
    if bias == 'none':
        return {k: v for k, v in sd.items() if 'lora_' in k}
    elif bias == 'all':
        return {k: v for k, v in sd.items() if 'lora_' in k or 'bias' in k}
    else:
        raise NotImplementedError


def get_lora_parameters(model: nn.Module, bias: str = 'none'):
    """Return LoRA parameter tensors for optimizers."""
    params = []
    for name, param in model.named_parameters():
        if 'lora_' in name:
            params.append(param)
            if bias == 'lora_only':
                bias_name = name.split('lora_')[0] + 'bias'
                sd = dict(model.named_parameters())
                if bias_name in sd:
                    params.append(sd[bias_name])
    return params
