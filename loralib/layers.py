"""
loralib/layers.py
=================
LoRA layer implementations compatible with CLIP's MultiheadAttention.
Based on CLIP-LoRA (MaxZanella/CLIP-LoRA), adapted for the DLED project.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Base LoRA mixin
# ---------------------------------------------------------------------------

class LoRALayer:
    def __init__(self, r: int, lora_alpha: int, fan_in_fan_out: bool = False,
                 dropout_rate: float = 0.0):
        self.r = r
        self.lora_alpha = lora_alpha
        self.dropout_rate = dropout_rate
        if r > 0:
            self.scaling = lora_alpha / math.sqrt(r)
        self.merged = False
        self.fan_in_fan_out = fan_in_fan_out
        self.params_with_lora = {}

    def register_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            w = getattr(self, param_name)
            assert len(w.size()) == 2
            self.register_parameter(
                f'{lora_name}_lora_A',
                nn.Parameter(w.new_zeros((self.r, w.size(1))))
            )
            self.register_parameter(
                f'{lora_name}_lora_B',
                nn.Parameter(w.new_zeros((w.size(0), self.r)))
            )
            w.requires_grad = False

    def init_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            if hasattr(self, f'{lora_name}_lora_A'):
                nn.init.kaiming_uniform_(
                    getattr(self, f'{lora_name}_lora_A'), a=math.sqrt(5)
                )
                nn.init.zeros_(getattr(self, f'{lora_name}_lora_B'))

    def transpose(self, w: torch.Tensor):
        return w.transpose(0, 1) if self.fan_in_fan_out else w

    def merge_BA(self, param_name: str):
        lora_name = self.params_with_lora[param_name]
        A = getattr(self, f'{lora_name}_lora_A')
        B = getattr(self, f'{lora_name}_lora_B')
        return self.transpose((B @ A).view(getattr(self, param_name).shape))

    def merge_lora_param(self):
        for param_name, lora_name in self.params_with_lora.items():
            p = getattr(self, param_name)
            p_new = p.detach() + self.merge_BA(param_name) * self.scaling
            delattr(self, param_name)
            setattr(self, param_name, p_new)

    def add_lora_data(self):
        for param_name in self.params_with_lora:
            getattr(self, param_name).data += self.merge_BA(param_name) * self.scaling

    def sub_lora_data(self):
        for param_name in self.params_with_lora:
            getattr(self, param_name).data -= self.merge_BA(param_name) * self.scaling

    def lora_train(self, mode: bool = True):
        if mode:
            if self.merged and self.r > 0:
                self.sub_lora_data()
            self.merged = False
        else:
            if not self.merged and self.r > 0:
                self.add_lora_data()
            self.merged = True


# ---------------------------------------------------------------------------
# MultiheadAttention with LoRA applied to in-projection weights
# ---------------------------------------------------------------------------

class MultiheadAttentionLoRA(nn.MultiheadAttention, LoRALayer):
    """
    Drop-in replacement for nn.MultiheadAttention with LoRA adapters applied
    to the in-projection weight matrix (q, k, v together in in_proj_weight).
    This matches the structure CLIP uses for its attention blocks.
    """

    def __init__(self, original_attn: nn.MultiheadAttention, r: int = 4,
                 lora_alpha: int = 1, dropout_rate: float = 0.0,
                 dtype=None):
        embed_dim = original_attn.embed_dim
        num_heads = original_attn.num_heads
        dropout = original_attn.dropout
        bias = original_attn.in_proj_bias is not None
        kdim = original_attn.kdim
        vdim = original_attn.vdim

        nn.MultiheadAttention.__init__(
            self, embed_dim, num_heads, dropout=dropout, bias=bias,
            kdim=kdim, vdim=vdim
        )
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           dropout_rate=dropout_rate)

        # Copy weights from original attention
        self.load_state_dict(original_attn.state_dict(), strict=False)

        source_weight = original_attn.in_proj_weight
        if source_weight is None:
            raise ValueError("LoRA attention currently requires a combined in_proj_weight")
        if dtype is not None and dtype != source_weight.dtype:
            raise ValueError(
                f"Requested LoRA dtype {dtype} does not match source attention dtype "
                f"{source_weight.dtype}"
            )
        self.to(device=source_weight.device, dtype=source_weight.dtype)

        # Freeze all parameters initially
        for p in self.parameters():
            p.requires_grad = False

        # Apply LoRA to q, k, v projection separately
        if r > 0:
            dim = embed_dim
            # We add separate A/B matrices for q, k, v
            self.lora_q_A = nn.Parameter(self.in_proj_weight.new_zeros((r, dim)))
            self.lora_q_B = nn.Parameter(self.in_proj_weight.new_zeros((dim, r)))
            self.lora_v_A = nn.Parameter(self.in_proj_weight.new_zeros((r, dim)))
            self.lora_v_B = nn.Parameter(self.in_proj_weight.new_zeros((dim, r)))

            nn.init.kaiming_uniform_(self.lora_q_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_q_B)
            nn.init.kaiming_uniform_(self.lora_v_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_v_B)

    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # If LoRA is active, add delta to in_proj_weight for q and v slices
        if self.r > 0 and not self.merged:
            dim = self.embed_dim
            delta_q = (self.lora_q_B @ self.lora_q_A) * self.scaling
            delta_v = (self.lora_v_B @ self.lora_v_A) * self.scaling

            # in_proj_weight layout: [W_q; W_k; W_v]  shape [3*dim, dim]
            weight = self.in_proj_weight.clone()
            weight[:dim] = weight[:dim] + delta_q
            weight[2 * dim:] = weight[2 * dim:] + delta_v
        else:
            weight = self.in_proj_weight

        return F.multi_head_attention_forward(
            query, key, value,
            self.embed_dim, self.num_heads,
            weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )


# ---------------------------------------------------------------------------
# PlainMultiheadAttentionLoRA — alias used in the original CLIP-LoRA codebase
# ---------------------------------------------------------------------------
PlainMultiheadAttentionLoRA = MultiheadAttentionLoRA
