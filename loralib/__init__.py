"""loralib — custom LoRA implementation for DLED."""
from .utils import (
    apply_lora,
    apply_double_lora,
    INDEX_POSITIONS_TEXT,
    INDEX_POSITIONS_VISION,
    mark_only_lora_as_trainable,
    lora_state_dict,
    get_lora_parameters,
)
from .layers import MultiheadAttentionLoRA, PlainMultiheadAttentionLoRA, LoRALayer

__all__ = [
    'apply_lora', 'apply_double_lora',
    'INDEX_POSITIONS_TEXT', 'INDEX_POSITIONS_VISION',
    'mark_only_lora_as_trainable', 'lora_state_dict', 'get_lora_parameters',
    'MultiheadAttentionLoRA', 'PlainMultiheadAttentionLoRA', 'LoRALayer',
]
