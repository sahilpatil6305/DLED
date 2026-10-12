#!/usr/bin/env python3
"""Offline, single-batch DLED runtime smoke test (no optimizer step/training)."""

import hashlib
import os
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT))

# Import the same entrypoint/model/data packages used by the DLED runner.
import train as training_entrypoint  # noqa: F401,E402
import dataset as dataset_package  # noqa: F401,E402
import detectors as detector_package  # noqa: F401,E402
from clip import clip as clip_module  # noqa: E402
from detectors.dled import DLED_Detector  # noqa: E402
from loralib.layers import MultiheadAttentionLoRA  # noqa: E402
from loralib.utils import INDEX_POSITIONS_TEXT, INDEX_POSITIONS_VISION  # noqa: E402


def _load_config():
    detector_path = ROOT / "training" / "config" / "detector" / "dled.yaml"
    train_config_path = ROOT / "training" / "config" / "train_config.yaml"
    with detector_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with train_config_path.open(encoding="utf-8") as stream:
        train_config = yaml.safe_load(stream)
    if "label_dict" in config:
        train_config["label_dict"] = config["label_dict"]
    config.update(train_config)

    # Match the NOEFS arguments assembled by training/train.py and train.sh.
    config.update(
        {
            "train_dataset": ["DF40_FF_train_NOEFS"],
            "test_dataset": ["DF40_FF_test_NOFE_all"],
            "training_data_ratio": 1.0,
            "max_samples_per_class": 13504,
            "detect_novel": False,
            "num_classes": 3,
            "training_mode": "multi",
            "device_choice": "old",
            "avu_reg": True,
            "kl_reg": False,
        }
    )
    config.setdefault("backbone_config", {})["num_classes"] = 3
    return config


def _require_cached_clip(config):
    root = Path(
        os.environ.get(
            "DLED_PRETRAINED_ROOT",
            ROOT / "training" / "pretrained" / "clip",
        )
    ).expanduser().resolve()
    os.environ["DLED_PRETRAINED_ROOT"] = str(root)

    backbone = config["MODEL"]["BACKBONE"]["NAME"]
    if backbone not in clip_module._MODELS:
        raise RuntimeError(f"No bundled CLIP URL for configured backbone {backbone!r}")
    url = clip_module._MODELS[backbone]
    checkpoint = root / os.path.basename(url)
    expected_sha256 = url.rstrip("/").split("/")[-2]
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Cached CLIP checkpoint is required; refusing download: {checkpoint}"
        )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"Cached CLIP checksum mismatch for {checkpoint}: "
            f"expected {expected_sha256}, got {digest}"
        )
    return root, url, checkpoint


def _assert_no_download(root, url, checkpoint):
    """Replace CLIP's downloader during initialization with a cache-only guard."""
    expected_sha256 = url.rstrip("/").split("/")[-2]
    original_download = clip_module._download
    cache_root = root

    def cache_only_download(requested_url, root=None):
        if requested_url != url:
            raise RuntimeError(f"Unexpected CLIP download request: {requested_url}")
        candidate = Path(root or cache_root) / checkpoint.name
        if candidate.resolve() != checkpoint.resolve():
            raise RuntimeError(f"Unexpected CLIP cache path: {candidate}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError(f"Cached CLIP checkpoint changed during initialization: {candidate}")
        return str(candidate)

    clip_module._download = cache_only_download
    return original_download


def _blocks(encoder):
    transformer = getattr(encoder, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    if resblocks is None:
        raise AssertionError(f"No transformer.resblocks on {type(encoder).__name__}")
    return resblocks


def _verify_lora(detector, config, device):
    model = detector.model
    text_indices = set(INDEX_POSITIONS_TEXT[config["TRAINER"]["POSITION"]])
    visual_indices = set(
        INDEX_POSITIONS_VISION[config["MODEL"]["BACKBONE"]["NAME"]][
            config["TRAINER"]["POSITION"]
        ]
    )
    encoders = {
        "spatial_text": (model.spatial_text_encoder, text_indices),
        "spatial_vision": (model.spatial_image_encoder, visual_indices),
        "frequency_text": (model.frequency_text_encoder, text_indices),
        "frequency_vision": (model.frequency_image_encoder, visual_indices),
    }
    replacements = []
    for name, (encoder, expected_indices) in encoders.items():
        blocks = _blocks(encoder)
        actual_indices = {
            index
            for index, block in enumerate(blocks)
            if isinstance(getattr(block, "attn", None), MultiheadAttentionLoRA)
        }
        if actual_indices != expected_indices:
            raise AssertionError(
                f"{name}: expected LoRA blocks {sorted(expected_indices)}, "
                f"found {sorted(actual_indices)}"
            )
        replacements.extend(
            block.attn
            for block in blocks
            if isinstance(getattr(block, "attn", None), MultiheadAttentionLoRA)
        )

    if len(replacements) != 48:
        raise AssertionError(f"Expected 48 ViT-B/16 attention replacements, found {len(replacements)}")

    expected_dtype = model.dtype
    for index, attention in enumerate(replacements):
        if not hasattr(attention, "lora_q_A") or not hasattr(attention, "lora_v_A"):
            raise AssertionError(f"LoRA replacement {index} is missing Q/V adapters")
        for parameter_name, parameter in attention.named_parameters():
            if parameter.device != device:
                raise AssertionError(
                    f"LoRA replacement {index}.{parameter_name} is on {parameter.device}, "
                    f"expected {device}"
                )
            if parameter.dtype != expected_dtype:
                raise AssertionError(
                    f"LoRA replacement {index}.{parameter_name} has {parameter.dtype}, "
                    f"expected {expected_dtype}"
                )

    actual_ranks = sorted({attention.r for attention in replacements})
    actual_projections = sorted(
        projection
        for projection, attribute in (
            ("q", "lora_q_A"),
            ("k", "lora_k_A"),
            ("v", "lora_v_A"),
            ("o", "lora_o_A"),
        )
        if any(hasattr(attention, attribute) for attention in replacements)
    )
    configured_rank = config["TRAINER"].get(
        "LORA_R", config["TRAINER"].get("RANK")
    )
    configured_projections = sorted(config["TRAINER"].get("ENABLE_LORA", []))
    print(
        "LoRA settings: configured rank/projections="
        f"{configured_rank}/{configured_projections}; effective="
        f"{actual_ranks}/{actual_projections}"
    )
    if actual_ranks != [configured_rank] or actual_projections != configured_projections:
        print(
            "WARNING: local LoRA behavior differs from the upstream DLED YAML. "
            "Rank/projection behavior was not modified; verify the omitted upstream loralib."
        )

    trainable = {
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    unexpected_trainable = {
        name
        for name in trainable
        if "spatial_prompt_learner" not in name
        and "frequency_prompt_learner" not in name
        and "lora_" not in name
    }
    if unexpected_trainable:
        raise AssertionError(f"Unexpected trainable parameters: {sorted(unexpected_trainable)}")
    if not any("spatial_prompt_learner.ctx" in name for name in trainable):
        raise AssertionError("Spatial prompt context is not trainable")
    if not any("frequency_prompt_learner.ctx" in name for name in trainable):
        raise AssertionError("Frequency prompt context is not trainable")
    if not any("lora_" in name for name in trainable):
        raise AssertionError("No LoRA parameters are trainable")

    return trainable


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires a CUDA-enabled Kaggle runtime")
    torch.cuda.set_device(torch.cuda.current_device())
    device = torch.device("cuda", torch.cuda.current_device())

    config = _load_config()
    cache_root, clip_url, checkpoint = _require_cached_clip(config)
    print(f"Using checksum-verified cached CLIP checkpoint: {checkpoint}")

    original_download = _assert_no_download(cache_root, clip_url, checkpoint)
    try:
        detector = DLED_Detector(config)
    finally:
        clip_module._download = original_download

    detector.model.train()
    trainable = _verify_lora(detector, config, device)
    labels = torch.tensor([0, 1], dtype=torch.long, device=device)
    batch = {
        "image": torch.randn(2, 3, 224, 224, device=device),
        "freq": torch.randn(2, 3, 224, 224, device=device),
        "label": labels,
    }

    predictions = detector(batch, "fusion")
    for key in ("spatial_pred_alpha", "frequency_pred_alpha", "pred_alpha"):
        if predictions[key].shape != (2, 3):
            raise AssertionError(f"Unexpected {key} shape: {tuple(predictions[key].shape)}")
    for key, value in predictions.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            if not torch.isfinite(value).all():
                raise AssertionError(f"Non-finite forward output: {key}")

    predictions["epoch"] = 0
    predictions["total_epoch"] = 1
    losses = detector.get_losses(batch, predictions, mode="train")
    loss = losses["overall"]
    if not torch.isfinite(loss):
        raise AssertionError(f"Non-finite smoke-test loss: {loss}")
    loss.backward()

    missing_gradients = [
        name
        for name, parameter in detector.model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_gradients:
        raise AssertionError(f"Trainable parameters without gradients: {missing_gradients[:20]}")
    nonfinite_gradients = [
        name
        for name, parameter in detector.model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if nonfinite_gradients:
        raise AssertionError(f"Non-finite gradients: {nonfinite_gradients[:20]}")
    frozen_with_gradients = [
        name
        for name, parameter in detector.model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if frozen_with_gradients:
        raise AssertionError(f"Frozen parameters received gradients: {frozen_with_gradients[:20]}")

    print(
        "RUNTIME SMOKE PASSED: imports, cached CLIP, DLED init, 48 attention "
        f"replacements, forward, backward; {len(trainable)} trainable parameters "
        "received finite gradients. No optimizer step or training epoch was run."
    )
    print("Note: this runtime pass does not resolve the declared/effective LoRA config mismatch.")


if __name__ == "__main__":
    main()
