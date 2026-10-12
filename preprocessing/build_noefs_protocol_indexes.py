#!/usr/bin/env python3
"""
build_noefs_protocol_indexes.py
================================
Generates the two official DLED NOEFS protocol index files from the raw
per-method JSON files distributed with DF40:

  - DF40_FF_train_NOEFS.json   — training set (10 methods, seen only)
  - DF40_FF_test_NOFE_all.json — test set (35 methods, seen + unseen)

Design decisions that reproduce the official merge_seen_unseen.py semantics:
  * Real samples are kept PER METHOD (e.g. fsgan_Real, fomm_Real ...).
    All 10 training methods share the same 719 FF++ real videos, but the
    original script does NOT deduplicate them; each method contributes its
    own {method}_Real key.  That behaviour is preserved here.
  * Only the 'train' split of each source file is written into the training
    index; only the 'test' split is written into the test index.
  * Face-Editing methods and Celeb-DF-based indexes are excluded from both outputs.
  * EFS methods are excluded from the NOEFS training set (they appear only
    in the test set).

Usage
-----
  python preprocessing/build_noefs_protocol_indexes.py \\
      --input_dir  dataset_json \\
      --output_dir datasets/DF40/json_file

Run from the repository root. Image files are not required for this step.
"""

import os
import sys
import json
import argparse


# ---------------------------------------------------------------------------
# Protocol definitions  (derived from merge_seen_unseen.py)
# ---------------------------------------------------------------------------

# Ten seen methods used for NOEFS training: 5 FS + 5 FR.
TRAIN_FILES = [
    # Face-Swapping (seen)
    'fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff',
    # Face-Reenactment (seen)
    'fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff',
]

# Thirty-five methods used for NOFE testing
TEST_FILES = [
    # Face-Swapping -- seen (5)
    'fsgan_ff', 'faceswap_ff', 'simswap_ff', 'inswap_ff', 'blendface_ff',
    # Face-Swapping -- unseen (5)
    'uniface_ff', 'mobileswap_ff', 'e4s_ff', 'facedancer_ff', 'deepfacelab',
    # Face-Reenactment -- seen (5)
    'fomm_ff', 'facevid2vid_ff', 'wav2lip_ff', 'MRAA_ff', 'one_shot_free_ff',
    # Face-Reenactment -- unseen (8)
    'pirender_ff', 'tpsm_ff', 'lia_ff', 'danet_ff',
    'sadtalker_ff', 'mcnet_ff', 'hyperreenact_ff', 'heygen',
    # Entire-Face-Synthesis -- seen (5)
    'VQGAN_ff', 'StyleGAN2_ff', 'StyleGAN3_ff', 'StyleGANXL_ff', 'sd2.1_ff',
    # Entire-Face-Synthesis -- unseen (7)
    'ddim_ff', 'rddm_ff', 'pixart_ff', 'DiT_ff', 'SiT_ff',
    'MidJourney', 'whichisreal',
]

# Summary-level and alias files to skip unconditionally.
SKIP_ALWAYS = {
    'EFSAll_cdf', 'EFSAll_ff', 'FRAll_cdf', 'FRAll_ff',
    'FSAll_cdf', 'FSAll_ff', 'uniface', 'uniface_ori',
    'simswap', 'e4e', 'DF40_all',
}

# Category for logging
METHOD_CATEGORY = {
    'fsgan_ff': 'FS-seen', 'faceswap_ff': 'FS-seen', 'simswap_ff': 'FS-seen',
    'inswap_ff': 'FS-seen', 'blendface_ff': 'FS-seen',
    'uniface_ff': 'FS-unseen', 'mobileswap_ff': 'FS-unseen',
    'e4s_ff': 'FS-unseen', 'facedancer_ff': 'FS-unseen', 'deepfacelab': 'FS-unseen',
    'fomm_ff': 'FR-seen', 'facevid2vid_ff': 'FR-seen', 'wav2lip_ff': 'FR-seen',
    'MRAA_ff': 'FR-seen', 'one_shot_free_ff': 'FR-seen',
    'pirender_ff': 'FR-unseen', 'tpsm_ff': 'FR-unseen', 'lia_ff': 'FR-unseen',
    'danet_ff': 'FR-unseen', 'sadtalker_ff': 'FR-unseen', 'mcnet_ff': 'FR-unseen',
    'hyperreenact_ff': 'FR-unseen', 'heygen': 'FR-unseen',
    'VQGAN_ff': 'EFS-seen', 'StyleGAN2_ff': 'EFS-seen', 'StyleGAN3_ff': 'EFS-seen',
    'StyleGANXL_ff': 'EFS-seen', 'sd2.1_ff': 'EFS-seen',
    'ddim_ff': 'EFS-unseen', 'rddm_ff': 'EFS-unseen', 'pixart_ff': 'EFS-unseen',
    'DiT_ff': 'EFS-unseen', 'SiT_ff': 'EFS-unseen',
    'MidJourney': 'EFS-unseen', 'whichisreal': 'EFS-unseen',
}

FACE_EDITING_KEYS = {
    'CollabDiff_Fake', 'e4e_Fake', 'stargan_Fake', 'starganv2_Fake', 'styleclip_Fake',
    'e4e_Real', 'CollabDiff_Real', 'stargan_Real', 'starganv2_Real', 'styleclip_Real',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_source_json(input_dir, file_name):
    path = os.path.join(input_dir, file_name + '.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Source file not found: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if file_name not in data:
        raise ValueError(
            f"Expected top-level key '{file_name}' in {path}, got: {list(data.keys())}"
        )
    return data[file_name]


def validate_video_entry(video_id, entry, context):
    errors = []
    warnings = []
    if not isinstance(entry, dict):
        errors.append(f"{context}/{video_id}: entry is not a dict")
        return errors, warnings
    if 'label' not in entry:
        errors.append(f"{context}/{video_id}: missing 'label' field")
    elif not isinstance(entry['label'], str) or not entry['label']:
        errors.append(f"{context}/{video_id}: 'label' must be a non-empty string")
    if 'frames' not in entry:
        errors.append(f"{context}/{video_id}: missing 'frames' field")
    elif not isinstance(entry['frames'], list):
        errors.append(f"{context}/{video_id}: 'frames' is not a list")
    elif len(entry['frames']) == 0:
        warnings.append(f"{context}/{video_id}: empty 'frames' list (retained for source fidelity)")
    elif any(not isinstance(frame, str) or not frame for frame in entry['frames']):
        errors.append(f"{context}/{video_id}: 'frames' must contain non-empty strings")
    return errors, warnings


# ---------------------------------------------------------------------------
# Index builders
# ---------------------------------------------------------------------------

def build_train_index(input_dir, output_name):
    """Build DF40_FF_train_NOEFS.json — train split only, 10 seen methods."""
    print(f"\n=== Building training index: {output_name} ===")
    result = {output_name: {}}

    for file_name in TRAIN_FILES:
        print(f"  Loading {file_name}.json ...", end=' ')
        method_data = load_source_json(input_dir, file_name)
        sub_keys = list(method_data.keys())
        print(f"sub_keys={sub_keys}")

        for set_key in sub_keys:
            split_dict = method_data[set_key]
            if 'train' not in split_dict:
                print(f"    WARNING: '{set_key}' has no 'train' split — available: {list(split_dict.keys())}")
                continue
            train_data = split_dict['train']
            if set_key in result[output_name]:
                print(f"    WARNING: duplicate set_key '{set_key}' — skipping duplicate")
                continue
            result[output_name][set_key] = {'train': train_data}
            n_videos = len(train_data)
            n_frames = sum(
                len(v['frames']) for v in train_data.values()
                if isinstance(v, dict) and 'frames' in v
            )
            print(f"    {set_key}: {n_videos} videos, {n_frames} frames")

    return result


def build_test_index(input_dir, output_name):
    """Build DF40_FF_test_NOFE_all.json — test split only, 35 methods."""
    print(f"\n=== Building test index: {output_name} ===")
    result = {output_name: {}}

    for file_name in TEST_FILES:
        cat = METHOD_CATEGORY.get(file_name, 'UNKNOWN')
        print(f"  Loading {file_name}.json [{cat}] ...", end=' ')
        method_data = load_source_json(input_dir, file_name)
        sub_keys = list(method_data.keys())
        print(f"sub_keys={sub_keys}")

        for set_key in sub_keys:
            if set_key in FACE_EDITING_KEYS:
                print(f"    [SKIP FE] {set_key}")
                continue

            split_dict = method_data[set_key]
            if 'test' not in split_dict:
                print(f"    WARNING: '{set_key}' has no 'test' split — available: {list(split_dict.keys())}")
                continue
            test_data = split_dict['test']
            if set_key in result[output_name]:
                print(f"    WARNING: duplicate set_key '{set_key}' — skipping duplicate")
                continue
            result[output_name][set_key] = {'test': test_data}
            n_videos = len(test_data)
            n_frames = sum(
                len(v['frames']) for v in test_data.values()
                if isinstance(v, dict) and 'frames' in v
            )
            print(f"    {set_key}: {n_videos} videos, {n_frames} frames")

    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_index(index, expected_split, output_name, label_dict):
    """Validate structure and label coverage. Returns summary dict."""
    print(f"\n=== Validating {output_name} ===")
    errors = []
    warnings = []

    assert len(index) == 1 and output_name in index
    body = index[output_name]

    found_real_keys = set()
    found_fake_keys = set()
    label_errors = []
    fe_present = []
    per_key_stats = {}
    video_schema_errors = []
    empty_frame_warnings = []

    for set_key, split_data in body.items():
        if set_key in FACE_EDITING_KEYS:
            fe_present.append(set_key)

        if 'Real' in set_key:
            found_real_keys.add(set_key)
        elif 'Fake' in set_key:
            found_fake_keys.add(set_key)

        if not isinstance(split_data, dict):
            errors.append(f"  set_key='{set_key}' split data is not a dict")
            continue
        if expected_split not in split_data:
            errors.append(f"  set_key='{set_key}' missing split '{expected_split}'; has {list(split_data.keys())}")
            continue

        split_videos = split_data[expected_split]
        if not isinstance(split_videos, dict):
            errors.append(f"  set_key='{set_key}' split '{expected_split}' is not a video dict")
            continue
        n_videos = len(split_videos)
        n_frames = 0
        for vid_id, vid_entry in split_videos.items():
            errs, entry_warnings = validate_video_entry(
                vid_id, vid_entry, f"{set_key}/{expected_split}"
            )
            video_schema_errors.extend(errs)
            empty_frame_warnings.extend(entry_warnings)
            if isinstance(vid_entry, dict) and isinstance(vid_entry.get('frames'), list):
                n_frames += len(vid_entry['frames'])
        per_key_stats[set_key] = (n_videos, n_frames)

        for vid_id, vid_entry in split_videos.items():
            if (
                isinstance(vid_entry, dict)
                and 'label' in vid_entry
                and vid_entry['label'] != set_key
            ):
                label_errors.append(
                    f"  set_key='{set_key}' video='{vid_id}' "
                    f"label='{vid_entry['label']}' (mismatch)"
                )

    errors.extend(f"Invalid video schema: {e}" for e in video_schema_errors)
    errors.extend(f"Invalid video label: {e}" for e in label_errors)

    if fe_present:
        errors.append(f"Face-Editing keys found in index: {fe_present}")

    uncovered = [k for k in body.keys() if label_dict and k not in label_dict]
    if uncovered:
        warnings.append(f"Keys not in train_config label_dict: {uncovered}")

    total_videos = sum(n for n, _ in per_key_stats.values())
    total_frames = sum(f for _, f in per_key_stats.values())
    real_videos  = sum(n for k, (n, _) in per_key_stats.items() if 'Real' in k)
    fake_videos  = sum(n for k, (n, _) in per_key_stats.items() if 'Fake' in k)
    real_frames  = sum(f for k, (_, f) in per_key_stats.items() if 'Real' in k)
    fake_frames  = sum(f for k, (_, f) in per_key_stats.items() if 'Fake' in k)

    print(f"  Set keys: {len(body)}")
    print(f"  Total videos: real={real_videos}, fake={fake_videos}, total={total_videos}")
    print(f"  Total frames: real={real_frames}, fake={fake_frames}, total={total_frames}")
    print(f"\n  Per-key breakdown:")
    for k in sorted(per_key_stats.keys()):
        nv, nf = per_key_stats[k]
        print(f"    {k:<32s}: {nv:>4} videos, {nf:>8} frames")

    if video_schema_errors:
        print(f"\n  Schema errors ({len(video_schema_errors)}):")
        for e in video_schema_errors[:20]:
            print(f"    {e}")
    else:
        print("\n  All video entries have valid schema fields.")

    if empty_frame_warnings:
        print(
            f"\n  Empty-frame warnings ({len(empty_frame_warnings)}; retained "
            "for source-index fidelity and skipped by the dataset loader):"
        )
        for warning in empty_frame_warnings[:20]:
            print(f"    {warning}")

    if label_errors:
        print("  Label mismatches:")
        for e in label_errors:
            print(e)
    else:
        print("  All sample label fields match their set_key.")

    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print("  No errors.")

    return {
        'n_set_keys': len(body),
        'n_videos': total_videos,
        'n_frames': total_frames,
        'n_real_videos': real_videos,
        'n_fake_videos': fake_videos,
        'n_real_frames': real_frames,
        'n_fake_frames': fake_frames,
        'schema_errors': len(video_schema_errors),
        'empty_frame_entries': len(empty_frame_warnings),
        'label_errors': len(label_errors),
        'fe_contamination': len(fe_present),
        'errors': errors,
        'warnings': warnings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate DLED NOEFS protocol indexes from raw DF40 JSON files."
    )
    p.add_argument('--input_dir', default='dataset_json',
                   help='Directory containing raw per-method JSON files (default: dataset_json)')
    p.add_argument('--output_dir', default='datasets/DF40/json_file',
                   help='Output directory for merged indexes (default: datasets/DF40/json_file)')
    p.add_argument('--validate_only', action='store_true',
                   help='Only validate existing output files; do not regenerate them')
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_dir  = os.path.join(repo_root, args.input_dir)
    output_dir = os.path.join(repo_root, args.output_dir)

    print(f"Repo root  : {repo_root}")
    print(f"Input dir  : {input_dir}")
    print(f"Output dir : {output_dir}")

    if not os.path.isdir(input_dir):
        sys.exit(f"ERROR: input_dir does not exist: {input_dir}")

    # Source file availability
    print("\n--- Source file availability check ---")
    all_required = sorted(set(TRAIN_FILES) | set(TEST_FILES))
    missing = []
    for f in all_required:
        p = os.path.join(input_dir, f + '.json')
        ok = os.path.exists(p)
        print(f"  [{'OK    ' if ok else 'MISSING'}] {f}.json")
        if not ok:
            missing.append(f)
    if missing:
        sys.exit(f"\nERROR: {len(missing)} required source file(s) missing: {missing}")
    print(f"\nAll {len(all_required)} required source files present.")

    # Load label_dict from train_config.yaml for validation
    label_dict = {}
    train_cfg_path = os.path.join(repo_root, 'training', 'config', 'train_config.yaml')
    if os.path.exists(train_cfg_path):
        try:
            import yaml
            with open(train_cfg_path) as f:
                cfg = yaml.safe_load(f)
            label_dict = cfg.get('label_dict', {})
            print(f"\nLoaded label_dict from train_config.yaml ({len(label_dict)} entries).")
        except Exception as e:
            print(f"\nWARNING: Could not load label_dict from train_config.yaml: {e}")

    TRAIN_NAME = 'DF40_FF_train_NOEFS'
    TEST_NAME  = 'DF40_FF_test_NOFE_all'
    train_path = os.path.join(output_dir, TRAIN_NAME + '.json')
    test_path  = os.path.join(output_dir, TEST_NAME + '.json')

    if not args.validate_only:
        os.makedirs(output_dir, exist_ok=True)

        train_index = build_train_index(input_dir, TRAIN_NAME)
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_index, f, indent=4)
        print(f"\nWritten: {train_path}")

        test_index = build_test_index(input_dir, TEST_NAME)
        with open(test_path, 'w', encoding='utf-8') as f:
            json.dump(test_index, f, indent=4)
        print(f"Written: {test_path}")
    else:
        print("\n--validate_only mode: loading existing files.")
        with open(train_path) as f:
            train_index = json.load(f)
        with open(test_path) as f:
            test_index = json.load(f)

    train_stats = validate_index(train_index, 'train', TRAIN_NAME, label_dict)
    test_stats  = validate_index(test_index,  'test',  TEST_NAME,  label_dict)

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\nTraining index  [{TRAIN_NAME}]")
    print(f"  Set keys:   {train_stats['n_set_keys']}  "
          f"(expect: {2 * len(TRAIN_FILES)} — 10 Real + 10 Fake)")
    print(f"  Videos:     real={train_stats['n_real_videos']},  "
          f"fake={train_stats['n_fake_videos']},  total={train_stats['n_videos']}")
    print(f"  Frames:     real={train_stats['n_real_frames']},  "
          f"fake={train_stats['n_fake_frames']},  total={train_stats['n_frames']}")
    print(f"  FE contamination: {train_stats['fe_contamination']}")
    print(f"  Errors: schema={train_stats['schema_errors']}, "
            f"empty_frames={train_stats['empty_frame_entries']}, "
            f"label={train_stats['label_errors']}, fatal={len(train_stats['errors'])}")

    print(f"\nTest index  [{TEST_NAME}]")
    print(f"  Set keys:   {test_stats['n_set_keys']}  "
          f"(expect: {2 * len(TEST_FILES)} — 35 Real + 35 Fake)")
    print(f"  Videos:     real={test_stats['n_real_videos']},  "
          f"fake={test_stats['n_fake_videos']},  total={test_stats['n_videos']}")
    print(f"  Frames:     real={test_stats['n_real_frames']},  "
          f"fake={test_stats['n_fake_frames']},  total={test_stats['n_frames']}")
    print(f"  FE contamination: {test_stats['fe_contamination']}")
    print(f"  Errors: schema={test_stats['schema_errors']}, "
            f"empty_frames={test_stats['empty_frame_entries']}, "
            f"label={test_stats['label_errors']}, fatal={len(test_stats['errors'])}")

    all_errors = train_stats['errors'] + test_stats['errors']
    if all_errors:
        print(f"\nFAILED — {len(all_errors)} error(s) require attention before training.")
        sys.exit(1)
    else:
        print("\nPASSED — Both indexes are structurally valid.")
        print(f"\nGenerated files:")
        print(f"  {train_path}")
        print(f"  {test_path}")


if __name__ == '__main__':
    main()
