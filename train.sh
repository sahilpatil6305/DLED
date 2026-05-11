#!/usr/bin/env bash
# Unified training launcher for DLED (Dual-Level Evidence Detector).
#
# This single script replaces the original train_protocol{1..8}.sh files.
# Only the protocols that train the DLED method itself (dled detector +
# Trainer class) are exposed here; baseline scripts have been removed.
#
# Usage:
#   ./train.sh <protocol> [extra_train.py_args...]
#
# Protocols:
#   NOEFS      — 3-class same-domain training (Real / Swapping / Reenactment),
#                Face-Editing + EFS held out. (Original train_protocol1.sh)
#   NOFS       — 3-class same-domain training (Real / EFS / Reenactment),
#                Face-Swapping held out.
#   NOFR       — 3-class same-domain training (Real / Swapping / EFS),
#                Face-Reenactment held out.
#   open       — 4-class open-test training (Real / Swap / Reenact / EFS),
#                Face-Editing held out. (Original train_protocol3.sh)
#
# Environment variables (override the defaults to match your machine):
#   DLED_LOG_DIR              — where to write the .log file
#                               (default: ./logs/$PROTOCOL.log)
#   DLED_LOCAL_RANK           — CUDA device index (default: 0)
#   DLED_MAX_SAMPLES_PER_CLS  — cap samples per training class (default: 13504)
#   DLED_NUM_EPOCHS           — number of epochs (default: 50)
#   DLED_TEST_FRE             — test frequency in epochs (default: 50)
#   DLED_PRETRAINED_ROOT      — directory used to cache CLIP weights
#   DF40_DATA_ROOT / DF40_OLD_ROOT — see README for path-rewriting setup

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <NOEFS|NOFS|NOFR|open> [extra_train.py_args...]" >&2
    exit 1
fi

PROTOCOL="$1"
shift

LOCAL_RANK="${DLED_LOCAL_RANK:-0}"
MAX_SAMPLES_PER_CLS="${DLED_MAX_SAMPLES_PER_CLS:-13504}"
NUM_EPOCHS="${DLED_NUM_EPOCHS:-50}"
TEST_FRE="${DLED_TEST_FRE:-50}"
LOG_DIR_DEFAULT="./logs"
LOG_DIR="${DLED_LOG_DIR:-$LOG_DIR_DEFAULT}"
mkdir -p "$LOG_DIR"

case "$PROTOCOL" in
    NOEFS)
        TRAIN_DATASET="DF40_FF_train_NOEFS"
        TEST_DATASET="DF40_FF_test_NOFE_all"
        NUM_CLASSES=3
        ;;
    NOFS)
        TRAIN_DATASET="DF40_FF_train_NOFS"
        TEST_DATASET="DF40_FF_test_NOFE_all"
        NUM_CLASSES=3
        ;;
    NOFR)
        TRAIN_DATASET="DF40_FF_train_NOFR"
        TEST_DATASET="DF40_FF_test_NOFE_all"
        NUM_CLASSES=3
        ;;
    open)
        TRAIN_DATASET="DF40_FF_train"
        TEST_DATASET="DF40_open_test"
        NUM_CLASSES=4
        ;;
    *)
        echo "Unknown protocol: $PROTOCOL" >&2
        echo "Valid protocols: NOEFS, NOFS, NOFR, open" >&2
        exit 1
        ;;
esac

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"

LOG_FILE="$LOG_DIR/DLED_${PROTOCOL}.log"
echo "Logging to $LOG_FILE"

python3 training/train.py \
    --detector_path ./training/config/detector/dled.yaml \
    --train_dataset "$TRAIN_DATASET" \
    --test_dataset  "$TEST_DATASET" \
    --local_rank "$LOCAL_RANK" \
    --num_classes "$NUM_CLASSES" \
    --test_fre "$TEST_FRE" \
    --num_epochs "$NUM_EPOCHS" \
    --avu_reg \
    --max_samples_per_class "$MAX_SAMPLES_PER_CLS" \
    "$@" \
    > "$LOG_FILE" 2>&1
