#!/usr/bin/env bash
# CPU-only smoke test of the adversarial-training pipeline (safe next to a GPU job).
# For each mode: 2 training steps (batch 2, 1 attack step), save the full checkpoint,
# recompute thresholds on 2 batches, then load that checkpoint in the attack harness
# and run a 1-step attack on 1 image per class. Takes ~15-30 min on CPU in total.
#
#   repro/run.sh at_smoke repro/adv_train/smoke_at.sh ~/DLED/runs/20261024-152447_baseline_noefs
set -u
BASE="${1:?usage: smoke_at.sh <baseline run folder>}"
CKPT="$BASE/dled_logs/test/DF40_FF_test_NOFE_all/ckpt_best.pth"
LOG="$BASE/DLED_NOEFS.log"
OUT="${DLED_LOG_DIR:-/tmp}/smoke"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
fail=0
for mode in clean pgd evat reject; do
  echo "=== $mode: training"
  python repro/adv_train/at_train.py --mode "$mode" --ckpt "$CKPT" --log "$LOG" --out "$OUT/$mode" \
      --device cpu --batch 2 --workers 0 --steps_per_epoch 2 --epochs 1 --attack_steps 1 \
      --thr_batches 2 --log_every 1 || { echo "!!! $mode training FAILED"; fail=1; continue; }
  echo "=== $mode: evaluating the saved checkpoint with the attack harness"
  python repro/attack/run_attack.py --ckpt "$OUT/$mode/at_last.pth" --log "$OUT/$mode/training.log" \
      --out "$OUT/$mode/eval" --device cpu --n_per_class 1 --steps 1 --eps 4 --batch 2 \
      --attacks E2 E4 || { echo "!!! $mode evaluation FAILED"; fail=1; }
done
[ $fail -eq 0 ] && echo "AT SMOKE TEST PASSED" || echo "AT SMOKE TEST FAILED"
exit $fail
