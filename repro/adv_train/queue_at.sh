#!/usr/bin/env bash
# Adversarial-training queue for DLED (NOEFS): train each method, then evaluate it with
# the same clean test and attack harness as the baseline. Each step is its own logged run.
# Start only when the GPU is free (after queue_paper.sh), detached:
#
#   nohup repro/adv_train/queue_at.sh ~/DLED/runs/20261024-152447_baseline_noefs > ~/DLED/runs/queue_at.log 2>&1 &
#
# Methods (5 fine-tuning epochs from the baseline checkpoint, eps_train 4/255, PGD-3):
#   ft_clean  control: same fine-tuning without attacks        (~1.5 h)
#   at_pgd    AT-1 Madry PGD-AT                                  (~6-7 h)
#   at_evat   AT-2 EV-AT-style                                   (~7 h)
#   at_reject AT-3 rejection-aware (ours)                        (~6-7 h)
# Each is followed by: clean full-test evaluation (~15 min) and the attack evaluation on
# the same 100 images/label as attack_main, eps 1 2 4 8, margin loss (~3.5 h).
set -u
BASE="${1:?usage: queue_at.sh <baseline run folder>}"
CKPT="$BASE/dled_logs/test/DF40_FF_test_NOFE_all/ckpt_best.pth"
LOG="$BASE/DLED_NOEFS.log"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source repro/env.sh > /dev/null
RUNS="${DLED_RUNS:-$HOME/DLED/runs}"

echo "[$(date -Is)] AT queue start"
# pre-flight: 20 real GPU steps of the most memory-hungry mode at full batch size; stop the
# whole queue if it fails (e.g. out of memory) instead of failing four long runs in a row
repro/run.sh at_preflight python repro/adv_train/at_train.py --mode evat --ckpt "$CKPT" --log "$LOG" \
    --out {RUN} --epochs 1 --steps_per_epoch 20 --thr_batches 2 --log_every 5 --eps 4 --attack_steps 3 \
    || { echo "[$(date -Is)] PRE-FLIGHT FAILED, queue stopped"; exit 1; }
PF="$(ls -td "$RUNS"/*_at_preflight | head -1)"
grep -E "s/step|precision|peak GPU" "$PF/console.log" | tail -3
for m in clean pgd evat reject; do
  name="at_$m"; [ "$m" = clean ] && name="ft_clean"
  repro/run.sh "$name" python repro/adv_train/at_train.py --mode "$m" --ckpt "$CKPT" --log "$LOG" \
      --out {RUN} --epochs 5 --eps 4 --attack_steps 3
  RD="$(ls -td "$RUNS"/*_"$name" | head -1)"
  echo "[$(date -Is)] $name trained (run $RD)"
  [ -f "$RD/at_last.pth" ] || { echo "no checkpoint for $name, skipping its evaluation"; continue; }
  repro/run.sh "${name}_clean_eval" python repro/attack/eval_reload.py --run "$RD" \
      --ckpt "$RD/at_last.pth" --log "$RD/training.log"
  repro/run.sh "${name}_attack" python repro/attack/run_attack.py --ckpt "$RD/at_last.pth" \
      --log "$RD/training.log" --n_per_class 100 --eps 1 2 4 8 --steps 50 --seed 0
  echo "[$(date -Is)] $name evaluated"
done
echo "[$(date -Is)] AT queue done"
