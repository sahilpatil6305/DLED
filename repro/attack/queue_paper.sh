#!/usr/bin/env bash
# Paper-grade attack queue for DLED (NOEFS). Each job is its own timestamped run
# (repro/run.sh), so a failure in one does not stop the others. Run it detached:
#
#   nohup repro/attack/queue_paper.sh ~/DLED/runs/20261024-152447_baseline_noefs > ~/DLED/runs/queue_paper.log 2>&1 &
#
# Progress: tail ~/DLED/runs/queue_paper.log ; results: ~/DLED/runs/RUNS_INDEX.txt
#
#   1. paper_margin  rejection-aware margin loss, 300 images/label, eps 1 2 4 8 16, 50 steps   (~13-16 h)
#   2. paper_ce      standard CE-PGD, same images and settings (fair comparison)              (~13-16 h)
#   3. paper_strong  stronger attack on a subset: 200 steps x 3 restarts, eps 1 2 4 8 32 255
#                    (masking check: success -> ~100% at large eps, conclusions unchanged)   (~3-5 h)
# Same --seed => same images in 1 and 2. GPU only; one job at a time.
set -u
BASE="${1:?usage: queue_paper.sh <baseline run folder>}"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source repro/env.sh > /dev/null
COMMON=(--run "$BASE" --seed 0)

echo "[$(date -Is)] queue start (baseline $BASE)"
repro/run.sh paper_margin python repro/attack/run_attack.py "${COMMON[@]}" \
    --n_per_class 300 --eps 1 2 4 8 16 --steps 50 --loss margin
echo "[$(date -Is)] paper_margin finished (exit $?)"
repro/run.sh paper_ce python repro/attack/run_attack.py "${COMMON[@]}" \
    --n_per_class 300 --eps 1 2 4 8 16 --steps 50 --loss ce
echo "[$(date -Is)] paper_ce finished (exit $?)"
repro/run.sh paper_strong python repro/attack/run_attack.py "${COMMON[@]}" \
    --n_per_class 50 --eps 1 2 4 8 32 255 --steps 200 --restarts 3 --loss margin
echo "[$(date -Is)] paper_strong finished (exit $?)"
echo "[$(date -Is)] queue done"
