#!/usr/bin/env bash
# Wait until the attack queue (queue_paper.sh, given by its PID) has finished, then start
# the adversarial-training queue. Run detached:
#   nohup repro/adv_train/after_paper_queue.sh <queue_paper PID> ~/DLED/runs/20261024-152447_baseline_noefs > ~/DLED/runs/queue_at.log 2>&1 &
set -u
PID="${1:?usage: after_paper_queue.sh <PID of queue_paper.sh> <baseline run folder>}"
BASE="${2:?usage: after_paper_queue.sh <PID> <baseline run folder>}"
echo "[$(date -Is)] waiting for PID $PID (queue_paper.sh) to finish"
while kill -0 "$PID" 2>/dev/null; do sleep 300; done
echo "[$(date -Is)] attack queue finished; starting the AT queue"
exec "$(dirname "${BASH_SOURCE[0]}")/queue_at.sh" "$BASE"
