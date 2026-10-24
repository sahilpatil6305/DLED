#!/usr/bin/env bash
# Run any command with a complete, timestamped record.
#
#   repro/run.sh <run-name> <command...>
#
# Examples (from the repo root, after `source repro/env.sh`):
#   repro/run.sh smoke  python repro/smoke_model.py
#   repro/run.sh audit  python repro/noefs_data.py audit --json $DLED_DATA/DF40/json_file --out $DLED_DATA/audit
#   repro/run.sh pilot  env DLED_NUM_EPOCHS=2 DLED_TEST_FRE=1 DLED_MAX_SAMPLES_PER_CLS=512 ./train.sh NOEFS
#   nohup repro/run.sh baseline ./train.sh NOEFS --save_model --save_path {RUN}/checkpoints > /dev/null 2>&1 &
#
# The literal text {RUN} in any argument is replaced by this run's folder, so
# checkpoints and other outputs land inside it.
#
# Creates $DLED_RUNS/<YYYYmmdd-HHMMSS>_<name>/ (default $HOME/DLED/runs) with:
#   meta.txt (command, times, duration, exit code, git commit), console.log,
#   env.txt, versions.txt, pip_freeze.txt, nvidia_smi_start.txt, gpu_usage.csv,
#   configs/ (YAML copies), json_sha256.txt, git_status.txt, git_diff.patch,
#   dled_logs/ (the logger folder train.py creates under ./logs: training.log,
#   TensorBoard events, ckpt_best.pth, metric pickles -- linked while running,
#   moved here at the end, with a link left behind in ./logs), and results.txt
#   (final accuracy, per-class / unknown-detection rates, thresholds, checkpoints).
# train.sh's own log (DLED_<protocol>.log) is written straight into the run folder.
set -uo pipefail

NAME="${1:?usage: repro/run.sh <run-name> <command...>}"; shift
[ $# -ge 1 ] || { echo "usage: repro/run.sh <run-name> <command...>" >&2; exit 2; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS="${DLED_RUNS:-$HOME/DLED/runs}"
ID="$(date +%Y%m%d-%H%M%S)_${NAME}"
RD="$RUNS/$ID"
mkdir -p "$RD/configs"
export DLED_LOG_DIR="$RD"
cd "$REPO"

ARGS=()
for a in "$@"; do ARGS+=("${a//\{RUN\}/$RD}"); done

{
  echo "run_id: $ID"
  echo "start: $(date -Is)"
  echo "host: $(hostname)  user: ${USER:-?}"
  echo "cwd: $PWD"
  printf 'command:'; printf ' %q' "${ARGS[@]}"; echo
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null)"
  echo "git_describe: $(git describe --tags --always --dirty 2>/dev/null)"
} > "$RD/meta.txt"

git status --short > "$RD/git_status.txt" 2>&1
git diff > "$RD/git_diff.patch" 2>&1
env | grep -E '^(DLED_|DF40_|CUDA|PYTHONPATH|VIRTUAL_ENV)' | sort > "$RD/env.txt"
python -m pip freeze > "$RD/pip_freeze.txt" 2>/dev/null
python -c "import sys,torch,numpy;print('python',sys.version.split()[0]);print('torch',torch.__version__,'| cuda',torch.version.cuda,'| available',torch.cuda.is_available());print('numpy',numpy.__version__)" > "$RD/versions.txt" 2>&1
nvidia-smi > "$RD/nvidia_smi_start.txt" 2>&1
cp training/config/detector/dled.yaml training/config/train_config.yaml training/config/test_config.yaml "$RD/configs/" 2>/dev/null
( cd "${DLED_DATA:-$HOME/DLED/data}/DF40/json_file" 2>/dev/null && sha256sum ./*.json ) > "$RD/json_sha256.txt" 2>/dev/null

nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total --format=csv -l 30 > "$RD/gpu_usage.csv" 2>/dev/null &
MON=$!

BEFORE="$(ls -1d logs/*/ 2>/dev/null | sort)"
new_logger_dirs() { comm -13 <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$(ls -1d logs/*/ 2>/dev/null | sort)") | sed '/^$/d'; }
# link DLED's log folders into the run folder as soon as they appear. DLED makes
# two, seconds apart: train.py's logger (training.log) and the trainer's
# (test/<dataset>/ckpt_best.pth, metric pickles).
( for _ in $(seq 120); do sleep 10; k=0
    while read -r d; do
      [ -z "$d" ] && continue
      [ $k -eq 0 ] && l="$RD/dled_logs" || l="$RD/dled_logs_$(basename "$d")"
      [ -e "$l" ] || ln -sfn "$REPO/${d%/}" "$l"; k=$((k + 1))
    done <<< "$(new_logger_dirs)"
    [ $k -ge 2 ] && break; done ) &
WATCH=$!

START=$(date +%s)
"${ARGS[@]}" 2>&1 | tee "$RD/console.log"
RC=${PIPESTATUS[0]}
END=$(date +%s)
kill "$MON" "$WATCH" 2>/dev/null

# move DLED's logger folder(s) into the run folder; leave a link in ./logs
n=0
while read -r d; do
  [ -z "$d" ] && continue
  src="$REPO/${d%/}"; [ -L "$src" ] && continue
  dst="$RD/dled_logs"; [ $n -gt 0 ] && dst="$RD/dled_logs_$(basename "$src")"
  [ -L "$dst" ] && rm -f "$dst"                     # drop the live link before moving
  for l in "$RD"/dled_logs*; do [ -L "$l" ] && [ "$(readlink "$l")" = "$src" ] && rm -f "$l"; done
  mv "$src" "$dst" && ln -s "$dst" "$src"
  n=$((n + 1))
done <<< "$(new_logger_dirs)"

# results.txt: final numbers pulled from DLED's own log
LOG="$(ls "$RD"/DLED_*.log 2>/dev/null | head -1)"
if [ -n "$LOG" ]; then
  {
    echo "run_id: $ID   exit: $RC"
    echo "--- final epoch, per-class fused accuracy (last class index = unknown-category detection rate, DR)"
    grep -E "Class [0-9]+ fused accuracy" "$LOG" | tail -n "$(grep -cE 'Class [0-9]+ fused accuracy' "$LOG" | awk '{print ($1>=4)?4:$1}')"
    echo "--- accuracy per tested epoch"
    grep -E "fused_Accuracy|fused_Binary Accuracy" "$LOG" | tail -2
    echo "--- uncertainty thresholds (per tested epoch)"
    grep -E "Uncertainty threshold" "$LOG" | tail -1
  } > "$RD/results.txt" 2>/dev/null
fi
find "$RD" -name '*.pth' -printf '%s %p\n' 2>/dev/null | awk '{printf "checkpoint: %.1f MB  %s\n", $1/1e6, $2}' >> "$RD/results.txt"
ACC="$(grep -E 'fused_Accuracy' "$RD/results.txt" 2>/dev/null | tail -1 | sed 's/.*, //; s/\]//')"
DR="$(grep -E 'Class [0-9]+ fused accuracy' "$RD/results.txt" 2>/dev/null | tail -1 | awk '{print $NF}')"

{
  echo "end: $(date -Is)"
  echo "duration_s: $((END - START))"
  echo "exit_code: $RC"
} >> "$RD/meta.txt"
printf '%s  exit=%s  %smin  acc=%s  last_class_acc=%s  %s\n' "$ID" "$RC" "$(((END - START) / 60))" "${ACC:--}" "${DR:--}" "${ARGS[*]}" >> "$RUNS/RUNS_INDEX.txt"
echo "Run logged in $RD (exit $RC)"
exit "$RC"
