#!/usr/bin/env bash
# Run any command with a complete, timestamped record.
#
#   repro/run.sh <run-name> <command...>
#
# Examples (from the repo root, after `source repro/env.sh`):
#   repro/run.sh smoke  python repro/smoke_model.py
#   repro/run.sh audit  python repro/noefs_data.py audit --json $DLED_DATA/DF40/json_file --out $DLED_DATA/audit
#   repro/run.sh pilot  env DLED_NUM_EPOCHS=2 DLED_TEST_FRE=1 DLED_MAX_SAMPLES_PER_CLS=512 ./train.sh NOEFS
#   nohup repro/run.sh baseline ./train.sh NOEFS --save_model --save_path ~/DLED/runs/ckpt > /dev/null 2>&1 &
#
# Creates $DLED_RUNS/<YYYYmmdd-HHMMSS>_<name>/ (default $HOME/DLED/runs) with:
#   meta.txt (command, times, duration, exit code, git commit), console.log,
#   env.txt, versions.txt, pip_freeze.txt, nvidia_smi_start.txt, gpu_usage.csv,
#   configs/ (YAML copies), json_sha256.txt, git_status.txt, git_diff.patch,
#   and links to the logger folders train.py creates under ./logs.
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

{
  echo "run_id: $ID"
  echo "start: $(date -Is)"
  echo "host: $(hostname)  user: ${USER:-?}"
  echo "cwd: $PWD"
  printf 'command:'; printf ' %q' "$@"; echo
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
START=$(date +%s)
"$@" 2>&1 | tee "$RD/console.log"
RC=${PIPESTATUS[0]}
END=$(date +%s)
kill "$MON" 2>/dev/null

AFTER="$(ls -1d logs/*/ 2>/dev/null | sort)"
comm -13 <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER") | while read -r d; do
  [ -n "$d" ] && ln -s "$REPO/${d%/}" "$RD/train_logger_$(basename "$d")"
done

{
  echo "end: $(date -Is)"
  echo "duration_s: $((END - START))"
  echo "exit_code: $RC"
} >> "$RD/meta.txt"
printf '%s  exit=%s  %smin  %s\n' "$ID" "$RC" "$(((END - START) / 60))" "$*" >> "$RUNS/RUNS_INDEX.txt"
echo "Run logged in $RD (exit $RC)"
exit "$RC"
