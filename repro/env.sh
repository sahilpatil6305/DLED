# Source from the repo root:  source repro/env.sh
# Sets every path DLED reads. Override any variable before sourcing.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Python environment: reuse the verified one (torch 2.2.2+cu121, numpy 1.23.5)
DLED_VENV="${DLED_VENV:-$HOME/DLED/venv}"
if [ -f "$DLED_VENV/bin/activate" ]; then
    source "$DLED_VENV/bin/activate"
else
    echo "WARNING: venv not found at $DLED_VENV (set DLED_VENV)" >&2
fi

# Reconstructed loralib (see repro/loralib_shim/loralib/utils.py header)
export PYTHONPATH="$REPO_ROOT/repro/loralib_shim${PYTHONPATH:+:$PYTHONPATH}"

# CLIP ViT-B/16 weights cache (downloaded once, ~335 MB)
export DLED_PRETRAINED_ROOT="${DLED_PRETRAINED_ROOT:-$HOME/DLED/cache/clip}"
mkdir -p "$DLED_PRETRAINED_ROOT"

# Data root. Layout: $DLED_DATA/DF40/<method>/{ff,cdf}/...  and  $DLED_DATA/DF40/json_file/*.json
export DLED_DATA="${DLED_DATA:-$HOME/DLED/data}"
mkdir -p "$DLED_DATA/DF40/json_file"

# Path rewriting done by the official loader: every JSON frame path has
# DF40_OLD_ROOT replaced by DF40_DATA_ROOT. The DF40 JSONs use the prefix
# 'deepfakes_detection_datasets' (checked on the real files, Oct 2026).
export DF40_DATA_ROOT="${DF40_DATA_ROOT:-$DLED_DATA}"
export DF40_OLD_ROOT="${DF40_OLD_ROOT:-deepfakes_detection_datasets}"

# train_config.yaml expects ./datasets/DF40/json_file; a symlink avoids editing it
if [ ! -e "$REPO_ROOT/datasets" ]; then
    ln -s "$DLED_DATA" "$REPO_ROOT/datasets"
fi

export DLED_RUNS="${DLED_RUNS:-$HOME/DLED/runs}"
export DLED_LOG_DIR="${DLED_LOG_DIR:-$DLED_RUNS/logs}"
mkdir -p "$DLED_LOG_DIR"

# One-time fetch of files the release omits (CLIP tokenizer vocab)
[ -f "$REPO_ROOT/clip/bpe_simple_vocab_16e6.txt.gz" ] || bash "$REPO_ROOT/repro/fetch_assets.sh"

echo "DLED env: repo=$REPO_ROOT data=$DLED_DATA clip=$DLED_PRETRAINED_ROOT"
echo "          DF40_OLD_ROOT='${DF40_OLD_ROOT}' -> DF40_DATA_ROOT='${DF40_DATA_ROOT}'"
python -c "import torch; print('          torch', torch.__version__, '| cuda', torch.cuda.is_available())" 2>/dev/null
