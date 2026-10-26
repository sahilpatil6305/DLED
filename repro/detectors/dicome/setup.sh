#!/usr/bin/env bash
# One-time setup of DiCoME (Kang et al., ICML 2026) next to DLED, without touching DLED's venv.
# Everything stays inside ~/DLED: code in ext/DiCoME (pinned commit), venv in venv_dicome,
# Hugging Face cache in cache/hf. No sudo. Run (CPU/network only, safe next to GPU jobs):
#   nohup repro/detectors/dicome/setup.sh > ~/DLED/runs/setup_dicome.log 2>&1 &
set -euo pipefail
ROOT="$HOME/DLED"
COMMIT=191fc70                       # DiCoME main, 2026-06-02 (checked 2026-10-26)
export HF_HOME="$ROOT/cache/hf"
mkdir -p "$ROOT/ext" "$HF_HOME"
df -h "$ROOT" | tail -1

if [ ! -d "$ROOT/ext/DiCoME/.git" ]; then
  git clone https://github.com/kxl0825/DiCoME.git "$ROOT/ext/DiCoME"
fi
git -C "$ROOT/ext/DiCoME" checkout -q "$COMMIT"
echo "DiCoME at $(git -C "$ROOT/ext/DiCoME" log -1 --format='%h %ci')"

if [ ! -x "$ROOT/venv_dicome/bin/python" ]; then
  /usr/bin/python3.10 -m venv "$ROOT/venv_dicome"
fi
source "$ROOT/venv_dicome/bin/activate"
python -m pip install -q --upgrade pip
# versions from DiCoME's requirements.txt (torch 2.6 + cu118, as verified by the authors)
python -m pip install -q torch==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -q lightning==2.5.0 torchmetrics==1.8.2 transformers==4.50.0 peft==0.14.0 \
    accelerate==1.10.1 h5py==3.14.0 numpy==2.2.6 pandas==2.3.2 scikit-learn==1.6.1 scipy==1.15.3 \
    Pillow==10.2.0 tqdm PyYAML==6.0.2 pydantic==2.9.2 rich==13.9.4 fire==0.7.1 huggingface_hub
python - <<'EOF'
import os
from huggingface_hub import hf_hub_download
from transformers import CLIPModel, CLIPProcessor
d = os.path.expanduser('~/DLED/ext/DiCoME/weights')
p = hf_hub_download('kxl0825/DiCoME', 'dicome-best.ckpt', local_dir=d)
print('checkpoint:', p, round(os.path.getsize(p) / 2**20), 'MB')
CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
CLIPModel.from_pretrained('openai/clip-vit-large-patch14')
print('CLIP ViT-L/14 cached in', os.environ['HF_HOME'])
import torch, numpy
print('torch', torch.__version__, '| cuda build', torch.version.cuda, '| numpy', numpy.__version__)
EOF
python -m pip freeze > "$ROOT/ext/DiCoME/pip_freeze_venv_dicome.txt"
echo "DICOME SETUP DONE"
