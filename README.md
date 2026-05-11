# Open Set Face Forgery Detection via Dual-Level Evidence Collection (OSFFD)

Official implementation of *Open Set Face Forgery Detection via Dual-Level
Evidence Collection* (FG 2026).

This work proposes the DualLevel Evidential face forgery Detection (DLED) approach, a CLIP-based detector with two parallel branches (spatial / frequency) and fuses them via subjective-logic evidence combination. 


## Repository layout

```
DLED/
├── train.sh                       # unified launcher (replaces protocol1-8)
├── Dassl/                         # third-party Dassl toolbox
├── clip/                          # CLIP model
├── preprocessing/                 # scripts to build the DF40 JSON indices
└── training/
    ├── train.py                   # training entry-point
    ├── test.py                    # evaluation entry-point
    ├── config/
    │   ├── train_config.yaml
    │   ├── test_config.yaml
    │   └── detector/dled.yaml
    ├── dataset/                   # DeepfakeFrequencyDataset etc.
    ├── detectors/dled.py
    ├── trainer/trainer.py
    ├── loss/edl_loss.py
    ├── metrics/                   # registry + metric helpers
    └── optimizor/                 # SAM, LinearLR
```

## Setup

```bash
pip install -r requirements.txt   # PyTorch, albumentations, timm, ftfy, regex, yacs, ...
```

The DF40 dataset must be available locally. Generate the JSON index files via
the scripts under `preprocessing/`, then point the config at them.

### Environment variables

| Variable                   | Purpose                                                                                  |
|----------------------------|------------------------------------------------------------------------------------------|
| `DLED_PRETRAINED_ROOT`     | Directory used to cache CLIP weights. Defaults to `training/pretrained/clip`.            |
| `DF40_DATA_ROOT`           | Root directory of the DF40 dataset on this machine.                                      |
| `DF40_OLD_ROOT`            | Root path embedded in the JSON index (if it differs from `DF40_DATA_ROOT`, paths are rewritten on the fly). |
| `DLED_LOG_DIR`             | Output directory for training logs (default: `./logs`).                                  |
| `DLED_LOCAL_RANK`          | CUDA device index (default: `0`).                                                        |
| `DLED_NUM_EPOCHS`          | Number of training epochs (default: `50`).                                               |
| `DLED_TEST_FRE`            | Test frequency in epochs (default: `50`).                                                |
| `DLED_MAX_SAMPLES_PER_CLS` | Cap on the number of samples per training class (default: `13504`).                      |

Also edit `training/config/train_config.yaml` so that `dataset_json_folder` and
`log_dir` point at the right locations on your machine.

## Training

The single launcher `train.sh` supersedes the old `train_protocol{1..8}.sh`
files. Pick the protocol you want to run:

```bash
# 3-class same-domain protocol with Entire-Face-Synthesis held out (paper Table 1)
./train.sh NOEFS

# 3-class same-domain protocol with Face-Swapping held out
./train.sh NOFS

# 3-class same-domain protocol with Face-Reenactment held out
./train.sh NOFR

# 4-class open-test protocol with Face-Editing held out (paper Table 2)
./train.sh open
```

Extra arguments after the protocol name are forwarded to `training/train.py`,
e.g.:

```bash
./train.sh NOEFS --save_model --save_path ./saved_models/NOEFS
```

## Evaluation

```bash
python3 training/test.py \
    --detector_path ./training/config/detector/dled.yaml \
    --train_dataset DF40_FF_train_NOEFS \
    --test_dataset  DF40_FF_test_NOFE_all \
    --num_classes 3 \
    --model_path  ./saved_models/NOEFS/<checkpoint>.pth
```

## Citation

If you use this code, please cite the paper.

@article{cai2025open,
  title={Open Set Face Forgery Detection via Dual-Level Evidence Collection},
  author={Cai, Zhongyi and Gernon, Bryce and Bao, Wentao and Li, Yifan and Wright, Matthew and Kong, Yu},
  journal={arXiv preprint arXiv:2512.04331},
  year={2025}
}