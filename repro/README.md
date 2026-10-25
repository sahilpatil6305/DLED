# DLED official reproduction — setup and run

Branch `main` = the official DLED history (`zhyczy/DLED`, commit `b8c41ea`,
tagged `baseline-b8c41ea`) plus this `repro/` folder, a root `.gitignore`, and
nothing else. **No official file is modified.** Check it any time:

```bash
git diff --stat baseline-b8c41ea -- . ':!repro' ':!.gitignore'   # must print nothing
```

What is here and why: `FINDINGS.md` (paper-vs-code findings, including the
missing `loralib` and the missing train-JSON generator).

| File | Purpose |
| --- | --- |
| `loralib_shim/` | Reconstructed `loralib` (DLED imports it; the release omits it). Documented deviation |
| `requirements-dled.txt` | Dependencies derived from the actual imports (README's `requirements.txt` does not exist) |
| `env.sh` | Sets venv, PYTHONPATH, CLIP cache, data root, path rewriting; symlinks `./datasets` |
| `smoke_model.py` | Builds DLED and runs one GPU forward/backward on random tensors. No data needed |
| `run.sh` | Runs any command with a full timestamped record in `~/DLED/runs/` |
| `fetch_assets.sh` | Fetches and checksums the CLIP tokenizer vocab the release omits |
| `get_test.sh` | Downloads, verifies and extracts the 35 NOEFS test zips via rclone |
| `df40_list.py` / `df40_get.sh` | List DF40 Drive files without downloading; download + extract one file |
| `noefs_data.py` | Build the NOEFS train/test JSONs; audit that every image the loader will read exists |

---

## Step 1 — VM: clone and verify

```bash
mkdir -p ~/DLED && cd ~/DLED
/usr/bin/python3.10 -m venv ~/DLED/venv          # own venv; no sudo, nothing system-wide
source ~/DLED/venv/bin/activate
python -m pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple
git clone https://github.com/sahilpatil6305/DLED.git DLED-repro    # private: username + read-only token
cd DLED-repro
git diff --stat baseline-b8c41ea -- . ':!repro' ':!.gitignore'   # must print nothing
```

## Step 2 — every new SSH session

```bash
cd ~/DLED/DLED-repro && source repro/env.sh
```

## Step 3 — VM: environment and smoke test

```bash
cd ~/DLED/DLED-repro
source repro/env.sh          # activates ~/DLED/venv; first time also fetches the CLIP vocab
pip install -r repro/requirements-dled.txt --index-url https://pypi.org/simple
repro/run.sh smoke python repro/smoke_model.py
```

Expect `48` LoRA modules, `3,932,160` LoRA params, a finite loss and
`SMOKE TEST PASSED`.

**Every run goes through `repro/run.sh <name> <command…>`.** It creates
`~/DLED/runs/<date-time>_<name>/` with the command, timings, exit code, git
commit and diff, environment, package versions, config copies, dataset-JSON
fingerprints, console output, GPU usage every 30 s, and DLED's own logs; and it
appends one line to `~/DLED/runs/RUNS_INDEX.txt`.

## Step 4 — VM: data

Anonymous Google Drive downloads (gdown) hit Drive's "too many users" limit for
DF40's large zips, so the images are fetched with rclone signed in to a Google
account with **read-only** Drive access. Revoke it afterwards at
myaccount.google.com/permissions.

1. JSON indexes (small; gdown is fine): `python repro/df40_list.py json`, download
   the 35 per-method JSONs listed in `noefs_data.TEST_NOFE_ALL` into
   `$DLED_DATA/df40_json`, then
   `python repro/noefs_data.py build --src $DLED_DATA/df40_json --dst $DLED_DATA/DF40/json_file`.
2. rclone: download the Linux binary from downloads.rclone.org into `~/DLED/tools`,
   `rclone --config ~/DLED/tools/rclone.conf config create gdrive drive scope=drive.readonly config_is_local=false`,
   get a token on a machine with a browser (`rclone authorize "drive"`), then
   `rclone --config ~/DLED/tools/rclone.conf config reconnect gdrive:` and paste it.
3. Training fakes (10 zips, 20.4 GB) from folder `1U8meBbqVvmUkc5GD0jxct6xe6Gwk9wKD`
   with `rclone copy … --include "/{fsgan,faceswap,simswap,inswap,blendface,fomm,facevid2vid,wav2lip,MRAA,one_shot_free}.zip"`,
   verify with `rclone check … --one-way`, unzip into `$DLED_DATA/DF40_train/`.
4. Real frames: `rclone backend copyid gdrive: 1dHJdS0NZ6wpewbGA5B0PdIBS9gz28pdb $DLED_DATA/_incoming/real/`,
   unzip into `$DLED_DATA` (the zip starts with `FaceForensics++/…`).
5. Test fakes (35 zips, 98 GB, Celeb-DF parts skipped):
   `nohup repro/get_test.sh > ~/DLED/runs/get_test.log 2>&1 &`
6. MidJourney (official `change_json_path.py` rule): `python repro/noefs_data.py gif2png`
   converts its GIFs to PNG; `build` already rewrote the paths and dropped the 5 broken images.
7. Audit until both splits report 0 missing — the loader silently replaces
   missing images with sample 0:
   `repro/run.sh audit python repro/noefs_data.py audit --json $DLED_DATA/DF40/json_file --out $DLED_DATA/audit`

Resulting layout (`$DLED_DATA` = `~/DLED/data`):

```
DF40_train/<method>/frames/<video>/<frame>.png          training fakes
DF40/<method>/ff/frames/<video>/<frame>.png             test fakes (FS, FR)
DF40/<method>/ff/<id>/<image>.png                       test fakes (EFS)
DF40/{deepfacelab,heygen,MidJourney,whichisreal}/{fake,real}/...
FaceForensics++/original_sequences/youtube/c23/frames/  real frames
DF40/json_file/DF40_FF_{train_NOEFS,test_NOFE_all}.json
```

## Step 5 — VM: pilot run (setup check, not a result)

```bash
repro/run.sh pilot env DLED_NUM_EPOCHS=2 DLED_TEST_FRE=1 DLED_MAX_SAMPLES_PER_CLS=512 ./train.sh NOEFS
```

## Step 6 — VM: full baseline (official settings, survives logout)

```bash
nohup repro/run.sh baseline_noefs ./train.sh NOEFS --save_model --save_path {RUN}/checkpoints > /dev/null 2>&1 &
tail -f ~/DLED/runs/RUNS_INDEX.txt     # one line appears when it finishes
```

Paper target (FG 2026, Table I, EFS unseen): Acc 75.52, DR 34.71 — single seed,
final epoch.

## Step 7 — attack harness (research code, `repro/attack/`)

`freq_torch.py` is a differentiable copy of the loader's input pipeline (spatial
normalisation + FFT log-magnitude map, uint8 truncation via a straight-through
estimator). The attacker perturbs the 224×224 image; a saved 224×224 PNG goes
back through the official loader unchanged. CPU-only equivalence test, safe to
run while the GPU trains:

```bash
repro/run.sh freq_equiv python repro/attack/test_freq_equivalence.py
```

Expect `FREQ EQUIVALENCE PASSED`: max 1 grey level difference on ~8e-4 of
pixels (torch vs NumPy FFT rounding at integer boundaries; measured
2026-10-24), identical spatial input, exact PNG round trip, finite STE
gradients, and zero gradients for naive floor (gradient masking if not
handled). Because of that small mismatch, the torch map is used only to
compute attacks; every reported number comes from the official loader.

`dled_attack.py` / `run_attack.py`: rejection-aware PGD (error types E1, E2,
E3, E3R, E4 defined at the top of `dled_attack.py`). The attack runs on a float32
copy; each adversarial image is rounded to 8 bits, saved as PNG, re-read by the
official loader and decided by the official `model_inference` with the
thresholds from the training log. CPU smoke test (works mid-training, uses the
epoch-0 checkpoint):

```bash
repro/run.sh attack_smoke python repro/attack/run_attack.py --run <baseline run folder> --device cpu --n_per_class 2 --steps 5 --eps 8 --batch 2
```

## Step 8 — adversarial training (research code, `repro/adv_train/`)

`at_train.py` fine-tunes the trained DLED checkpoint with one of four modes, using DLED's
own model, loss and optimisers: `clean` (control: same fine-tuning, no attack), `pgd`
(Madry PGD-AT), `evat` (EV-AT-style evidential alignment), `reject` (rejection-aware: the
adversary pushes known inputs into rejection and fakes towards "real"). The attack
perturbs the un-augmented 224×224 image, as at test time. It saves the complete weights
(including the frozen LoRA base weights DLED's checkpoints omit, F13) and recomputes the
thresholds the official way into `training.log`, so the attack harness evaluates it unchanged.

```bash
repro/run.sh at_smoke repro/adv_train/smoke_at.sh ~/DLED/runs/20261024-152447_baseline_noefs   # CPU, ~15-30 min
nohup repro/adv_train/queue_at.sh ~/DLED/runs/20261024-152447_baseline_noefs > ~/DLED/runs/queue_at.log 2>&1 &   # GPU, ~1.5 days
```
