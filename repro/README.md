# DLED official reproduction — setup and run

Branch `official-repro` = the official DLED history (`zhyczy/DLED`, commit
`b8c41ea`) plus this `repro/` folder, a root `.gitignore`, and nothing else.
**No official file is modified.** Check it any time:

```bash
git diff --stat b8c41ea -- . ':!repro' ':!.gitignore'   # must print nothing
```

What is here and why: `FINDINGS.md` (paper-vs-code findings, including the
missing `loralib` and the missing train-JSON generator).

| File | Purpose |
| --- | --- |
| `loralib_shim/` | Reconstructed `loralib` (DLED imports it; the release omits it). Documented deviation |
| `requirements-dled.txt` | Dependencies derived from the actual imports (README's `requirements.txt` does not exist) |
| `env.sh` | Sets venv, PYTHONPATH, CLIP cache, data root, path rewriting; symlinks `./datasets` |
| `smoke_model.py` | Builds DLED and runs one GPU forward/backward on random tensors. No data needed |
| `df40_list.py` / `df40_get.sh` | List DF40 Drive files without downloading; download + extract one file |
| `noefs_data.py` | Build the NOEFS train/test JSONs; audit that every image the loader will read exists |

---

## Step 1 — Mac: put this branch on your GitHub

```bash
cd ~/Downloads                                    # where DLED-repro.bundle is
git clone -b official-repro DLED-repro.bundle DLED-repro
cd DLED-repro
git log --oneline | tail -3                       # must end with b8c41ea, 4967dd1, 4ffd8df
git remote set-url origin https://github.com/sahilpatil6305/DLED.git
git remote add upstream https://github.com/zhyczy/DLED.git
git push -u origin official-repro                 # new branch; your old main is untouched
```

## Step 2 — VM: clone and verify

```bash
cd ~/DLED
git clone -b official-repro https://github.com/sahilpatil6305/DLED.git DLED-repro
cd DLED-repro
git diff --stat b8c41ea -- . ':!repro' ':!.gitignore'   # must print nothing
nvidia-smi --query-gpu=name,memory.total --format=csv && df -h ~
```

## Step 3 — VM: environment and smoke test

```bash
cd ~/DLED/DLED-repro
source repro/env.sh                       # reuses ~/DLED/official/.venv310
pip install -r repro/requirements-dled.txt "numpy==1.26.4" --index-url https://pypi.org/simple
python -c "import torch,numpy,albumentations;print(torch.__version__,numpy.__version__,albumentations.__version__)"
python repro/smoke_model.py 2>&1 | tee ~/DLED/runs/smoke_model.log
```

Expect `48` LoRA modules, `3,932,160` LoRA params, a finite loss and
`SMOKE TEST PASSED`. Send the log before going further.

## Step 4 — VM: data (JSONs first, images second)

```bash
source repro/env.sh
python repro/df40_list.py json > ~/DLED/runs/df40_json_files.tsv && cat ~/DLED/runs/df40_json_files.tsv
```

Download the per-method JSON files (or the whole JSON folder) into
`$DLED_DATA/df40_json`, then:

```bash
python repro/noefs_data.py build --src $DLED_DATA/df40_json --dst $DLED_DATA/DF40/json_file
python - <<'EOF'
import json,os
d=json.load(open(os.path.expandvars('$DLED_DATA/DF40/json_file/DF40_FF_test_NOFE_all.json')))
k=next(iter(d)); s=next(iter(d[k].values())); v=next(iter(s['test'].values()))
print(v['frames'][0])
EOF
```

The printed path shows the prefix to rewrite. Set it and audit:

```bash
export DF40_OLD_ROOT=<prefix up to, not including, /DF40/...>
source repro/env.sh
python repro/noefs_data.py audit --json $DLED_DATA/DF40/json_file --out $DLED_DATA/audit
```

The audit lists missing frames per method. Download only those methods
(`df40_list.py train|test`, then `df40_get.sh <id> <dest>`), move each into
`$DLED_DATA/DF40/<method>/`, and re-run the audit. **Do not train until it
reports 0 missing** — the loader silently replaces missing images with sample 0.

## Step 5 — VM: pilot run (setup check, not a result)

```bash
DLED_NUM_EPOCHS=2 DLED_TEST_FRE=1 DLED_MAX_SAMPLES_PER_CLS=512 \
  DLED_LOG_DIR=~/DLED/runs/pilot ./train.sh NOEFS
tail -50 ~/DLED/runs/pilot/DLED_NOEFS.log
```

## Step 6 — VM: full baseline (official settings)

```bash
tmux new -s dled
source repro/env.sh
DLED_LOG_DIR=~/DLED/runs/NOEFS_baseline ./train.sh NOEFS --save_model --save_path ~/DLED/runs/NOEFS_baseline
# detach: Ctrl-b d     reattach: tmux attach -t dled
```

Paper target (FG 2026, Table I, EFS unseen): Acc 75.52, DR 34.71 — single seed,
final epoch. Record commit, `pip freeze`, env vars, JSON sha256s and the log.
