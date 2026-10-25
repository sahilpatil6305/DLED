# Official DLED — verified findings and VM runbook

Source inspected: `zhyczy/DLED` @ `b8c41ea` (same commit as your VM clone), read file by file on 23 Oct 2026.
Nothing here has been run on a GPU yet; items marked **[run on VM]** are what the scripts in this kit check.

---

## 1. Findings that change the plan

### F1. The official release cannot run as published — `loralib` is missing
`training/detectors/dled.py:30` imports `apply_lora, apply_double_lora, INDEX_POSITIONS_TEXT, INDEX_POSITIONS_VISION` from `loralib.utils`. There is no `loralib/` in the repo, and PyPI `loralib` (Microsoft) has none of those names. The only public match is CLIP-LoRA (MaxZanella/CLIP-LoRA), but its `apply_lora(args, clip_model)` has a different signature from DLED's call `apply_lora(cfg, encoder, indices, dtype)`.

**Consequence:** any DLED run requires a reconstructed module. `loralib_shim/` provides one (CLIP-LoRA's attention layer, unmodified, behind DLED's signature). This is a necessary, documented deviation — state it in the thesis. Assumptions it makes are listed at the top of `loralib_shim/loralib/utils.py`; the one most likely to affect numbers is LoRA scaling (CLIP-LoRA uses α/√r = 0.25; Microsoft loralib uses α/r = 0.0625).

**Reframes the history:** the Kaggle "custom LoRA" smoke test reported 48 replacements and 3,932,160 LoRA params. That is exactly what the *official* config (`ENABLE_LORA [q,k,v,o]`, `RANK 16`, `POSITION all`, 4 ViT-B/16 encoders) produces. That branch was very likely a reconstruction of this missing module, not an unofficial extension. Compare its LoRA code with this shim before discarding it.

### F2. `requirements.txt` does not exist
README says `pip install -r requirements.txt`; the file is absent. `requirements-dled.txt` here is derived from the actual imports. Critical pin: **albumentations 1.3.1** (the loader uses `ImageCompression(quality_lower=…)` and `crops.functional.crop`, both gone in 2.x).

### F3. `DF40_FF_train_NOEFS.json` has no generator in the repo
`train.sh NOEFS` needs `DF40_FF_train_NOEFS`; no script produces that name (`organize_json.py` only produces `DF40_mixed_train_NOEFS`). The test file `DF40_FF_test_NOFE_all` *is* produced by `merge_seen_unseen.py`. `noefs_data.py build` reproduces the test file exactly and reconstructs the train file from the NOEFS method list — a second documented inference.

### F4. What NOEFS actually needs (storage scope)
| Split | Methods (FF++-derived `_ff` versions only) | Split key |
|---|---|---|
| Train | FS: fsgan, faceswap, simswap, inswap, blendface · FR: fomm, facevid2vid, wav2lip, MRAA, one_shot_free (+ their Real frames) | `train` |
| Test | 35 sources: 9 FS + deepfacelab, 12 FR + heygen, 10 EFS + MidJourney + whichisreal | `test` |

Not needed for NOEFS: any `_cdf` data, any Face-Editing data (e4e, CollabDiff, StarGAN, StarGANv2, StyleCLIP), and the `val` split. Also the loader reads **only the first 32 sorted frames per video** (the `step` computation always yields 1), so frames beyond 32 per video are never touched. Both facts can shrink the download substantially — measure with the audit before downloading everything.

Labels in the test set: 0 real, 1 FS, 2 FR, **3 EFS = the unknown class** for the 3-class model. Unseen FS/FR *methods* keep labels 1/2 (known category, unseen generator).

### F5. Silent data corruption risk
If an image is missing or unreadable, `__getitem__` prints a line and returns sample 0 instead. A half-downloaded dataset trains without error and produces quietly wrong numbers. **Do not train until `noefs_data.py audit` reports 0 missing.**

### F6. Protocol properties to report honestly
- **No validation split.** "Best metric" and checkpoint selection are computed on the test set (`test_fre`). With the defaults (`test_fre=50`, 50 epochs) testing happens once, at the end — report that setting; don't lower `test_fre` and then pick the best epoch.
- **Test set is subsampled** by `balance_dataset` (random, seeded via `manualSeed=1024`), capped at 13,504 per class.
- **Open-set rule:** per-class uncertainty thresholds are set from the training data (top 5% vacuity, `threshold_ratio=0.05`); a test sample whose vacuity exceeds its predicted class's threshold is relabelled as class `num_classes` (unknown).
- **Checkpoint saved only at the end** (`--save_model`) — a crash at epoch 49 loses the run. Use `tmux`/`nohup`.
- `train_batchSize 32`, `PREC fp16`, `workers 8` (VM has ~15 GiB RAM; reduce workers if it OOMs — record any change).

### F7. Directly relevant to the adversarial extension
The frequency inputs (FFT log-magnitude spectrum, high-pass) are computed **in NumPy/OpenCV inside the data loader**, from the *un-augmented* image. A PGD attack on the RGB tensor therefore **does not propagate through the frequency branch** unless you re-implement `load_rgb`'s FFT + median-blur high-pass in differentiable PyTorch. Two consequences:
1. A naive attack on `data_dict['image']` only perturbs the spatial branch — it is a partial threat model, and it would make DLED look artificially robust.
2. The extension must define the threat model in pixel space and push the perturbation through *both* branches (and through the subjective-logic fusion and vacuity rule). This is a concrete, defensible design point for the thesis.

Also note `ImageCompression`, `normalize` with mean/std 0.5 (the dled.yaml `INPUT.PIXEL_MEAN` CLIP values are **not** used by this loader) — your ε must be stated in [0,1] pixel units before normalisation.

### F8. Minor
- NOFS protocol looks broken: its class names map EFS to index 1, but `label_dict` gives EFS label 3. Not your protocol; mention only if you run it.
- CLIP-LoRA code is AGPL-3.0 — fine for research use; credit it.
- I could not read the DLED GitHub issue tracker from my environment. Check https://github.com/zhyczy/DLED/issues for a released `loralib`, requirements, or JSONs before relying on the reconstructions; if nothing is there, consider emailing the authors (Zhongyi Cai et al.) about F1 and F3.

---


### F9. Two more files the release omits (found while running)
- `clip/bpe_simple_vocab_16e6.txt.gz`, the CLIP tokenizer vocabulary. DLED's `clip/simple_tokenizer.py` is byte-identical to OpenAI CLIP's and loads this file at import, so nothing runs without it. `repro/fetch_assets.sh` copies OpenAI's original (sha256 `924691ac…804a`). Packaging gap, not a method change.
- The per-method DF40 JSON files use the path prefix `deepfakes_detection_datasets`; `env.sh` now sets `DF40_OLD_ROOT` accordingly.

### F10. Data facts measured on the real files (24 Oct 2026)
- NOEFS train: 470,526 candidate frames (real 229,930 · FS 126,518 · FR 114,078); official balancing uses 54,016 per epoch.
- NOEFS test: 302,502 candidate frames over 35 sources; balancing uses 81,024 (real 40,512; FS, FR, EFS 13,504 each).
- Drive sizes: training zips for the 10 NOEFS methods 20.4 GB; the 35 test zips 98.1 GB (ff + cdf; cdf is not used).
- In DF40's training Drive folder, `StyleGAN2.zip` and `StyleGAN3.zip` have identical byte sizes (1,445,525,543): possibly a duplicate upload. Not used by NOEFS training; check the test versions before relying on them.

### F11. Found by the pilot run (24 Oct 2026)
- **NumPy must be ≤ 1.23.** `trainer.py` builds `np.array()` from per-batch lists with a shorter last batch; NumPy ≥ 1.24 raises *inhomogeneous shape*. The NOEFS test set (81,024 frames, batch 100) always ends with a 24-frame batch, so newer NumPy crashes after the full 50 epochs. Pinned `numpy==1.23.5` (environment change, no code change).
- **MidJourney needs the official `change_json_path.py` fix-up.** Some MidJourney files are GIFs, which OpenCV cannot read (the loader then silently substitutes sample 0), and 5 images are broken. Reproduced exactly: GIF→PNG on disk, `.gif`→`.png` in paths, the 5 listed images dropped. The audit now also flags unreadable extensions.
- **`mobileswap.zip` (test) contains a nested `ff/frames.zip`** whose images sit under the authors' internal path `…/deepfakes_detection_datasets/DF40/mobileswap/ff/frames/`; moved into place (27,279 images, exactly the missing count).
- **StyleGAN2 and StyleGAN3 test images (FF domain) are identical** (same 3,711 file names; first 200 byte-identical — full comparison pending). Kept as released for the baseline; report per-generator results with this caveat.

## F12. Rejection thresholds (relevant to attacks)

- The per-class vacuity thresholds are **not saved** with the checkpoint; they exist only as the log line `Retrieved uncertainty dict {...}` (trainer.calculate_uthre). The attack harness parses the last such line, which belongs to the `ckpt_best.pth` saved at the same test epoch.
- `calculate_uthre` runs on the **augmented training loader** with the model still in train mode (no `setEval`). LoRA dropout is 0 and CLIP has no BatchNorm, so train vs eval mode does not change outputs; augmentation does.
- Fused rule: accepted as c ⇔ c = argmax e and 1/(1+e_c) ≤ τ_c ⇔ log(1+e_c) ≥ −log τ_c. Rejection depends only on the top evidence, so an attacker who can raise one class's evidence past its gate is accepted regardless of the other classes.
## F13. Checkpoints omit the LoRA base weights; in-training weights drift in fp16

- `LinearLoRA.forward` (dropout 0, train mode) calls `merge_lora_param`, which `delattr`s the frozen `weight` Parameter and puts back a plain tensor `p + s·BA`, then `sub_lora_data` subtracts `s·BA` again. After the first step, the q/k/v/out weights of all 48 LoRA'd attention layers are no longer parameters, so `ckpt_best.pth` and the `--save_model` file do not contain them (343 MB instead of ~600 MB). Reloading takes them from pretrained CLIP. This is inherited from CLIP-LoRA's code (DLED's release omits loralib; our shim copies it).
- In fp16, `(p + d) − d ≠ p` in general, so the "frozen" weights take a rounding random walk every step. A simulation with CLIP-like weights gives ≈0.26% mean relative drift after 3,000 steps, growing like √steps (estimate ≈1–1.5% after 84,400 steps). The official in-training test uses the drifted weights; a reloaded checkpoint uses pristine ones. **Measured (25 Oct 2026, `eval_reload.py`, run 20261025-141229_reload_check):** the test set is rebuilt exactly (binary label sequence identical over 81,024 images; per-sample binary agreement 94.5%, far above the 83.8% expected for unrelated images at these accuracies). So every difference is the model: reloaded vs logged = Acc 76.93 vs 77.41, real 96.16 vs 91.84, FS 85.07 vs 83.56, FR 77.19 vs 88.94 (−11.8), DR 10.83 vs 16.43 (−5.6); 5.5% of binary decisions flip. **The released checkpoint is not the model that produced the logged numbers.** Clean confusion (reloaded): 69.1% of unseen-EFS fakes are accepted as real, 19.2% as FS/FR, 10.8% rejected. The logged thresholds were calibrated on the drifted model; recalibrating them on the reloaded model is a fairer 'released' baseline (to do).

## F12 (cont.)

- Frequency map in torch vs official NumPy: ≤1 grey level on 8.4e-4 of pixels (48 DF40 frames). Attacks use the torch map; all reported numbers use the official loader on saved 8-bit PNGs.

---

Setup and run steps: see `repro/README.md`.
