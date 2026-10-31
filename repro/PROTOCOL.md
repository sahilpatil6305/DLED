# Research protocol (fixed 31 Oct 2026)

Rules every reported result follows. Changes to this file are dated and explained at the bottom.

## 1. Authentic implementations
- Each detector runs from its **official code at a pinned commit, unmodified** (DLED `b8c41ea`, DiCoME `191fc70`). Our code lives in `repro/` and only wraps the official model.
- Before a detector is attacked it passes two gates:
  1. **Reproduction gate**: a number from its own paper is reproduced with its official checkpoint or training recipe; any gap is reported (e.g. DLED DR 16.4 vs 34.7, FINDINGS F13).
  2. **Input-path gate**: our differentiable input path gives the official outputs (DLED: `test_freq_equivalence.py`; DiCoME: `detectors/dicome/check.py`, passed 31 Oct 2026 with zero difference).
- **Same training protocol for all detectors**: retrained on DF40 NOEFS with their official training code. Official checkpoints trained on other data are used only for the reproduction gate. Unavoidable deviations (e.g. fp16 instead of bf16 on a T4) are listed per detector.

## 2. Measurement
- Attack success is counted only from the **official decision code on saved 8-bit PNGs** (OFFICIAL column).
- Attacks must succeed on the 8-bit image, not only in continuous space (FINDINGS F14; `grid_verify`).
- Robustness = **worst case over all attacks run** (rejection-aware margin, CE, and APGD-style), per image.
- Checks reported with every table: uniform-noise baseline, monotonicity in eps, strong attack (more steps and restarts must not lower success on the same images), large-eps sanity.
- Model comparisons use the **same images** (`--samples_csv`), 95% Wilson intervals and exact McNemar tests.
- Claims about training methods need **at least 2 training seeds**.

## 3. Claims
- Decision rules are written down **before** a run (e.g. C3: at_reject must beat at_pgd and at_evat on E2/E3R/E4 at eps 1-4 beyond the intervals, without clean collapse). Failed rules are reported.
- Every number traces to a timestamped run folder in `~/DLED/runs` (`RUNS_INDEX.txt`). Inferred or unverified statements are marked as such.

## Changes
- 31 Oct 2026: created. Same day: C3 evaluated on the 50-step attacks -> failed; those attacks were then found to undercount (F14), so C3 is re-evaluated after the grid-verified re-runs.
