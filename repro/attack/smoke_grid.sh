#!/usr/bin/env bash
# CPU smoke test of grid-verified PGD (FINDINGS F14). Safe next to GPU jobs (CPU only, 2 threads).
# Same 6 known images (2 per label), E2 at 2/255, 10 steps: old behaviour vs grid_verify.
# Pass: both runs finish, and the grid_verify run has official success >= the old run and
# every grid_verify in-loop hit is also an official success (LOST_AT_ROUNDING close to 0).
#   repro/attack/smoke_grid.sh ~/DLED/runs/20261027-042301_at_pgd
set -euo pipefail
RUN=${1:?usage: smoke_grid.sh <AT run folder with at_last.pth and training.log>}
S=~/DLED/runs/20261026-001437_paper_margin/samples.csv
OUT=~/DLED/runs/smoke_grid_$(date +%Y%m%d-%H%M%S)
COMMON=(--ckpt "$RUN/at_last.pth" --log "$RUN/training.log" --samples_csv "$S" --n_per_class 2
        --attacks E2 --eps 2 --steps 10 --device cpu --threads 2 --batch 6 --keep_examples 0)
python repro/attack/run_attack.py "${COMMON[@]}" --no_grid_verify --out "$OUT/old" > "$OUT.old.log" 2>&1 \
  || { mkdir -p "$OUT"; tail -20 "$OUT.old.log"; exit 1; }
python repro/attack/run_attack.py "${COMMON[@]}" --out "$OUT/grid" > "$OUT.grid.log" 2>&1 \
  || { tail -20 "$OUT.grid.log"; exit 1; }
python - "$OUT" <<'EOF'
import csv, sys
for v in ('old', 'grid'):
    r = list(csv.DictReader(open(f'{sys.argv[1]}/{v}/per_sample.csv')))
    hit = [x for x in r if int(x['first_success_step']) >= 0]
    off = sum(int(float(x['success_official'])) for x in r)
    lost = sum(1 for x in hit if int(float(x['success_official'])) == 0)
    print(f'{v:4s} n {len(r)}  hit_in_loop {len(hit)}  official {off}  LOST_AT_ROUNDING {lost}')
print('SMOKE GRID DONE ->', sys.argv[1])
EOF
