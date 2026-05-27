#!/usr/bin/env bash
# Compare CE training: augmentation off vs on (flip/rotate/jitter + RandomErasing).
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

RUNS=results/runs.csv
mkdir -p results checkpoints

echo "=== CE, no augmentation ==="
python -m src.train --device cuda --loss ce --use_aug 0 --seed 0 --epochs 10 \
  --runs_csv "$RUNS" --out_dir results

echo "=== CE, augmentation + RandomErasing ==="
python -m src.train --device cuda --loss ce --use_aug 1 --seed 0 --epochs 10 \
  --runs_csv "$RUNS" --out_dir results

echo "=== Grad-CAM (aug-on checkpoint, fixed manifest) ==="
python -m src.gradcam_export --device cuda \
  --ckpt checkpoints/resnet18_ce_aug1_frac1.0_seed0.pt --out_dir results

echo "Done. Compare test rows: resnet18_ce_aug0 vs resnet18_ce_aug1 in $RUNS"
