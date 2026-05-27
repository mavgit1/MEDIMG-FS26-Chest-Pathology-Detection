#!/usr/bin/env bash
set -euo pipefail

mkdir -p results

# Baselines only
python -m src.train --run_baselines_only --device cpu --out_dir results --runs_csv results/runs.csv

# Main model (CE, aug on)
python -m src.train --loss ce --use_aug 1 --seed 0 --out_dir results --runs_csv results/runs.csv

# Model 2 (focal, aug on)
python -m src.train --loss focal --use_aug 1 --seed 0 --out_dir results --runs_csv results/runs.csv

# Ablations (quick defaults; adjust epochs if needed)
python -m src.train --loss ce --use_aug 0 --seed 0 --out_dir results --runs_csv results/runs.csv
python -m src.train --loss ce --use_aug 1 --train_fraction 0.25 --seed 0 --out_dir results --runs_csv results/runs.csv

echo "Done. See results/runs.csv and results/*.png"

