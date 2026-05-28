# Chest X-Ray Pneumonia Screening (MEDIMG FS26)

Minimal, reproducible chest X-ray **NORMAL vs PNEUMONIA** classifier using a public Hugging Face dataset.

## Dataset

- HF: `hf-vision/chest-xray-pneumonia` (`revision=refs/convert/parquet`)
- Validation: 10% stratified hold-out from **train** (HF validation set is tiny)
- Test: untouched HF test split

## Setup

```bash
bash scripts/setup.sh
source .venv/bin/activate
```

Optional overrides: copy `configs/reproduce.env.example` → `configs/reproduce.env`.

## Reproducible pipeline (stages)

Run named stages in order. Each training stage logs to `results/runs.csv` with a `stage` column.

```bash
bash scripts/reproduce.sh list          # show all stages
bash scripts/reproduce.sh clean         # wipe results/* and checkpoints/*
bash scripts/reproduce.sh baselines     # random, majority, frozen ResNet18 + LR
bash scripts/reproduce.sh train_ce_no_aug   # CE, no augmentation (before aug)
bash scripts/reproduce.sh train_ce_aug      # CE + aug + RandomErasing (after aug)
bash scripts/reproduce.sh train_focal_aug   # focal loss + augmentation (model 2)
bash scripts/reproduce.sh ablation_data_25  # CE + aug, 25% train data
bash scripts/reproduce.sh gradcam_manifest  # build configs/gradcam_manifest.json
bash scripts/reproduce.sh gradcam_compare   # aug off vs on figure
bash scripts/reproduce.sh all             # clean + full pipeline (SEEDS=0)
```

Multi-seed robustness (CE + focal, aug on):

```bash
SEEDS="0 1 2" bash scripts/reproduce.sh robustness
```

Environment variables: `DEVICE`, `EPOCHS`, `BATCH_SIZE`, `SEEDS`, `RUNS_CSV`, `OUT_DIR`.

## Manual training (single run)

```bash
python -m src.train --device cuda --loss ce --use_aug 1 --seed 0 --stage train_ce_aug --skip_baselines
```

Baselines only:

```bash
python -m src.train --run_baselines_only --stage baselines --device cuda
```

## Grad-CAM

Fixed test indices: `configs/gradcam_manifest.json`

```bash
python -m src.gradcam_export --ckpt checkpoints/resnet18_ce_aug1_frac1.0_seed0.pt --build_manifest
python -m src.gradcam_export --ckpt checkpoints/resnet18_ce_aug1_frac1.0_seed0.pt

# Aug off vs on (ground truth | CAM | CAM)
python scripts/compare_gradcam_panel.py \
  --left_ckpt checkpoints/resnet18_ce_aug0_frac1.0_seed0.pt \
  --right_ckpt checkpoints/resnet18_ce_aug1_frac1.0_seed0.pt \
  --out results/gradcam_aug_off_vs_on.png
```

## Artifacts

| Path | Description |
|------|-------------|
| `results/runs.csv` | Full log (every val epoch + test); columns `stage`, `artifact_dir`, `ckpt_path` |
| `results/test_summary.csv` | **One row per test eval** — use this for slides / seed aggregates |
| `results/<stage>/seed_<N>/` | `test_roc.png`, `test_cm.png` for that run |
| `results/baselines/` | Baseline ROC/CM plots |
| `results/gradcam/` | Grad-CAM figures |
| `checkpoints/resnet18_{loss}_aug{0,1}_frac{…}_seed{N}.pt` | Best val-AUC weights |

Example layout after `reproduce.sh all`:

```
results/
  runs.csv
  test_summary.csv
  baselines/
    baseline_frozen_resnet_lr_test_roc.png
  train_ce_no_aug/seed_0/test_roc.png
  train_ce_aug/seed_0/test_roc.png
  train_focal_aug/seed_0/test_roc.png
  ablation_data_25/seed_0/test_roc.png
  gradcam/aug_off_vs_on.png
```

## Slides

`slides/final.md` (Marp) → export PDF for submission.

## Models

1. **ResNet18 + CE** — with optional augmentation (flip / rotate / jitter / RandomErasing).
2. **ResNet18 + focal loss (γ=2)** — same backbone, typically with augmentation on.
