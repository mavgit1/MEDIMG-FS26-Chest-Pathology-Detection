# Chest X-Ray Pneumonia Screening (MEDIMG FS26)

Minimal, reproducible chest X-ray **NORMAL vs PNEUMONIA** classifier using a small public Hugging Face dataset and a library-first pipeline.

## Dataset

- HF dataset: `hf-vision/chest-xray-pneumonia` (loaded from parquet conversion branch)

## Setup

**Recommended (uv)** — faster on vast.ai / fresh machines; `requirements.txt` stays the source of truth:

```bash
# if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Alternative (pip + venv):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (CPU smoke test)

```bash
python -m src.train --run_baselines_only --device cpu
```

## Run (GPU training)

```bash
python -m src.train --device cuda --loss ce --use_aug 1 --seed 0
python -m src.train --device cuda --loss focal --use_aug 1 --seed 0
```

Artifacts:

- `results/runs.csv` (all metrics)
- `results/*_roc.png`, `results/*_cm.png`

## Grad-CAM (fixed images for comparison)

```bash
# Once: lock which test indices to visualize (uses this checkpoint's TP/TN/FP/FN)
python -m src.gradcam_export --ckpt checkpoints/resnet18_ce_aug1_frac1.0_seed0.pt --build_manifest --per_category 3

# Same indices every time — compare preprocessing
python -m src.gradcam_export --ckpt <ckpt> --compare

# Or single panel
python -m src.gradcam_export --ckpt <ckpt> --center_crop
```

Manifest: `configs/gradcam_manifest.json`

## Slides (Marp)

Slides live in `slides/final.md` and should be exported to PDF for submission.

## Reproducibility

- Fixed seeds `{0,1,2}`
- Ablations: augmentation on/off, CE vs focal loss, 25% vs 100% train