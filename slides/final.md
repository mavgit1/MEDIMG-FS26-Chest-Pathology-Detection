---
marp: true
title: Chest X-Ray Pneumonia Screening
paginate: true
---

## Chest X-Ray Pneumonia Screening

**MEDIMG FS26** — Decision Support in Medical Imaging  
Marvin Huber · Alex Libov · Joel Greiner

---

## Motivation (Decision Support)

- **Triage**: flag likely pneumonia cases for faster review
- **Goal**: support workflow prioritization, not autonomous diagnosis

---

## Dataset

- Hugging Face: `hf-vision/chest-xray-pneumonia` (parquet export)
- **Binary labels**: NORMAL vs PNEUMONIA
- **HF splits (predefined):** train ~5.2k · official validation ~16 · test ~624
- **Our validation:** 10% stratified hold-out from **train** (HF validation too small)
- **Test:** untouched HF test set — used only for final numbers
- License: CC BY 4.0

---

## Task & Evaluation

- **Binary classification** on chest X-rays
- Metrics:
  - **ROC-AUC** (threshold-free)
  - Accuracy + **F1**
  - Confusion matrix at threshold 0.5
- Robustness: 3 seeds + 3 ablations

---

## Baselines

1. **Random** — labels drawn with train-set pneumonia rate (per course)
2. **Majority** — always predict most common **train** class
3. **Frozen ResNet18 + logistic regression** — ImageNet features, no CNN training

---

## Model 1 (Standard)

**ResNet18 (ImageNet pretrained)** fine-tuned on train split  
Loss: cross-entropy

---

## Model 2 (Student modification)

Same ResNet18 + **focal loss** (γ = 2) for class imbalance

**Augmentation (when enabled):** flip / rotation / jitter + **RandomErasing** to discourage border shortcuts (real training change, not viz-only)

**Grad-CAM:** fixed test indices · CAM targets **each model’s predicted class**

Reproduce: `bash scripts/reproduce.sh gradcam_compare`

---

## Results (Test)

- Table from `results/runs.csv` (filter `split=test`)
- Pipeline: `bash scripts/reproduce.sh all`

---

## Robustness & Ablations

| Stage | Command |
|-------|---------|
| No aug | `train_ce_no_aug` |
| Aug + RandomErasing | `train_ce_aug` |
| Focal loss | `train_focal_aug` |
| 25% data | `ablation_data_25` |
| 3 seeds | `SEEDS="0 1 2" bash scripts/reproduce.sh robustness` |

---

## Qualitative: Grad-CAM

![gradcam](results/gradcam_aug_off_vs_on.png)

Ground truth · aug-off CAM (pred class) · aug-on CAM (pred class)  
Same indices (`configs/gradcam_manifest.json`)

---

## Limitations & Ethics

- Aug improved **test metrics** more than obvious **visual** CAM differences
- Attention can follow edges/markers, not only pathology
- Not deployment-ready without external validation and stronger explainability
- Automation bias; false negatives are clinically critical

---

## Conclusion & Next Steps

- Simple pipeline, reproducible repo, clear baselines
- Next: external validation; multi-label pathology datasets (e.g., CheXpert) as future work

