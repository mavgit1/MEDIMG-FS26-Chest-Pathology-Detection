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

**Grad-CAM:** fixed test indices · targets **PNEUMONIA** logit · TP/TN/FP/FN panel

---

## Results (Test)

- Table filled from `results/runs.csv`
- Figures auto-generated in `results/`

---

## Robustness & Ablations

- Loss: CE vs focal
- **Augmentation:** off vs on (on includes **RandomErasing**)
- Data efficiency: 25% vs 100% train (if run)

---

## Qualitative: Grad-CAM

![gradcam](results/gradcam_fixed.png)

Fixed test indices (see `configs/gradcam_manifest.json`) · PNEUMONIA logit

---

## Limitations & Ethics

- Attention is **not always on pathology** — edges/markers can act as shortcuts
- Maps are **partially anatomical, partially artifact-driven** → needs more analysis
- Not deployment-ready without external validation and stronger explainability
- Automation bias; false negatives are clinically critical

---

## Conclusion & Next Steps

- Simple pipeline, reproducible repo, clear baselines
- Next: external validation; multi-label pathology datasets (e.g., CheXpert) as future work

