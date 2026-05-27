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
- Splits: train / valid / test provided
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

## Baselines (as discussed in class)

- **Distribution-matched random** classifier
- **Majority class** classifier
- (Optional) Frozen-feature logistic regression (if time)

---

## Model 1 (Standard)

**ResNet18 (ImageNet pretrained)** fine-tuned on train split  
Loss: cross-entropy

---

## Model 2 (Student modification)

Same ResNet18 backbone with:
- stronger augmentation
- **focal loss** (γ = 2) to emphasize harder examples

---

## Results (Test)

- Table filled from `results/runs.csv`
- Figures auto-generated in `results/`

---

## Robustness & Ablations

- Loss: CE vs focal
- Augmentation: on vs off
- Data efficiency: 25% vs 100% train

---

## Qualitative: Grad-CAM

Insert `results/gradcam_grid.png`

---

## Limitations & Ethics

- Dataset bias / limited collection context → unknown generalization
- Automation bias: model outputs must be interpreted cautiously
- False negatives are critical in clinical settings (risk framing)

---

## Conclusion & Next Steps

- Simple pipeline, reproducible repo, clear baselines
- Next: external validation; multi-label pathology datasets (e.g., CheXpert) as future work

