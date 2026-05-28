#!/usr/bin/env bash
# Reproducible pipeline stages for MEDIMG FS26 chest X-ray project.
#
# Usage:
#   bash scripts/reproduce.sh list
#   bash scripts/reproduce.sh clean
#   bash scripts/reproduce.sh baselines
#   bash scripts/reproduce.sh train_ce_no_aug
#   bash scripts/reproduce.sh train_ce_aug
#   bash scripts/reproduce.sh train_focal_aug
#   bash scripts/reproduce.sh ablation_data_25
#   bash scripts/reproduce.sh gradcam_manifest
#   bash scripts/reproduce.sh gradcam_compare
#   bash scripts/reproduce.sh robustness
#   bash scripts/reproduce.sh all
#
# Optional: configs/reproduce.env (see configs/reproduce.env.example)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f configs/reproduce.env ]]; then
  # shellcheck disable=SC1091
  source configs/reproduce.env
fi

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEEDS="${SEEDS:-0}"
RUNS_CSV="${RUNS_CSV:-results/runs.csv}"
OUT_DIR="${OUT_DIR:-results}"

activate_venv() {
  if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
  else
    echo "No .venv — run: bash scripts/setup.sh" >&2
    exit 1
  fi
}

train_stage() {
  local stage="$1"
  shift
  activate_venv
  python -m src.train \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --runs_csv "$RUNS_CSV" \
    --out_dir "$OUT_DIR" \
    --stage "$stage" \
    --skip_baselines \
    "$@"
}

stage_clean() {
  echo "Removing generated artifacts in results/ and checkpoints/ ..."
  rm -rf "${OUT_DIR:?}"/* "${OUT_DIR:?}"/.[!.]* 2>/dev/null || true
  rm -rf checkpoints/* checkpoints/.[!.]* 2>/dev/null || true
  mkdir -p "$OUT_DIR" checkpoints
  touch "$OUT_DIR/.gitkeep" checkpoints/.gitkeep
  echo "Clean."
}

stage_baselines() {
  activate_venv
  python -m src.train \
    --device "$DEVICE" \
    --runs_csv "$RUNS_CSV" \
    --out_dir "$OUT_DIR" \
    --stage baselines \
    --run_baselines_only
}

stage_train_ce_no_aug() {
  for seed in $SEEDS; do
    train_stage train_ce_no_aug --loss ce --use_aug 0 --seed "$seed"
  done
}

stage_train_ce_aug() {
  for seed in $SEEDS; do
    train_stage train_ce_aug --loss ce --use_aug 1 --seed "$seed"
  done
}

stage_train_focal_aug() {
  for seed in $SEEDS; do
    train_stage train_focal_aug --loss focal --use_aug 1 --seed "$seed"
  done
}

stage_ablation_data_25() {
  for seed in $SEEDS; do
    train_stage ablation_data_25 --loss ce --use_aug 1 --train_fraction 0.25 --seed "$seed"
  done
}

ckpt_ce_no_aug() {
  echo "checkpoints/resnet18_ce_aug0_frac1.0_seed${1:-0}.pt"
}

ckpt_ce_aug() {
  echo "checkpoints/resnet18_ce_aug1_frac1.0_seed${1:-0}.pt"
}

stage_gradcam_manifest() {
  activate_venv
  local ckpt
  ckpt="$(ckpt_ce_aug 0)"
  python -m src.gradcam_export \
    --device "$DEVICE" \
    --ckpt "$ckpt" \
    --out_dir "$OUT_DIR" \
    --build_manifest \
    --per_category 3
}

stage_gradcam_single() {
  activate_venv
  local ckpt="${1:-$(ckpt_ce_aug 0)}"
  python -m src.gradcam_export \
    --device "$DEVICE" \
    --ckpt "$ckpt" \
    --out_dir "$OUT_DIR"
}

stage_gradcam_compare() {
  activate_venv
  local seed="${1:-0}"
  python scripts/compare_gradcam_panel.py \
    --device "$DEVICE" \
    --left_ckpt "$(ckpt_ce_no_aug "$seed")" \
    --right_ckpt "$(ckpt_ce_aug "$seed")" \
    --left_label "No augmentation" \
    --right_label "Aug + RandomErasing" \
    --case_source manifest \
    --max_rows 5 \
    --out "${OUT_DIR}/gradcam/aug_off_vs_on.png"
}

stage_gradcam_compare_full() {
  activate_venv
  local seed="${1:-0}"
  python scripts/compare_gradcam_panel.py \
    --device "$DEVICE" \
    --left_ckpt "$(ckpt_ce_no_aug "$seed")" \
    --right_ckpt "$(ckpt_ce_aug "$seed")" \
    --left_label "No augmentation" \
    --right_label "Aug + RandomErasing" \
    --case_source manifest \
    --max_rows 9 \
    --out "${OUT_DIR}/gradcam/aug_off_vs_on_full.png"
}

stage_robustness() {
  local saved_seeds="$SEEDS"
  SEEDS="0 1 2"
  stage_train_ce_aug
  stage_train_focal_aug
  SEEDS="$saved_seeds"
}

stage_all() {
  stage_clean
  stage_baselines
  stage_train_ce_no_aug
  stage_train_ce_aug
  stage_train_focal_aug
  stage_ablation_data_25
  stage_gradcam_manifest
  stage_gradcam_compare
  stage_gradcam_compare_full
  echo "Pipeline complete. Metrics: ${RUNS_CSV}"
}

stage_list() {
  cat <<'EOF'
Stages (run in order for a full fresh experiment):

  clean              Remove results/* and checkpoints/*
  baselines          Random, majority, frozen ResNet18 + LR (test only)
  train_ce_no_aug    Model 1 baseline: CE, no augmentation
  train_ce_aug       Model 1: CE + augmentation (+ RandomErasing)
  train_focal_aug    Model 2: focal loss + augmentation
  ablation_data_25   CE + aug, 25% training data
  gradcam_manifest   Build configs/gradcam_manifest.json (once per run)
  gradcam_single     Single-model panel (optional ckpt path arg)
  gradcam_compare    Aug off vs on (truth | CAM | CAM)
  gradcam_compare_full  Same, using full manifest (9 rows)
  robustness         train_ce_aug + train_focal_aug for SEEDS=0 1 2
  all                clean + full pipeline (SEEDS default 0 only)

Environment (optional configs/reproduce.env):
  DEVICE  EPOCHS  BATCH_SIZE  SEEDS  RUNS_CSV  OUT_DIR
EOF
}

STAGE="${1:-list}"
shift || true

case "$STAGE" in
  list) stage_list ;;
  clean) stage_clean ;;
  baselines) stage_baselines ;;
  train_ce_no_aug) stage_train_ce_no_aug ;;
  train_ce_aug) stage_train_ce_aug ;;
  train_focal_aug) stage_train_focal_aug ;;
  ablation_data_25) stage_ablation_data_25 ;;
  gradcam_manifest) stage_gradcam_manifest ;;
  gradcam_single) stage_gradcam_single "$@" ;;
  gradcam_compare) stage_gradcam_compare "$@" ;;
  gradcam_compare_full) stage_gradcam_compare_full "$@" ;;
  robustness) stage_robustness ;;
  all) stage_all ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    stage_list
    exit 1
    ;;
esac
