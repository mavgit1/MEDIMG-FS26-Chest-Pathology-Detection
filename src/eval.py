from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .utils import ensure_dir


def _collect_probs_and_targets(model: torch.nn.Module, loader, device: str):
    model.eval()
    probs = []
    targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            p = torch.softmax(logits, dim=1)[:, 1]  # P(PNEUMONIA)
            probs.append(p.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
    return np.concatenate(probs), np.concatenate(targets)


def evaluate_classifier_probs(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(np.int64)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "auc": auc,
        "acc": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1]),
    }


def plot_file_stem(model_name: str, split: str) -> str:
    """Baselines: named files in shared folder; trained models: split-only inside per-run dir."""
    if model_name.startswith("baseline_"):
        return f"{model_name}_{split}"
    return split


def save_eval_artifacts(
    out_dir: str | Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    file_stem: str,
):
    out = ensure_dir(out_dir)
    stem = file_stem.lower().replace(" ", "_")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{file_stem} ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out / f"{stem}_roc.png", dpi=180)
    plt.close()

    y_pred = (y_prob >= 0.5).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["NORMAL", "PNEUMONIA"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title(f"{file_stem} Confusion (thr=0.5)")
    plt.tight_layout()
    plt.savefig(out / f"{stem}_cm.png", dpi=180)
    plt.close(fig)


def append_run_row(runs_csv: str | Path, row: dict):
    runs_csv = Path(runs_csv)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if runs_csv.exists():
        df0 = pd.read_csv(runs_csv)
        df = pd.concat([df0, df], ignore_index=True)
    df.to_csv(runs_csv, index=False)


def append_test_summary(summary_csv: str | Path, row: dict):
    """One row per (stage, seed, model_name) on test — easy slide tables."""
    summary_csv = Path(summary_csv)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    key = ("stage", "seed", "model_name")
    df = pd.DataFrame([row])
    if summary_csv.exists():
        old = pd.read_csv(summary_csv)
        mask = np.ones(len(old), dtype=bool)
        for k in key:
            if k in old.columns and k in row:
                mask &= old[k].astype(str) == str(row[k])
        old = old[~mask]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(summary_csv, index=False)


def run_eval_and_log(
    *,
    runs_csv: str | Path,
    out_dir: str | Path,
    split: str,
    model_name: str,
    cfg,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ckpt_path: str | Path | None = None,
    summary_csv: str | Path | None = None,
    save_plots: bool = True,
):
    metrics = evaluate_classifier_probs(y_true=y_true, y_prob=y_prob, threshold=0.5)
    stem = plot_file_stem(model_name, split)
    if save_plots:
        save_eval_artifacts(out_dir=out_dir, y_true=y_true, y_prob=y_prob, file_stem=stem)

    cfg_dict = asdict(cfg) if is_dataclass(cfg) else {}
    if not cfg_dict.get("stage"):
        cfg_dict.pop("stage", None)

    artifact_rel = str(Path(out_dir))
    row = {
        **cfg_dict,
        "split": split,
        "model_name": model_name,
        "artifact_dir": artifact_rel,
        "ckpt_path": str(ckpt_path) if ckpt_path else "",
        "auc": metrics["auc"],
        "acc": metrics["acc"],
        "f1": metrics["f1"],
        "cm00": int(metrics["cm"][0, 0]),
        "cm01": int(metrics["cm"][0, 1]),
        "cm10": int(metrics["cm"][1, 0]),
        "cm11": int(metrics["cm"][1, 1]),
    }
    append_run_row(runs_csv=runs_csv, row=row)

    if split == "test" and summary_csv is not None:
        append_test_summary(
            summary_csv,
            {
                "stage": cfg_dict.get("stage", ""),
                "seed": cfg_dict.get("seed", ""),
                "model_name": model_name,
                "loss": cfg_dict.get("loss", ""),
                "use_aug": cfg_dict.get("use_aug", ""),
                "train_fraction": cfg_dict.get("train_fraction", ""),
                "auc": metrics["auc"],
                "acc": metrics["acc"],
                "f1": metrics["f1"],
                "artifact_dir": artifact_rel,
                "ckpt_path": str(ckpt_path) if ckpt_path else "",
                "roc_png": str(Path(out_dir) / f"{stem.lower().replace(' ', '_')}_roc.png"),
                "cm_png": str(Path(out_dir) / f"{stem.lower().replace(' ', '_')}_cm.png"),
            },
        )

    return metrics
