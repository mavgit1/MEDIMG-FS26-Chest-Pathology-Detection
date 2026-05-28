#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", default="results/test_summary.csv")
    ap.add_argument("--out_dir", default="results/summary")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    out = _ensure_dir(args.out_dir)

    # Keep core stages for plots
    dfm = df[df["stage"].isin(["train_ce_no_aug", "train_ce_aug", "train_focal_aug", "ablation_data_25"])].copy()

    # Derived rates from confusion matrix cells (inferred from runs.csv is better, but summary has no cm cells)
    # We'll compute FP/FN only for seed 0 using runs.csv if present; otherwise skip those plots.

    # 1) AUC by stage across seeds
    order = ["train_ce_no_aug", "train_ce_aug", "train_focal_aug", "ablation_data_25"]
    labels = {
        "train_ce_no_aug": "CE (no aug)",
        "train_ce_aug": "CE (aug)",
        "train_focal_aug": "Focal (aug)",
        "ablation_data_25": "CE (aug, 25% data)",
    }
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for stage in order:
        d = dfm[dfm["stage"] == stage]
        if len(d) == 0:
            continue
        xs = np.arange(len(d))
        ax.scatter([stage] * len(d), d["auc"], alpha=0.9)
        ax.plot([stage] * len(d), d["auc"], alpha=0.2)
        ax.scatter(stage, d["auc"].mean(), marker="D", s=70)
    ax.set_xticks(order, [labels.get(s, s) for s in order], rotation=15, ha="right")
    ax.set_ylabel("Test AUC")
    ax.set_title("Test AUC by stage (each dot = seed)")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "auc_by_stage.png", dpi=180)
    plt.close(fig)

    # 2) Accuracy vs AUC (shows threshold vs ranking tension)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for stage in order:
        d = dfm[dfm["stage"] == stage]
        if len(d) == 0:
            continue
        ax.scatter(d["auc"], d["acc"], label=labels.get(stage, stage))
        for _, r in d.iterrows():
            ax.annotate(f"s{int(r['seed'])}", (r["auc"], r["acc"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("Test AUC")
    ax.set_ylabel("Test Accuracy (thr=0.5)")
    ax.set_title("Ranking vs threshold metric")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "auc_vs_acc.png", dpi=180)
    plt.close(fig)

    print(f"wrote {out/'auc_by_stage.png'}")
    print(f"wrote {out/'auc_vs_acc.png'}")


if __name__ == "__main__":
    main()

