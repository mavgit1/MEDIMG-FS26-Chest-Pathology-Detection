from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .baselines import (
    distribution_matched_random_baseline,
    frozen_resnet18_logistic_baseline,
    majority_class_baseline,
)
from .config import TrainConfig
from .data import LABEL_TO_ID, load_data
from .eval import run_eval_and_log
from .models import FocalLoss, build_resnet18
from .utils import artifact_dir, checkpoint_path, ensure_dir, model_tag, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--runs_csv", type=str, default="results/runs.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--use_amp", type=int, default=1)
    p.add_argument("--use_aug", type=int, default=1)
    p.add_argument("--train_fraction", type=float, default=1.0)
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    p.add_argument("--run_baselines_only", action="store_true")
    p.add_argument(
        "--skip_baselines",
        action="store_true",
        help="Skip baseline eval (use after stage `baselines` in reproduce.sh)",
    )
    p.add_argument(
        "--stage",
        type=str,
        default="",
        help="Pipeline stage name (stored in runs.csv for reproducibility)",
    )
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        total += float(loss.detach().cpu().item()) * x.size(0)
        n += x.size(0)
        pbar.set_postfix(loss=total / max(1, n))
    return total / max(1, n)


@torch.no_grad()
def predict_probs(model, loader, device: str):
    model.eval()
    probs = []
    targets = []
    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1]
        probs.append(p.detach().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def main():
    args = parse_args()
    cfg = TrainConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        use_amp=bool(args.use_amp),
        use_aug=bool(args.use_aug),
        train_fraction=float(args.train_fraction),
        loss=args.loss,
        stage=args.stage.strip(),
    )

    set_seed(cfg.seed)
    results_root = ensure_dir(args.out_dir)
    summary_csv = results_root / "test_summary.csv"
    stage = cfg.stage or "manual"
    run_dir = artifact_dir(results_root, cfg, stage)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        cfg = replace(cfg, device=device, use_amp=False)

    bundle = load_data(
        dataset_id=cfg.dataset_id,
        dataset_name=cfg.dataset_name,
        dataset_revision=getattr(cfg, "dataset_revision", None),
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_aug=cfg.use_aug,
        train_fraction=cfg.train_fraction,
        seed=cfg.seed,
    )

    baseline_dir = artifact_dir(results_root, cfg, "baselines")

    if not args.skip_baselines:
        y_test = np.array(bundle.ds["test"]["label"], dtype=np.int64)
        rng = np.random.default_rng(cfg.seed)
        y_prob_rnd = rng.binomial(1, bundle.train_prevalence, size=len(y_test)).astype(np.float64)
        distribution_matched_random_baseline(
            y_true=y_test, train_prevalence=bundle.train_prevalence, seed=cfg.seed
        )
        run_eval_and_log(
            runs_csv=args.runs_csv,
            out_dir=baseline_dir,
            split="test",
            model_name="baseline_random_dist",
            cfg=replace(cfg, stage="baselines"),
            y_true=y_test,
            y_prob=y_prob_rnd,
            summary_csv=summary_csv,
        )
        majority_class_baseline(y_true=y_test, train_majority_label=bundle.train_majority_label)
        maj_prob = np.full(shape=(len(y_test),), fill_value=float(bundle.train_majority_label), dtype=np.float64)
        run_eval_and_log(
            runs_csv=args.runs_csv,
            out_dir=baseline_dir,
            split="test",
            model_name="baseline_majority",
            cfg=replace(cfg, stage="baselines"),
            y_true=y_test,
            y_prob=maj_prob,
            summary_csv=summary_csv,
        )
        _, frozen_prob = frozen_resnet18_logistic_baseline(
            bundle.train_loader, bundle.test_loader, device=device, seed=cfg.seed
        )
        run_eval_and_log(
            runs_csv=args.runs_csv,
            out_dir=baseline_dir,
            split="test",
            model_name="baseline_frozen_resnet_lr",
            cfg=replace(cfg, stage="baselines"),
            y_true=y_test,
            y_prob=frozen_prob,
            summary_csv=summary_csv,
        )

    if args.run_baselines_only:
        print("Baselines logged to", args.runs_csv, "| plots:", baseline_dir)
        return

    model = build_resnet18(num_classes=2, pretrained=True).to(device)

    if cfg.loss == "focal":
        criterion = FocalLoss(gamma=2.0)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    scaler = torch.cuda.amp.GradScaler() if (device == "cuda" and cfg.use_amp) else None

    tag = model_tag(cfg)
    best_auc = -1.0
    ckpt_path = checkpoint_path(cfg)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=bundle.train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )
        val_prob, val_y = predict_probs(model, bundle.val_loader, device=device)
        val_metrics = run_eval_and_log(
            runs_csv=args.runs_csv,
            out_dir=run_dir,
            split="valid",
            model_name=tag,
            cfg=cfg,
            y_true=val_y,
            y_prob=val_prob,
            save_plots=False,
        )
        dt = time.time() - t0
        print(
            f"epoch={epoch+1}/{cfg.epochs} loss={train_loss:.4f} val_auc={val_metrics['auc']:.4f} time_s={dt:.1f}"
        )
        if np.isfinite(val_metrics["auc"]) and val_metrics["auc"] > best_auc:
            best_auc = float(val_metrics["auc"])
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, ckpt_path)

    # Final test eval on best checkpoint
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_prob, test_y = predict_probs(model, bundle.test_loader, device=device)
    run_eval_and_log(
        runs_csv=args.runs_csv,
        out_dir=run_dir,
        split="test",
        model_name=tag,
        cfg=cfg,
        y_true=test_y,
        y_prob=test_prob,
        ckpt_path=ckpt_path,
        summary_csv=summary_csv,
    )
    print(f"Test metrics -> {summary_csv} | plots: {run_dir} | ckpt: {ckpt_path}")


if __name__ == "__main__":
    main()

