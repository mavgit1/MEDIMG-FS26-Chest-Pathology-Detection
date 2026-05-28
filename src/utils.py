from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # torch may not be installed for ultra-minimal local checks
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_tag(cfg) -> str:
    """Stable model id for logs, plots, and checkpoints."""
    tag = f"resnet18_{cfg.loss}_aug{int(cfg.use_aug)}"
    if float(cfg.train_fraction) != 1.0:
        frac_s = str(cfg.train_fraction).replace(".", "p")
        tag += f"_frac{frac_s}"
    return tag


def checkpoint_path(cfg, ckpt_root: str | Path = "checkpoints") -> Path:
    return Path(ckpt_root) / (
        f"resnet18_{cfg.loss}_aug{int(cfg.use_aug)}_frac{cfg.train_fraction}_seed{cfg.seed}.pt"
    )


def artifact_dir(base: str | Path, cfg, stage: str = "") -> Path:
    """
    Per-run output folder, e.g. results/train_ce_aug/seed_0/ or results/baselines/.
    """
    base = Path(base)
    if stage:
        sub = base / stage
    else:
        sub = base / model_tag(cfg)
    if stage != "baselines":
        sub = sub / f"seed_{cfg.seed}"
    return ensure_dir(sub)


def to_device(x: Any, device: str):
    import torch

    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=True)
    return x

