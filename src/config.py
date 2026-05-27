from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    dataset_id: str = "hf-vision/chest-xray-pneumonia"
    dataset_name: str | None = None
    dataset_revision: str = "refs/convert/parquet"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    use_amp: bool = True

    # Ablations
    use_aug: bool = True
    train_fraction: float = 1.0
    loss: str = "ce"  # "ce" | "focal"

    # Reproduce pipeline label (logged in results/runs.csv)
    stage: str = ""

