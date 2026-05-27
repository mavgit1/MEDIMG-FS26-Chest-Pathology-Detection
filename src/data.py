from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


LABELS = ["NORMAL", "PNEUMONIA"]
LABEL_TO_ID = {k: i for i, k in enumerate(LABELS)}


def build_transforms(image_size: int, train: bool, use_aug: bool) -> Callable[[Image.Image], torch.Tensor]:
    base = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train and use_aug:
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]
        return T.Compose(aug + base)
    return T.Compose(base)


class HFDataset(Dataset):
    def __init__(self, hf_split, tfm: Callable[[Image.Image], torch.Tensor], train_fraction: float = 1.0, seed: int = 0):
        self.hf_split = hf_split
        self.tfm = tfm
        self.indices = np.arange(len(hf_split))
        if train_fraction < 1.0:
            rng = np.random.default_rng(seed)
            n = max(1, int(round(len(self.indices) * train_fraction)))
            self.indices = rng.choice(self.indices, size=n, replace=False)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        row = self.hf_split[int(self.indices[idx])]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        x = self.tfm(img.convert("RGB"))
        y = int(row["label"])
        return x, y


@dataclass(frozen=True)
class DataBundle:
    ds: DatasetDict
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_prevalence: float
    train_majority_label: int
    val_source: str  # how validation was built (for README / slides)


def load_data(
    dataset_id: str,
    dataset_name: str | None,
    dataset_revision: str | None,
    image_size: int,
    batch_size: int,
    num_workers: int,
    use_aug: bool,
    train_fraction: float,
    seed: int,
) -> DataBundle:
    ds = load_dataset(dataset_id, name=dataset_name, revision=dataset_revision) if dataset_revision else load_dataset(dataset_id, name=dataset_name)

    # HF provides train/validation/test, but validation can be tiny (e.g. 16 images).
    # Use HF validation only when it has enough samples; otherwise hold out 10% of train (stratified).
    if "validation" in ds and len(ds["validation"]) >= 100:
        val_source = "hf_validation"
        train_split = ds["train"]
        valid_split = ds["validation"]
    elif "valid" in ds and len(ds["valid"]) >= 100:
        val_source = "hf_valid"
        train_split = ds["train"]
        valid_split = ds["valid"]
    else:
        split = ds["train"].train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")
        train_split = split["train"]
        valid_split = split["test"]
        val_source = "train_90_10_stratified"
    tfm_train = build_transforms(image_size=image_size, train=True, use_aug=use_aug)
    tfm_eval = build_transforms(image_size=image_size, train=False, use_aug=False)

    train = HFDataset(train_split, tfm_train, train_fraction=train_fraction, seed=seed)
    val = HFDataset(valid_split, tfm_eval, train_fraction=1.0, seed=seed)
    test = HFDataset(ds["test"], tfm_eval, train_fraction=1.0, seed=seed)

    # prevalence computed from FULL train split (not fractional), to define distribution-matched random baseline
    train_labels = np.array(train_split["label"], dtype=np.int64)
    train_prevalence = float((train_labels == LABEL_TO_ID["PNEUMONIA"]).mean())
    counts = np.bincount(train_labels, minlength=2)
    train_majority_label = int(np.argmax(counts))

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return DataBundle(
        ds=ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_prevalence=train_prevalence,
        train_majority_label=train_majority_label,
        val_source=val_source,
    )

