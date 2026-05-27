from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms as T

from .data import LABELS, LABEL_TO_ID
from .models import build_resnet18
from .utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--center_crop", action="store_true", help="Resize then center-crop to reduce border shortcuts")
    return p.parse_args()


def build_eval_transform(image_size: int, center_crop: bool) -> T.Compose:
    if center_crop:
        scale = int(image_size * 1.14)
        return T.Compose(
            [
                T.Resize((scale, scale)),
                T.CenterCrop(image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _pil_for_overlay(img: Image.Image, image_size: int, center_crop: bool) -> np.ndarray:
    if center_crop:
        scale = int(image_size * 1.14)
        img = img.resize((scale, scale))
        w, h = img.size
        left = (w - image_size) // 2
        top = (h - image_size) // 2
        img = img.crop((left, top, left + image_size, top + image_size))
    else:
        img = img.resize((image_size, image_size))
    return np.array(img).astype(np.float32) / 255.0


def select_case_indices(y_true: np.ndarray, y_pred: np.ndarray) -> list[tuple[int, str]]:
    """Pick one TP, TN, FP, FN if available (student qualitative protocol)."""
    cases: list[tuple[int, str]] = []
    for name, mask_fn in [
        ("TP", lambda t, p: (t == 1) & (p == 1)),
        ("TN", lambda t, p: (t == 0) & (p == 0)),
        ("FP", lambda t, p: (t == 0) & (p == 1)),
        ("FN", lambda t, p: (t == 1) & (p == 0)),
    ]:
        idxs = np.where(mask_fn(y_true, y_pred))[0]
        if len(idxs):
            cases.append((int(idxs[0]), name))
    return cases


def main():
    args = parse_args()
    out = ensure_dir(args.out_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ds = load_dataset("hf-vision/chest-xray-pneumonia", revision="refs/convert/parquet")["test"]
    tfm = build_eval_transform(args.image_size, args.center_crop)

    model = build_resnet18(num_classes=2, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    pneumonia_target = [ClassifierOutputTarget(LABEL_TO_ID["PNEUMONIA"])]

    y_true_list, y_pred_list, probs_list = [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            row = ds[i]
            x = tfm(row["image"].convert("RGB")).unsqueeze(0).to(device)
            logits = model(x)
            pred = int(torch.argmax(logits, dim=1).item())
            y_true_list.append(int(row["label"]))
            y_pred_list.append(pred)
            probs_list.append(float(torch.softmax(logits, dim=1)[0, 1].item()))

    y_true = np.array(y_true_list, dtype=np.int64)
    y_pred = np.array(y_pred_list, dtype=np.int64)
    cases = select_case_indices(y_true, y_pred)

    suffix = "_centercrop" if args.center_crop else ""
    n = max(1, len(cases))
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (i, case_name) in zip(axes, cases):
        row = ds[i]
        img = row["image"].convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        true = int(row["label"])
        pred = y_pred[i]

        grayscale = cam(input_tensor=x, targets=pneumonia_target)[0]
        rgb = _pil_for_overlay(img, args.image_size, args.center_crop)
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"{case_name} true={LABELS[true]} pred={LABELS[pred]}")

    for ax in axes[len(cases) :]:
        ax.axis("off")

    fig.suptitle("Grad-CAM (target: PNEUMONIA logit)" + (" + center crop" if args.center_crop else ""), fontsize=12)
    fig.tight_layout()
    out_path = Path(out) / f"gradcam_cases{suffix}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
