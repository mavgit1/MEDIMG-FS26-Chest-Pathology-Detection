#!/usr/bin/env python3
"""Grad-CAM comparison: ground truth | model A (CAM on prediction) | model B."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import LABELS, build_transforms
from src.models import build_resnet18


def load_model(ckpt: str, device: str):
    model = build_resnet18(num_classes=2, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def predict(model, x) -> tuple[int, float]:
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred])


def overlay_for(model, img, tfm, device: str, image_size: int, class_id: int) -> np.ndarray:
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    x = tfm(img).unsqueeze(0).to(device)
    grayscale = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_id)])[0]
    rgb = np.array(img.resize((image_size, image_size))).astype(np.float32) / 255.0
    return show_cam_on_image(rgb, grayscale, use_rgb=True)


def rescued_indices(left, right, ds, tfm, device: str, limit: int) -> list[dict]:
    picked: list[dict] = []
    with torch.no_grad():
        for i in range(len(ds)):
            img = ds[i]["image"].convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            true = int(ds[i]["label"])
            pl, _ = predict(left, x)
            pr, _ = predict(right, x)
            if true == 0 and pl == 1 and pr == 0:
                picked.append({"index": i, "case_type": "rescued"})
            if len(picked) >= limit:
                break
    return picked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left_ckpt", required=True)
    p.add_argument("--right_ckpt", required=True)
    p.add_argument("--left_label", default="No augmentation")
    p.add_argument("--right_label", default="Aug + RandomErasing")
    p.add_argument("--manifest", default="configs/gradcam_manifest.json")
    p.add_argument("--indices", default="", help="Comma-separated test indices")
    p.add_argument("--case_source", choices=["manifest", "rescued"], default="manifest")
    p.add_argument("--out", default="results/gradcam_compare_aug.png")
    p.add_argument("--device", default="cuda")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--max_rows", type=int, default=6)
    args = p.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ds = load_dataset("hf-vision/chest-xray-pneumonia", revision="refs/convert/parquet")["test"]
    tfm = build_transforms(args.image_size, train=False, use_aug=False)
    left = load_model(args.left_ckpt, device)
    right = load_model(args.right_ckpt, device)

    if args.indices.strip():
        cases = [{"index": int(x.strip()), "case_type": "custom"} for x in args.indices.split(",")]
    elif args.case_source == "rescued":
        cases = rescued_indices(left, right, ds, tfm, device, args.max_rows)
    else:
        cases = json.loads(Path(args.manifest).read_text())["cases"][: args.max_rows]

    n = len(cases)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 2.8 * n))
    if n == 1:
        axes = np.array([axes])

    for row, case in enumerate(cases):
        i = case["index"]
        img = ds[i]["image"].convert("RGB")
        true = int(ds[i]["label"])
        x = tfm(img).unsqueeze(0).to(device)
        pred_l, conf_l = predict(left, x)
        pred_r, conf_r = predict(right, x)

        rgb = np.array(img.resize((args.image_size, args.image_size))).astype(np.float32) / 255.0
        ov_l = overlay_for(left, img, tfm, device, args.image_size, pred_l)
        ov_r = overlay_for(right, img, tfm, device, args.image_size, pred_r)

        ax_truth, ax_l, ax_r = axes[row]
        ax_truth.imshow(rgb)
        ax_l.imshow(ov_l)
        ax_r.imshow(ov_r)
        for ax in (ax_truth, ax_l, ax_r):
            ax.axis("off")

        ok_l = "✓" if pred_l == true else "✗"
        ok_r = "✓" if pred_r == true else "✗"
        ax_truth.set_title(f"#{i}\nGround truth: {LABELS[true]}", fontsize=9, fontweight="bold")
        ax_l.set_title(
            f"{args.left_label}\nPrediction: {LABELS[pred_l]} {ok_l}  ({conf_l:.0%})\n"
            f"Grad-CAM → {LABELS[pred_l]}",
            fontsize=8,
        )
        ax_r.set_title(
            f"{args.right_label}\nPrediction: {LABELS[pred_r]} {ok_r}  ({conf_r:.0%})\n"
            f"Grad-CAM → {LABELS[pred_r]}",
            fontsize=8,
        )

    fig.suptitle(
        "Left: ground truth · Center/right: Grad-CAM for each model’s predicted class",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
