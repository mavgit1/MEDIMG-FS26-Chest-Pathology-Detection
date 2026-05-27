from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from .data import LABELS, LABEL_TO_ID, build_transforms
from .models import build_resnet18
from .utils import ensure_dir

DEFAULT_MANIFEST = Path("configs/gradcam_manifest.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--per_category", type=int, default=3)
    p.add_argument("--build_manifest", action="store_true")
    return p.parse_args()


def predict_all(model, ds, tfm, device: str) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_pred = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            row = ds[i]
            x = tfm(row["image"].convert("RGB")).unsqueeze(0).to(device)
            logits = model(x)
            y_true.append(int(row["label"]))
            y_pred.append(int(torch.argmax(logits, dim=1).item()))
    return np.array(y_true, dtype=np.int64), np.array(y_pred, dtype=np.int64)


def select_case_indices(y_true: np.ndarray, y_pred: np.ndarray, per_category: int) -> list[dict]:
    cases: list[dict] = []
    for name, mask_fn in [
        ("TP", lambda t, p: (t == 1) & (p == 1)),
        ("TN", lambda t, p: (t == 0) & (p == 0)),
        ("FP", lambda t, p: (t == 0) & (p == 1)),
        ("FN", lambda t, p: (t == 1) & (p == 0)),
    ]:
        idxs = np.sort(np.where(mask_fn(y_true, y_pred))[0])
        for idx in idxs[:per_category]:
            cases.append({"index": int(idx), "case_type": name, "true_label": int(y_true[idx])})
    return cases


def load_model(ckpt: str, device: str):
    model = build_resnet18(num_classes=2, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def main():
    args = parse_args()
    out = ensure_dir(args.out_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    manifest_path = Path(args.manifest)
    ds = load_dataset("hf-vision/chest-xray-pneumonia", revision="refs/convert/parquet")["test"]
    tfm = build_transforms(args.image_size, train=False, use_aug=False)
    model = load_model(args.ckpt, device)

    if args.build_manifest:
        y_true, y_pred = predict_all(model, ds, tfm, device)
        cases = select_case_indices(y_true, y_pred, per_category=args.per_category)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps({"ckpt_used_to_build": args.ckpt, "per_category": args.per_category, "cases": cases}, indent=2)
        )
        print(f"wrote manifest ({len(cases)} cases) -> {manifest_path}")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}; run with --build_manifest first")

    cases = json.loads(manifest_path.read_text())["cases"]
    y_true, y_pred = predict_all(model, ds, tfm, device)

    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    pneumonia_target = [ClassifierOutputTarget(LABEL_TO_ID["PNEUMONIA"])]

    n = len(cases)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, case in zip(axes, cases):
        i = case["index"]
        row = ds[i]
        img = row["image"].convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        pred = int(y_pred[i])
        true = int(row["label"])

        grayscale = cam(input_tensor=x, targets=pneumonia_target)[0]
        rgb = np.array(img.resize((args.image_size, args.image_size))).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(
            f"#{i} {case['case_type']} true={LABELS[true]} pred={LABELS[pred]}",
            fontsize=8,
        )

    for ax in axes[len(cases) :]:
        ax.axis("off")

    fig.suptitle("Grad-CAM (fixed test indices, target: PNEUMONIA)", fontsize=12)
    fig.tight_layout()
    out_path = Path(out) / "gradcam_fixed.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
