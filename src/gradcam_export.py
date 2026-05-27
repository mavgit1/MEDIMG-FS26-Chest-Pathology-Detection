from __future__ import annotations

import argparse
import json
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

DEFAULT_MANIFEST = Path("configs/gradcam_manifest.json")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--per_category", type=int, default=3, help="Examples per TP/TN/FP/FN in manifest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--build_manifest",
        action="store_true",
        help="Write fixed test indices (from this ckpt) then exit",
    )
    p.add_argument(
        "--center_crop",
        action="store_true",
        help="Use center-crop preprocessing (single-panel mode)",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Side-by-side full-frame vs center-crop on the same fixed indices",
    )
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


def select_case_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    per_category: int,
) -> list[dict]:
    """Deterministic: lowest test index first in each category."""
    cases: list[dict] = []
    for name, mask_fn in [
        ("TP", lambda t, p: (t == 1) & (p == 1)),
        ("TN", lambda t, p: (t == 0) & (p == 0)),
        ("FP", lambda t, p: (t == 0) & (p == 1)),
        ("FN", lambda t, p: (t == 1) & (p == 0)),
    ]:
        idxs = np.sort(np.where(mask_fn(y_true, y_pred))[0])
        for idx in idxs[:per_category]:
            cases.append(
                {
                    "index": int(idx),
                    "case_type": name,
                    "true_label": int(y_true[idx]),
                    "ref_pred": int(y_pred[idx]),
                }
            )
    return cases


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return data["cases"]


def save_manifest(path: Path, cases: list[dict], ckpt: str, per_category: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ckpt_used_to_build": ckpt,
                "per_category": per_category,
                "cases": cases,
            },
            indent=2,
        )
    )


def load_model(ckpt: str, device: str):
    model = build_resnet18(num_classes=2, pretrained=False).to(device)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def render_cam(
    model,
    cam: GradCAM,
    ds,
    case: dict,
    image_size: int,
    center_crop: bool,
    device: str,
    y_pred_current: int,
) -> np.ndarray:
    tfm = build_eval_transform(image_size, center_crop)
    pneumonia_target = [ClassifierOutputTarget(LABEL_TO_ID["PNEUMONIA"])]

    i = case["index"]
    row = ds[i]
    img = row["image"].convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    grayscale = cam(input_tensor=x, targets=pneumonia_target)[0]
    rgb = _pil_for_overlay(img, image_size, center_crop)
    overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

    true = int(row["label"])
    title = (
        f"#{i} {case['case_type']} "
        f"true={LABELS[true]} pred={LABELS[y_pred_current]}"
    )
    return overlay, title


def main():
    args = parse_args()
    out = ensure_dir(args.out_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    manifest_path = Path(args.manifest)
    ds = load_dataset("hf-vision/chest-xray-pneumonia", revision="refs/convert/parquet")["test"]
    tfm_eval = build_eval_transform(args.image_size, center_crop=False)
    model = load_model(args.ckpt, device)

    if args.build_manifest:
        y_true, y_pred = predict_all(model, ds, tfm_eval, device)
        cases = select_case_indices(y_true, y_pred, per_category=args.per_category)
        save_manifest(manifest_path, cases, args.ckpt, args.per_category)
        print(f"wrote manifest ({len(cases)} cases) -> {manifest_path}")
        return

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run with --build_manifest once, e.g.\n"
            f"  python -m src.gradcam_export --ckpt <ckpt> --build_manifest --per_category {args.per_category}"
        )

    cases = load_manifest(manifest_path)
    # Current predictions (for titles only; indices stay fixed)
    y_true, y_pred = predict_all(model, ds, tfm_eval, device)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    if args.compare:
        n = len(cases)
        fig, axes = plt.subplots(n, 2, figsize=(9, 2.8 * n))
        if n == 1:
            axes = np.array([axes])
        for row_ax, case in zip(axes, cases):
            idx = case["index"]
            pred_now = int(y_pred[idx])
            for col, cc in enumerate([False, True]):
                overlay, title = render_cam(
                    model, cam, ds, case, args.image_size, cc, device, pred_now
                )
                row_ax[col].imshow(overlay)
                row_ax[col].axis("off")
                col_title = "Center crop" if cc else "Full frame"
                row_ax[col].set_title(f"{col_title}\n{title}", fontsize=9)
        fig.suptitle(
            "Grad-CAM comparison (fixed test indices, target: PNEUMONIA)",
            fontsize=12,
        )
        fig.tight_layout()
        out_path = Path(out) / "gradcam_compare_fixed.png"
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"wrote {out_path}")
        return

    center_crop = args.center_crop
    suffix = "_centercrop" if center_crop else "_fullframe"
    n = len(cases)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, case in zip(axes, cases):
        idx = case["index"]
        pred_now = int(y_pred[idx])
        overlay, title = render_cam(
            model, cam, ds, case, args.image_size, center_crop, device, pred_now
        )
        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(title, fontsize=8)

    for ax in axes[len(cases) :]:
        ax.axis("off")

    mode = "center crop" if center_crop else "full frame"
    fig.suptitle(f"Grad-CAM fixed indices ({mode}, target: PNEUMONIA)", fontsize=12)
    fig.tight_layout()
    out_path = Path(out) / f"gradcam_fixed{suffix}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
