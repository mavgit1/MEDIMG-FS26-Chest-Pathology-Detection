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
from torchvision import transforms as T

from .data import LABELS
from .models import build_resnet18
from .utils import ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    out = ensure_dir(args.out_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ds = load_dataset("hf-vision/chest-xray-pneumonia", revision="refs/convert/parquet")["test"]

    tfm = T.Compose(
        [
            T.Resize((args.image_size, args.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model = build_resnet18(num_classes=2, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    # resnet final conv block
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    idxs = np.linspace(0, len(ds) - 1, num=min(args.n, len(ds)), dtype=int)
    fig, axes = plt.subplots(2, (len(idxs) + 1) // 2, figsize=(4 * ((len(idxs) + 1) // 2), 8))
    axes = np.array(axes).reshape(-1)

    for ax, i in zip(axes, idxs):
        row = ds[int(i)]
        img: Image.Image = row["image"].convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        true = int(row["label"])

        grayscale = cam(input_tensor=x)[0]
        rgb = np.array(img.resize((args.image_size, args.image_size))).astype(np.float32) / 255.0
        overlay = show_cam_on_image(rgb, grayscale, use_rgb=True)

        ax.imshow(overlay)
        ax.axis("off")
        ax.set_title(f"true={LABELS[true]} pred={LABELS[pred]}")

    for ax in axes[len(idxs) :]:
        ax.axis("off")

    fig.tight_layout()
    out_path = Path(out) / "gradcam_grid.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

