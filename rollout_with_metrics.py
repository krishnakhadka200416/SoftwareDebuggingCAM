"""Run Attention Rollout on test_folder × {ViT-B/16, ViT-B/32}, compute all
Table 1 metrics (ADCC, AD, Coh, Com, Inc, ADD, ROAD, ROAD_Del, ROAD_Ins, Time).

Output: results_rollout/per_image.csv and summary.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    vit_b_16, ViT_B_16_Weights,
    vit_b_32, ViT_B_32_Weights,
)

from attention_rollout_vit import run_attention_rollout
from pytorch_grad_cam.metrics.ADCC import ADCC
from pytorch_grad_cam.metrics.road import (
    ROADCombined, ROADMostRelevantFirstAverage, ROADLeastRelevantFirstAverage,
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputReST, ClassifierOutputSoftmaxTarget, ClassifierOutputTarget,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path: str, device: str, size: int = 224) -> Tuple[torch.Tensor, np.ndarray]:
    img = Image.open(path).convert("RGB")
    pre = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    x = pre(img).to(device)
    vis = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])(Image.open(path).convert("RGB"))
    rgb = vis.numpy().transpose(1, 2, 0)
    return x, np.clip(rgb, 0.0, 1.0)


def normalize_heatmap(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32)
    lo, hi = float(h.min()), float(h.max())
    if hi - lo < 1e-8:
        return np.zeros_like(h)
    return (h - lo) / (hi - lo)


def evaluate_metrics(model, x: torch.Tensor, heatmap: np.ndarray, target_class: int) -> Dict:
    out = {k: float("nan") for k in
           ["ADCC", "AD", "Coh", "Com", "Inc", "ADD",
            "ROAD", "ROAD_Del", "ROAD_Ins"]}
    pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    try:
        cam_metric = ADCC()
        targets = [ClassifierOutputReST(target_class)]
        metric_targets = [ClassifierOutputSoftmaxTarget(target_class)]
        adcc_v, drop_v, coh_v, com_v, inc_v, dropindel_v = cam_metric(
            x.unsqueeze(0), heatmap[np.newaxis, ...], targets, metric_targets, model,
            lambda xi, t, _g=heatmap: torch.from_numpy(_g).unsqueeze(0),
        )
        out["ADCC"] = float(adcc_v[0]); out["AD"] = float(drop_v[0])
        out["Coh"]  = float(coh_v[0]);  out["Com"] = float(com_v[0])
        out["Inc"]  = float(inc_v[0]);  out["ADD"] = float(dropindel_v[0])
    except Exception as e:
        print(f"  [WARN] ADCC failed: {e}")

    targets = [ClassifierOutputTarget(target_class)]
    try:
        out["ROAD"] = float(np.array(
            ROADCombined(percentiles=pcts)(x.unsqueeze(0), heatmap[np.newaxis, ...], targets, model)
        ).reshape(-1)[0])
    except Exception as e:
        print(f"  [WARN] ROAD failed: {e}")
    try:
        out["ROAD_Del"] = float(np.array(
            ROADMostRelevantFirstAverage(percentiles=pcts)(x.unsqueeze(0), heatmap[np.newaxis, ...], targets, model)
        ).reshape(-1)[0])
    except Exception as e:
        print(f"  [WARN] ROAD_Del failed: {e}")
    try:
        out["ROAD_Ins"] = float(np.array(
            ROADLeastRelevantFirstAverage(percentiles=pcts)(x.unsqueeze(0), heatmap[np.newaxis, ...], targets, model)
        ).reshape(-1)[0])
    except Exception as e:
        print(f"  [WARN] ROAD_Ins failed: {e}")

    return out


def load_model(name: str, device: str):
    if name == "vit_b_16":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
    elif name == "vit_b_32":
        m = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1).to(device).eval()
    else:
        raise ValueError(name)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="test_folder")
    parser.add_argument("--max_images", type=int, default=10)
    parser.add_argument("--models", default="vit_b_16,vit_b_32")
    parser.add_argument("--out_dir", default="results_rollout")
    parser.add_argument("--head_fusion", default="mean", choices=["mean", "max", "min"])
    parser.add_argument("--discard_ratio", type=float, default=0.9)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    images = []
    for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
        images += glob.glob(str(Path(args.images_dir) / ext))
    images = sorted(set(images))[: args.max_images]
    if not images:
        raise SystemExit(f"No images in {args.images_dir}")

    print(f"[config] device={device} models={model_names} images={len(images)} "
          f"head_fusion={args.head_fusion} discard_ratio={args.discard_ratio}")

    rows = []
    for mname in model_names:
        print(f"\n=== {mname} ===", flush=True)
        model = load_model(mname, device)
        for i, img_path in enumerate(images, 1):
            name = Path(img_path).stem
            x, _ = load_image(img_path, device)
            with torch.no_grad():
                logits = model(x.unsqueeze(0))
                probs = F.softmax(logits[0], dim=0)
                target_class = int(torch.argmax(probs).item())

            heatmap_raw, t_method = run_attention_rollout(
                model, x, target_size=224,
                head_fusion=args.head_fusion, discard_ratio=args.discard_ratio,
            )
            heatmap = normalize_heatmap(heatmap_raw)
            metrics = evaluate_metrics(model, x, heatmap, target_class)
            row = {
                "Image": name, "Model": mname,
                **metrics,
                "Time": t_method,
            }
            rows.append(row)
            print(f"  [{i}/{len(images)}] {name}  "
                  f"ADCC={metrics['ADCC']:.3f} AD={metrics['AD']:.3f} "
                  f"ROAD={metrics['ROAD']:.2f} ROAD_Del={metrics['ROAD_Del']:.2f} "
                  f"ROAD_Ins={metrics['ROAD_Ins']:.2f} t={t_method:.2f}s",
                  flush=True)

    per_path = out_dir / "per_image.csv"
    keys = list(rows[0].keys())
    with open(per_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {per_path}")

    import pandas as pd
    df = pd.DataFrame(rows)
    metric_cols = ["ADCC", "AD", "Coh", "Com", "Inc", "ADD",
                   "ROAD", "ROAD_Del", "ROAD_Ins", "Time"]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    summary = out_dir / "summary.md"
    with open(summary, "w") as f:
        f.write(f"# Attention Rollout (Abnar & Zuidema 2020) on ViT — Table 1 row\n\n")
        f.write(f"head_fusion={args.head_fusion}  discard_ratio={args.discard_ratio}  "
                f"images={len(images)}  models={model_names}\n\n")
        f.write("## Per-model (mean ± std)\n\n")
        f.write("| Model | ADCC↑ | AD↓ | Coh↑ | Com↓ | Inc↑ | ADD↑ | ROAD↑ | ROAD_Del | ROAD_Ins | Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for m in model_names:
            sub = df[df["Model"] == m]
            def fmt(c, dec=4):
                mu, sd = sub[c].mean(), sub[c].std()
                return f"{mu:.{dec}f} ± {sd:.{dec}f}"
            f.write(f"| {m} | {fmt('ADCC')} | {fmt('AD')} | {fmt('Coh')} | "
                    f"{fmt('Com')} | {fmt('Inc')} | {fmt('ADD')} | "
                    f"{fmt('ROAD', 3)} | {fmt('ROAD_Del', 3)} | {fmt('ROAD_Ins', 3)} | "
                    f"{fmt('Time', 3)} |\n")

        f.write("\n## ViT pooled (B/16 + B/32)\n\n")
        f.write("| N | ADCC↑ | AD↓ | Coh↑ | Com↓ | Inc↑ | ADD↑ | ROAD↑ | ROAD_Del | ROAD_Ins | Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        sub = df
        def fmt(c, dec=4):
            mu, sd = sub[c].mean(), sub[c].std()
            return f"{mu:.{dec}f} ± {sd:.{dec}f}"
        f.write(f"| {len(sub)} | {fmt('ADCC')} | {fmt('AD')} | {fmt('Coh')} | "
                f"{fmt('Com')} | {fmt('Inc')} | {fmt('ADD')} | "
                f"{fmt('ROAD', 3)} | {fmt('ROAD_Del', 3)} | {fmt('ROAD_Ins', 3)} | "
                f"{fmt('Time', 3)} |\n")

    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
