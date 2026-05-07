"""Run SIS on med_img × DenseNet-121 for RQ2 (localization) metrics:
IoU, Precision, Recall, Regions vs the NIH ChestX-ray bbox annotations.

Same compute_metrics() as minimality_localization.py (τ=0.5, 8-connectivity).
"""
from __future__ import annotations
import argparse, csv, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import torchxrayvision as xrv

from minimality_localization import compute_metrics, resolve_image_path, get_pathology_index
from sis_module import backselect_image, backselect_to_heatmap, find_sis_from_backselect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="med_img")
    ap.add_argument("--bbox-csv", default="BBox_List_2017.csv")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Binarization threshold for heatmap (matches minimality_localization).")
    ap.add_argument("--threshold_alpha", type=float, default=0.7,
                    help="SIS τ = α·p₀ for backselect.")
    ap.add_argument("--grid_h", type=int, default=7)
    ap.add_argument("--grid_w", type=int, default=7)
    ap.add_argument("--out-csv", default="results_sis_rq2/per_pair.csv")
    ap.add_argument("--max-samples", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DenseNet-121 (torchxrayvision)...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    labels = model.pathologies

    df = pd.read_csv(args.bbox_csv)
    if len(df.columns) >= 6:
        colmap = {df.columns[0]: "Image Index", df.columns[1]: "Finding Label",
                  df.columns[2]: "x", df.columns[3]: "y",
                  df.columns[4]: "w", df.columns[5]: "h"}
        df = df.rename(columns=colmap)
    df = df.dropna(subset=["x", "y", "w", "h"])

    root = Path(args.dataset_path)
    candidates = []
    for _, r in df.iterrows():
        p = resolve_image_path(root, str(r["Image Index"]))
        if p is not None:
            candidates.append((r, p))
    if not candidates:
        raise SystemExit(f"No matching images in {args.dataset_path}.")
    if args.max_samples and args.max_samples > 0:
        candidates = candidates[: args.max_samples]
    print(f"Processing {len(candidates)} (image, finding) pairs.")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2048.0 - 1024.0),
    ])

    results = []
    for row, img_path in candidates:
        image_name = str(row["Image Index"])
        finding = str(row["Finding Label"])
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"failed to open {img_path}: {e}")
            continue
        ow, oh = img.size
        bbox = (float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"]))
        x = preprocess(img).to(device)
        target_idx = get_pathology_index(finding, labels)
        if target_idx is None:
            with torch.no_grad():
                logits = model(x.unsqueeze(0))
                target_idx = int(torch.argmax(logits if logits.ndim == 1 else logits[0]).item())

        # Baseline probability for threshold
        with torch.no_grad():
            logits = model(x.unsqueeze(0))[0]
            p0 = float(torch.sigmoid(logits[target_idx]).item()) if logits.shape[0] < 30 \
                  else float(torch.softmax(logits, dim=0)[target_idx].item())
        # NIH torchxrayvision uses sigmoid (multi-label), so use sigmoid-based probability.

        threshold_sis = args.threshold_alpha * p0
        # The masked-input value: zeros after the (x*2048-1024) preprocessing means raw 0 input.
        # Since preprocessing is x*2048-1024 (mapping [0,1] → [-1024, 1024]), a raw 0.5 image
        # would map to 0, so "fully masked" = mid-gray. Simplest faithful default: zeros tensor.
        fully_masked = torch.zeros_like(x)

        t0 = time.perf_counter()
        try:
            stack = backselect_image(model, x, fully_masked, target_idx,
                                     grid_h=args.grid_h, grid_w=args.grid_w, batch_size=32,
                                     score="sigmoid")
            sis_idxs = find_sis_from_backselect(stack, threshold_sis)
            heatmap = backselect_to_heatmap(stack, args.grid_h, args.grid_w, 224, 224)
            # Min-max normalize
            lo, hi = heatmap.min(), heatmap.max()
            if hi > lo:
                heatmap = (heatmap - lo) / (hi - lo)
            else:
                heatmap = np.zeros_like(heatmap)
        except Exception as e:
            print(f"  [WARN] SIS failed: {e}")
            continue
        elapsed = time.perf_counter() - t0

        m = compute_metrics(heatmap, bbox, (oh, ow), threshold=args.threshold)
        m.update({"finding": finding, "image": image_name,
                  "sis_size": len(sis_idxs), "p0": p0, "time": elapsed})
        results.append(m)
        print(f"  {image_name} {finding}  |SIS|={len(sis_idxs):<2d} "
              f"IoU={m['iou']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"reg={m['num_regions']} t={elapsed:.1f}s", flush=True)

    keys = list(results[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved: {args.out_csv}")

    df_out = pd.DataFrame(results)
    summary = out_dir / "summary.md"
    with open(summary, "w") as f:
        f.write("# SIS RQ2 — med_img × DenseNet-121\n\n")
        f.write(f"N pairs: {len(df_out)}; patch grid: {args.grid_h}×{args.grid_w}; "
                f"τ_SIS = {args.threshold_alpha}·p₀; binarization τ = {args.threshold}; "
                f"8-connectivity for region count.\n\n")
        f.write("| Method | IoU↑ | Precision↑ | Recall↑ | Regions↓ |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| SIS | {df_out['iou'].mean():.3f} ± {df_out['iou'].std():.3f} | "
                f"{df_out['precision'].mean():.3f} ± {df_out['precision'].std():.3f} | "
                f"{df_out['recall'].mean():.3f} ± {df_out['recall'].std():.3f} | "
                f"{df_out['num_regions'].mean():.2f} ± {df_out['num_regions'].std():.2f} |\n")
    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
