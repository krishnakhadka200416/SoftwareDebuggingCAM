"""Run SIS on test_folder × 8 ImageNet models and compute RQ1 metrics
(ADCC, AD, Coh, Com, Inc, ADD, ROAD, ROAD_Del, ROAD_Ins, Time).

7×7 patch grid, threshold τ = 0.7 · p_0. Heatmap from backselect ordering.
"""
from __future__ import annotations
import argparse, csv, glob, time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from comparison import load_image_as_tensor, get_top1_class, load_model
from sis_module import backselect_image, backselect_to_heatmap, find_sis_from_backselect

from pytorch_grad_cam.metrics.ADCC import ADCC
from pytorch_grad_cam.metrics.road import (
    ROADCombined, ROADMostRelevantFirstAverage, ROADLeastRelevantFirstAverage,
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputReST, ClassifierOutputSoftmaxTarget, ClassifierOutputTarget,
)


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
        a, d, c, com, inc, dropdel = cam_metric(
            x.unsqueeze(0), heatmap[np.newaxis, ...], targets, metric_targets, model,
            lambda xi, t, _g=heatmap: torch.from_numpy(_g).unsqueeze(0),
        )
        out["ADCC"] = float(a[0]); out["AD"] = float(d[0])
        out["Coh"]  = float(c[0]); out["Com"] = float(com[0])
        out["Inc"]  = float(inc[0]); out["ADD"] = float(dropdel[0])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", default="test_folder")
    ap.add_argument("--max_images", type=int, default=10)
    ap.add_argument("--models",
                    default="resnet50,efficientnet_b0,inception_v3,vgg11,vgg16,vgg19,vit_b_16,vit_b_32")
    ap.add_argument("--grid_h", type=int, default=7)
    ap.add_argument("--grid_w", type=int, default=7)
    ap.add_argument("--threshold_alpha", type=float, default=0.7)
    ap.add_argument("--out_csv", default="results_sis_rq1/per_image.csv")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    images = []
    for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
        images += glob.glob(str(Path(args.images_dir) / ext))
    images = sorted(set(images))[: args.max_images]
    if not images:
        raise SystemExit(f"No images in {args.images_dir}")
    print(f"[config] device={device} models={model_names} images={len(images)} grid={args.grid_h}x{args.grid_w} α={args.threshold_alpha}")

    rows = []
    for mname in model_names:
        print(f"\n=== {mname} ===", flush=True)
        model, target_layers, reshape_transform, is_vit, image_size = load_model(mname, device)
        for i, img_path in enumerate(images, 1):
            name = Path(img_path).stem.lower()
            x, rgb = load_image_as_tensor(img_path, device, image_size=image_size)
            target = get_top1_class(model, x)
            with torch.no_grad():
                p0 = float(F.softmax(model(x.unsqueeze(0))[0], dim=0)[target].item())
            threshold = args.threshold_alpha * p0
            fully_masked = torch.zeros_like(x)
            t0 = time.perf_counter()
            try:
                stack = backselect_image(model, x, fully_masked, target,
                                         grid_h=args.grid_h, grid_w=args.grid_w, batch_size=32)
                sis_idxs = find_sis_from_backselect(stack, threshold)
                heatmap = backselect_to_heatmap(stack, args.grid_h, args.grid_w,
                                                image_size, image_size)
                heatmap = normalize_heatmap(heatmap)
            except Exception as e:
                print(f"  [WARN] SIS failed on {name}: {e}")
                continue
            elapsed = time.perf_counter() - t0
            metrics = evaluate_metrics(model, x, heatmap, target)
            row = {"Image": name, "Model": mname,
                   "p0": p0, "threshold": threshold,
                   "sis_size": len(sis_idxs),
                   **metrics, "Time": elapsed}
            rows.append(row)
            print(f"  [{i}/{len(images)}] {name}  |SIS|={len(sis_idxs):<2d} "
                  f"ADCC={metrics['ADCC']:.3f} AD={metrics['AD']:.3f} ROAD={metrics['ROAD']:.2f} "
                  f"t={elapsed:.1f}s", flush=True)

    if not rows:
        raise SystemExit("No measurements collected.")
    keys = list(rows[0].keys())
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {args.out_csv}")

    import pandas as pd
    df = pd.DataFrame(rows)
    metric_cols = ["sis_size", "ADCC", "AD", "Coh", "Com", "Inc", "ADD", "ROAD", "ROAD_Del", "ROAD_Ins", "Time"]
    for c in metric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    summary = Path(args.out_csv).parent / "summary.md"
    groups = {
        "CNN Linear": ["resnet50", "efficientnet_b0", "inception_v3"],
        "CNN Non-Linear": ["vgg11", "vgg16", "vgg19"],
        "ViT": ["vit_b_16", "vit_b_32", "vit_l_16"],
    }
    with open(summary, "w") as f:
        f.write("# SIS (Sufficient Input Subsets) RQ1 — test_folder × 8 ImageNet models\n\n")
        f.write(f"Patch grid: {args.grid_h}×{args.grid_w} = {args.grid_h*args.grid_w} patches; "
                f"masking: zero; threshold τ = {args.threshold_alpha}·p₀.\n\n")
        f.write("## Per-model (mean ± std)\n\n")
        f.write("| Model | |SIS| | ADCC↑ | AD↓ | Coh↑ | Com↓ | Inc↑ | ADD↑ | ROAD↑ | ROAD_Del | ROAD_Ins | Time (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for m in model_names:
            sub = df[df["Model"] == m]
            if len(sub) == 0: continue
            def fmt(c, dec=4):
                return f"{sub[c].mean():.{dec}f} ± {sub[c].std():.{dec}f}"
            f.write(f"| {m} | {fmt('sis_size', 1)} | {fmt('ADCC')} | {fmt('AD')} | "
                    f"{fmt('Coh')} | {fmt('Com')} | {fmt('Inc')} | {fmt('ADD')} | "
                    f"{fmt('ROAD', 3)} | {fmt('ROAD_Del', 3)} | {fmt('ROAD_Ins', 3)} | {fmt('Time', 2)} |\n")
        f.write("\n## Group averages\n\n")
        for gname, gmodels in groups.items():
            sub = df[df["Model"].isin(gmodels)]
            if len(sub) == 0: continue
            def fmt(c, dec=4):
                return f"{sub[c].mean():.{dec}f} ± {sub[c].std():.{dec}f}"
            f.write(f"### {gname} (N = {len(sub)})\n\n")
            f.write("| ADCC↑ | AD↓ | Coh↑ | Com↓ | Inc↑ | ADD↑ | ROAD↑ | ROAD_Del | ROAD_Ins | Time (s) |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")
            f.write(f"| {fmt('ADCC')} | {fmt('AD')} | {fmt('Coh')} | {fmt('Com')} | "
                    f"{fmt('Inc')} | {fmt('ADD')} | {fmt('ROAD', 3)} | {fmt('ROAD_Del', 3)} | "
                    f"{fmt('ROAD_Ins', 3)} | {fmt('Time', 2)} |\n\n")
    print(f"Saved: {summary}")


if __name__ == "__main__":
    main()
