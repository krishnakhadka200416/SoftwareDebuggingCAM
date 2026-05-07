"""Run only the 5 non-CAM baselines (EP, RISE, IG, SHAP, LIME) on the
NIH ChestX-ray14 bbox subset, with DenseNet-121, for Table 2 (RQ2) of the paper.

Each (image, finding) pair from BBox_List_2017.csv is processed once per method.
Metrics (IoU, Precision, Recall, num_regions) are computed by the same
compute_metrics() function used by minimality_localization.py — same threshold
(τ = 0.5), same 8-connectivity, same min-max normalization upstream of
binarization. So results are directly comparable to the existing Table 2.
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import torchxrayvision as xrv
from torchray.attribution.extremal_perturbation import (
    extremal_perturbation, PRESERVE_VARIANT, BLUR_PERTURBATION, simple_reward,
)
from torchray.attribution.rise import rise_class
from captum.attr import IntegratedGradients, GradientShap
import lime.lime_image

# Reuse the metric, path resolver, and pathology-index helpers from the original script
from minimality_localization import (
    compute_metrics, resolve_image_path, get_pathology_index,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="med_img")
    parser.add_argument("--bbox-csv", default="BBox_List_2017.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Cap on (image, finding) pairs to process (0 = all matching).")
    parser.add_argument("--out-csv", default="results_med_baselines/per_pair.csv")
    parser.add_argument("--shap-stdevs", type=float, default=0.0001,
                        help="Stdev for GradientShap. Original Table-2 setup uses 0.0001 (≈ IG).")
    args = parser.parse_args()

    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading DenseNet-121 (torchxrayvision)...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(True)
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
    print(f"Processing {len(candidates)} (image, finding) pairs from {args.dataset_path}.")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2048.0 - 1024.0),
    ])

    methods = ["EP", "RISE", "IG", "SHAP", "LIME"]
    results = {m: [] for m in methods}

    for row, img_path in tqdm(candidates, total=len(candidates), desc="med-img"):
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

        # ---------------- EP
        try:
            for p in model.parameters():
                p.requires_grad_(True)
            x_ep = x.unsqueeze(0)
            masks, _ = extremal_perturbation(
                model, x_ep, target_idx,
                areas=[0.1], perturbation=BLUR_PERTURBATION, max_iter=800,
                num_levels=8, step=7, sigma=21, jitter=True,
                variant=PRESERVE_VARIANT, reward_func=simple_reward,
                resize=False, smooth=0,
            )
            ep_cam = masks.sum(dim=0).squeeze(0).detach().cpu().numpy()
            mn, mx = ep_cam.min(), ep_cam.max()
            heat_ep = (ep_cam - mn) / (mx - mn) if mx > mn else np.zeros_like(ep_cam)
            if heat_ep.shape != (224, 224):
                heat_ep = cv2.resize(heat_ep, (224, 224), interpolation=cv2.INTER_LINEAR)
            m = compute_metrics(heat_ep, bbox, (oh, ow), threshold=args.threshold)
            m.update({"finding": finding, "image": image_name})
            results["EP"].append(m)
        except Exception as e:
            print(f"  [WARN] EP failed: {e}")

        # ---------------- RISE
        try:
            with torch.no_grad():
                x_rise = x.unsqueeze(0)
                saliency = rise_class(model, x_rise, target=[target_idx],
                                      num_masks=1000, num_cells=7, batch_size=128)
                cam_rise = saliency[0, 0].detach().cpu().numpy()
                mn, mx = cam_rise.min(), cam_rise.max()
                heat_rise = (cam_rise - mn) / (mx - mn) if mx > mn else np.zeros_like(cam_rise)
                if heat_rise.shape != (224, 224):
                    heat_rise = cv2.resize(heat_rise, (224, 224), interpolation=cv2.INTER_LINEAR)
            m = compute_metrics(heat_rise, bbox, (oh, ow), threshold=args.threshold)
            m.update({"finding": finding, "image": image_name})
            results["RISE"].append(m)
        except Exception as e:
            print(f"  [WARN] RISE failed: {e}")

        # ---------------- IG
        try:
            x_ig = x.unsqueeze(0); x_ig.requires_grad_(True)
            ig = IntegratedGradients(model)
            attributions = ig.attribute(x_ig, baselines=torch.zeros_like(x_ig),
                                        target=target_idx, n_steps=50)
            cam_ig = attributions[0].abs().sum(dim=0).detach().cpu().numpy()
            mn, mx = cam_ig.min(), cam_ig.max()
            heat_ig = (cam_ig - mn) / (mx - mn) if mx > mn else np.zeros_like(cam_ig)
            if heat_ig.shape != (224, 224):
                heat_ig = cv2.resize(heat_ig, (224, 224), interpolation=cv2.INTER_LINEAR)
            m = compute_metrics(heat_ig, bbox, (oh, ow), threshold=args.threshold)
            m.update({"finding": finding, "image": image_name})
            results["IG"].append(m)
        except Exception as e:
            print(f"  [WARN] IG failed: {e}")

        # ---------------- SHAP
        try:
            x_shap = x.unsqueeze(0); x_shap.requires_grad_(True)
            baselines_shap = torch.cat([
                torch.zeros_like(x_shap),
                torch.randn_like(x_shap) * 0.1,
            ], dim=0)
            gs = GradientShap(model)
            attributions = gs.attribute(x_shap, baselines=baselines_shap,
                                        target=target_idx,
                                        n_samples=50, stdevs=args.shap_stdevs)
            cam_shap = attributions[0].abs().sum(dim=0).detach().cpu().numpy()
            mn, mx = cam_shap.min(), cam_shap.max()
            heat_shap = (cam_shap - mn) / (mx - mn) if mx > mn else np.zeros_like(cam_shap)
            if heat_shap.shape != (224, 224):
                heat_shap = cv2.resize(heat_shap, (224, 224), interpolation=cv2.INTER_LINEAR)
            m = compute_metrics(heat_shap, bbox, (oh, ow), threshold=args.threshold)
            m.update({"finding": finding, "image": image_name})
            results["SHAP"].append(m)
        except Exception as e:
            print(f"  [WARN] SHAP failed: {e}")

        # ---------------- LIME
        try:
            def batch_predict_lime(images):
                batch = []
                for img_arr in images:
                    gray = np.mean(img_arr, axis=2) if img_arr.ndim == 3 else img_arr
                    t = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0)
                    t = t * 2048.0 - 1024.0
                    batch.append(t)
                batch = torch.stack(batch).to(device)
                with torch.no_grad():
                    logits = model(batch)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                return probs.cpu().numpy()
            img_gray = np.array(img.convert("L").resize((224, 224))).astype(np.float32) / 255.0
            img_rgb_for_lime = np.stack([img_gray] * 3, axis=-1)
            lime_explainer = lime.lime_image.LimeImageExplainer()
            explanation = lime_explainer.explain_instance(
                img_rgb_for_lime, batch_predict_lime,
                labels=(target_idx,), hide_color=0, num_samples=1000,
            )
            local_exp = explanation.local_exp[target_idx]
            segments = explanation.segments
            cam_lime = np.zeros(segments.shape, dtype=np.float32)
            for seg_id, weight in local_exp:
                cam_lime[segments == seg_id] = weight
            cam_lime = np.abs(cam_lime)
            mn, mx = cam_lime.min(), cam_lime.max()
            heat_lime = (cam_lime - mn) / (mx - mn) if mx > mn else np.zeros_like(cam_lime)
            if heat_lime.shape != (224, 224):
                heat_lime = cv2.resize(heat_lime, (224, 224), interpolation=cv2.INTER_LINEAR)
            m = compute_metrics(heat_lime, bbox, (oh, ow), threshold=args.threshold)
            m.update({"finding": finding, "image": image_name})
            results["LIME"].append(m)
        except Exception as e:
            print(f"  [WARN] LIME failed: {e}")

    # Save per-pair CSV
    rows_out = []
    for method, vals in results.items():
        for r in vals:
            rows_out.append({"Method": method, **r})
    if not rows_out:
        raise SystemExit("No measurements collected.")
    pd.DataFrame(rows_out).to_csv(args.out_csv, index=False)
    print(f"\nSaved per-pair: {args.out_csv}")

    # Summary table
    summary_lines = []
    summary_lines.append(f"# Table 2 (RQ2) — non-CAM baselines on med_img × DenseNet-121\n")
    summary_lines.append(f"N pairs per method: {len(rows_out) // 5}\n")
    summary_lines.append(f"Threshold: {args.threshold}  Connectivity: 8  Mean ± std across pairs\n\n")
    summary_lines.append("| Approach | IoU↑ | Precision↑ | Recall↑ | Regions↓ |\n")
    summary_lines.append("|---|---|---|---|---|\n")
    for m in methods:
        if not results[m]:
            summary_lines.append(f"| {m} | nan | nan | nan | nan |\n")
            continue
        df_m = pd.DataFrame(results[m])
        for c in ["iou", "precision", "recall", "num_regions"]:
            df_m[c] = pd.to_numeric(df_m[c], errors="coerce")
        summary_lines.append(
            f"| {m} | "
            f"{df_m['iou'].mean():.3f} ± {df_m['iou'].std():.3f} | "
            f"{df_m['precision'].mean():.3f} ± {df_m['precision'].std():.3f} | "
            f"{df_m['recall'].mean():.3f} ± {df_m['recall'].std():.3f} | "
            f"{df_m['num_regions'].mean():.2f} ± {df_m['num_regions'].std():.2f} |\n"
        )
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.writelines(summary_lines)
    print(f"Saved: {summary_path}")
    print()
    print("".join(summary_lines))


if __name__ == "__main__":
    main()
