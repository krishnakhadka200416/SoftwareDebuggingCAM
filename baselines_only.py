"""Run only the 5 non-CAM baselines (EP, RISE, IG, SHAP, LIME) on a set of images
and models, and compute the standard faithfulness metrics (ROAD, ADCC, AD, etc.).

Skips all CAM-family methods and DD-CAM to save wall time when you already have
those numbers. Output matches the CSV schema from comparison.py so rows can be
concatenated to the existing results file.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comparison import (
    load_image_as_tensor,
    get_top1_class,
    load_model,
    run_rise,
    run_integrated_gradients,
    run_gradient_shap,
    run_lime_explanation,
    run_extremal_perturbation,
    compute_cam_scores,
)

from pytorch_grad_cam.metrics.ADCC import ADCC
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputReST,
    ClassifierOutputSoftmaxTarget,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="new_test")
    parser.add_argument("--max_images", type=int, default=100)
    parser.add_argument("--models", required=True,
                        help="Comma-separated list of model names.")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write the metrics CSV.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.JPEG", "*.png"):
        image_paths += glob.glob(os.path.join(args.images_dir, ext))
    image_paths = sorted(set(os.path.normcase(os.path.abspath(p)) for p in image_paths))
    image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise SystemExit(f"No images in {args.images_dir}")

    print(f"[config] device={device} models={model_names} images={len(image_paths)} out={args.out_dir}")

    csv_rows = []

    for model_name in model_names:
        print(f"\n=== {model_name} ===", flush=True)
        model, target_layers, reshape_transform, is_vit, image_size = load_model(model_name, device)

        for idx, img_path in enumerate(image_paths, 1):
            print(f"[{model_name}] [{idx}/{len(image_paths)}] {os.path.basename(img_path)}", flush=True)
            try:
                input_tensor, rgb_img = load_image_as_tensor(img_path, device, image_size=image_size)
                target_class = get_top1_class(model, input_tensor)

                cam_metric = ADCC()
                targets = [ClassifierOutputReST(target_class)]
                metric_targets = [ClassifierOutputSoftmaxTarget(target_class)]

                methods_outputs = {}

                try:
                    ep_gray, _, ep_time = run_extremal_perturbation(
                        model, input_tensor, rgb_img, target_class, device,
                        areas=[0.1], max_iter=800,
                    )
                    methods_outputs["EP"] = (ep_gray, ep_time)
                except Exception as e:
                    print(f"  [WARN] EP failed: {e}")
                    methods_outputs["EP"] = (None, float("nan"))

                try:
                    rise_gray, _, rise_time = run_rise(
                        model, input_tensor, rgb_img, target_class, device,
                        num_masks=1000, num_cells=7, batch_size=128,
                    )
                    methods_outputs["RISE"] = (rise_gray, rise_time)
                except Exception as e:
                    print(f"  [WARN] RISE failed: {e}")
                    methods_outputs["RISE"] = (None, float("nan"))

                try:
                    ig_gray, _, ig_time = run_integrated_gradients(
                        model, input_tensor, rgb_img, target_class, device, n_steps=50,
                    )
                    methods_outputs["IG"] = (ig_gray, ig_time)
                except Exception as e:
                    print(f"  [WARN] IG failed: {e}")
                    methods_outputs["IG"] = (None, float("nan"))

                try:
                    shap_gray, _, shap_time = run_gradient_shap(
                        model, input_tensor, rgb_img, target_class, device,
                        n_samples=50, stdevs=0.0001,
                    )
                    methods_outputs["SHAP"] = (shap_gray, shap_time)
                except Exception as e:
                    print(f"  [WARN] SHAP failed: {e}")
                    methods_outputs["SHAP"] = (None, float("nan"))

                try:
                    lime_gray, _, lime_time = run_lime_explanation(
                        model, input_tensor, rgb_img, target_class, device,
                        num_samples=1000,
                    )
                    methods_outputs["LIME"] = (lime_gray, lime_time)
                except Exception as e:
                    print(f"  [WARN] LIME failed: {e}")
                    methods_outputs["LIME"] = (None, float("nan"))

                base = os.path.splitext(os.path.basename(img_path))[0]
                for method_name, (gray, m_time) in methods_outputs.items():
                    if gray is None:
                        continue
                    road = compute_cam_scores(model, input_tensor, gray, target_class)
                    try:
                        _g = gray
                        adcc_v, drop_v, coh_v, com_v, inc_v, dropindel_v = cam_metric(
                            input_tensor.unsqueeze(0), gray[np.newaxis, ...], targets, metric_targets, model,
                            lambda x, t, _g=_g: torch.from_numpy(_g).unsqueeze(0),
                        )
                        csv_rows.append({
                            "Image": base, "Model": model_name, "Method": method_name,
                            "ROAD": road, "ADCC": adcc_v[0], "Drop": drop_v[0],
                            "Coherency": coh_v[0], "Complexity": com_v[0],
                            "Inc": inc_v[0], "DropInDeletion": dropindel_v[0],
                            "Time": m_time,
                        })
                    except Exception as e:
                        print(f"  [WARN] ADCC failed for {method_name}: {e}")
            except Exception as e:
                import traceback
                print(f"  [ERROR] image failed: {e}")
                print(traceback.format_exc())

    csv_path = os.path.join(args.out_dir, "baselines_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        fields = ["Image", "Model", "Method", "ROAD", "ADCC", "Drop",
                  "Coherency", "Complexity", "Inc", "DropInDeletion", "Time"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)
    print(f"\n✓ Saved: {csv_path}")

    # Aggregated by model/method
    if csv_rows:
        from collections import defaultdict
        groups = defaultdict(list)
        for r in csv_rows:
            groups[(r["Model"], r["Method"])].append(r)
        metric_fields = ["ROAD", "ADCC", "Drop", "Coherency", "Complexity", "Inc", "DropInDeletion", "Time"]
        avg_rows = []
        for (mdl, mth), rows in sorted(groups.items()):
            row = {"Model": mdl, "Method": mth, "NumImages": len(rows)}
            for mf in metric_fields:
                vals = [float(r[mf]) for r in rows if not (isinstance(r[mf], float) and np.isnan(r[mf]))]
                row[mf] = round(float(np.mean(vals)), 6) if vals else float("nan")
            avg_rows.append(row)
        avg_csv = os.path.join(args.out_dir, "baselines_metrics_avg.csv")
        with open(avg_csv, "w", newline="") as f:
            fields = ["Model", "Method", "NumImages"] + metric_fields
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in avg_rows:
                w.writerow(r)
        print(f"✓ Saved: {avg_csv}")


if __name__ == "__main__":
    main()
