"""
Task 2: Stability Analysis of 1-Minimal Sets (DD-CAM, CNN / ResNet-50).

For each image in a small subset we run delta debugging K times, each with a
different random ordering of the channel universe. We then report:

  - Jaccard similarity between the 1-minimal sets across runs
  - Variance of |S| and of retention (faithfulness) across seeds
  - Variance of IoU between resulting saliency maps (thresholded)
  - Side-by-side saliency maps from different seeds

Output:
  results_stability/
    per_image_<image>_<model>.csv     per-seed sizes/retention/passes/kept
    summary_<model>.csv               aggregated stats per image
    grid_<image>_<model>.png          side-by-side overlays for 5 seeds
    stability_report.md               human-readable summary
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from ddcam.multi_class import MultiClassMinimalChannelsCNN


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


def overlay(rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    h = np.uint8(255 * np.clip(heatmap, 0, 1))
    color = cv2.applyColorMap(h, cv2.COLORMAP_JET).astype(np.float32) / 255.0
    color = color[..., ::-1]
    out = color * alpha + rgb * (1 - alpha)
    out = out / max(out.max(), 1e-8)
    return np.uint8(255 * out)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def heatmap_iou(h1: np.ndarray, h2: np.ndarray, thresh: float = 0.5) -> float:
    m1 = (h1 >= thresh).astype(np.uint8)
    m2 = (h2 >= thresh).astype(np.uint8)
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    if union == 0:
        return 1.0
    return inter / union


def normalize_heatmap(h: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32)
    lo, hi = float(h.min()), float(h.max())
    if hi - lo < 1e-8:
        return np.zeros_like(h)
    return (h - lo) / (hi - lo)


def pick_model(device: str):
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    target_layer = model.layer4[-1].conv3
    return model, target_layer


def run_one_seed(
    model,
    target_layer,
    x: torch.Tensor,
    target_class: int,
    baseline_prob: float,
    seed: int,
    device: str,
    retention_ratio: float,
    min_absolute: float,
    max_iterations: int,
    early_stop_size: int,
) -> Dict:
    """Run a single DD-CAM invocation with a specific seed. Returns per-seed record."""
    explainer = MultiClassMinimalChannelsCNN(
        model,
        target_layer,
        device=device,
        target_class=target_class,
        retention_ratio=retention_ratio,
        min_absolute_prob=min_absolute,
    )
    explainer._baseline_class = target_class
    explainer._baseline_prob = baseline_prob

    kept, forwards, final_prob, retention = explainer.find_minimal(
        x,
        verbose=False,
        max_iterations=max_iterations,
        early_stop_size=early_stop_size,
        initial_partition_n=None,
        seed=seed,
    )
    heatmap = explainer.build_heatmap(x, kept)
    return {
        "seed": seed,
        "kept": list(map(int, kept)),
        "size": len(kept),
        "forwards": forwards,
        "final_prob": float(final_prob),
        "retention": float(retention),
        "heatmap": heatmap.astype(np.float32),
    }


def analyze_image(
    image_path: str,
    model,
    target_layer,
    device: str,
    seeds: List[int],
    retention_ratio: float,
    min_absolute: float,
    max_iterations: int,
    early_stop_size: int,
    out_dir: Path,
    model_name: str,
):
    x, rgb = load_image(image_path, device)

    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs = F.softmax(logits[0], dim=0)
        target_class = int(torch.argmax(probs).item())
        baseline_prob = float(probs[target_class].item())

    records: List[Dict] = []
    print(f"  baseline: class={target_class}, p={baseline_prob:.4f}")
    for s in seeds:
        r = run_one_seed(
            model,
            target_layer,
            x,
            target_class,
            baseline_prob,
            s,
            device,
            retention_ratio,
            min_absolute,
            max_iterations,
            early_stop_size,
        )
        records.append(r)
        print(f"    seed={s:<4d} |S|={r['size']:<3d} retention={r['retention']:.3f} fwd={r['forwards']}")

    sets = [set(r["kept"]) for r in records]
    pairs = list(itertools.combinations(range(len(sets)), 2))
    jaccs = [jaccard(sets[i], sets[j]) for i, j in pairs]

    heatmaps = [normalize_heatmap(r["heatmap"]) for r in records]
    ious = [heatmap_iou(heatmaps[i], heatmaps[j], thresh=0.5) for i, j in pairs]

    sizes = np.array([r["size"] for r in records], dtype=float)
    retentions = np.array([r["retention"] for r in records], dtype=float)
    fwds = np.array([r["forwards"] for r in records], dtype=float)

    union_all = set().union(*sets)
    inter_all = set.intersection(*sets) if sets else set()
    stability_core = len(inter_all) / max(1, len(union_all))

    stem = Path(image_path).stem
    per_image_csv = out_dir / f"per_image_{stem}_{model_name}.csv"
    with open(per_image_csv, "w") as f:
        f.write("seed,size,retention,final_prob,forwards\n")
        for r in records:
            f.write(f"{r['seed']},{r['size']},{r['retention']:.6f},{r['final_prob']:.6f},{r['forwards']}\n")

    summary = {
        "image": stem,
        "model": model_name,
        "target_class": target_class,
        "baseline_prob": baseline_prob,
        "n_seeds": len(seeds),
        "size_mean": float(sizes.mean()),
        "size_std": float(sizes.std()),
        "size_cv": float(sizes.std() / max(sizes.mean(), 1e-8)),
        "retention_mean": float(retentions.mean()),
        "retention_std": float(retentions.std()),
        "forwards_mean": float(fwds.mean()),
        "union_size": len(union_all),
        "intersection_size": len(inter_all),
        "core_stability": stability_core,
        "jaccard_mean": float(np.mean(jaccs)) if jaccs else 1.0,
        "jaccard_std": float(np.std(jaccs)) if jaccs else 0.0,
        "jaccard_min": float(np.min(jaccs)) if jaccs else 1.0,
        "jaccard_max": float(np.max(jaccs)) if jaccs else 1.0,
        "heatmap_iou_mean": float(np.mean(ious)) if ious else 1.0,
        "heatmap_iou_std": float(np.std(ious)) if ious else 0.0,
    }

    num_panels = min(6, len(records))
    pick = np.linspace(0, len(records) - 1, num_panels, dtype=int).tolist()
    fig, axes = plt.subplots(1, num_panels + 1, figsize=(3.2 * (num_panels + 1), 3.2))
    axes[0].imshow(rgb)
    axes[0].set_title(f"original\nclass={target_class} p={baseline_prob:.2f}", fontsize=9)
    axes[0].axis("off")
    for ax_i, seed_idx in enumerate(pick, start=1):
        r = records[seed_idx]
        hm = normalize_heatmap(r["heatmap"])
        ov = overlay(rgb, hm)
        axes[ax_i].imshow(ov)
        axes[ax_i].set_title(
            f"seed={r['seed']}\n|S|={r['size']}, ret={r['retention']:.2f}",
            fontsize=9,
        )
        axes[ax_i].axis("off")
    fig.suptitle(
        f"{stem} — Jaccard mean={summary['jaccard_mean']:.2f}, "
        f"heatmap IoU mean={summary['heatmap_iou_mean']:.2f}",
        fontsize=11,
    )
    plt.tight_layout()
    grid_path = out_dir / f"grid_{stem}_{model_name}.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return summary


def write_summary_csv(out_dir: Path, model_name: str, rows: List[Dict]):
    path = out_dir / f"summary_{model_name}.csv"
    keys = list(rows[0].keys())
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(
                f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k])
                for k in keys
            ) + "\n")
    return path


def write_report(out_dir: Path, model_name: str, rows: List[Dict], cfg: Dict):
    path = out_dir / f"stability_report_{model_name}.md"
    jacc = np.array([r["jaccard_mean"] for r in rows])
    iou = np.array([r["heatmap_iou_mean"] for r in rows])
    size_cv = np.array([r["size_cv"] for r in rows])
    ret_std = np.array([r["retention_std"] for r in rows])
    core = np.array([r["core_stability"] for r in rows])

    with open(path, "w") as f:
        f.write(f"# DD-CAM Stability Analysis ({model_name})\n\n")
        f.write(f"Images analyzed: {len(rows)}  |  Seeds per image: {cfg['n_seeds']}\n")
        f.write(f"Retention ratio α = {cfg['retention_ratio']}  |  "
                f"Min absolute prob = {cfg['min_absolute']}  |  "
                f"Max iterations = {cfg['max_iterations']}  |  "
                f"Early stop size = {cfg['early_stop_size']}\n\n")

        f.write("## Aggregate stability across images\n\n")
        f.write("| metric | mean | std | min | max |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Jaccard(minimal sets) | {jacc.mean():.3f} | {jacc.std():.3f} | {jacc.min():.3f} | {jacc.max():.3f} |\n")
        f.write(f"| IoU(heatmaps @0.5)     | {iou.mean():.3f} | {iou.std():.3f} | {iou.min():.3f} | {iou.max():.3f} |\n")
        f.write(f"| |S| CV (size/variation) | {size_cv.mean():.3f} | {size_cv.std():.3f} | {size_cv.min():.3f} | {size_cv.max():.3f} |\n")
        f.write(f"| retention std          | {ret_std.mean():.4f} | {ret_std.std():.4f} | {ret_std.min():.4f} | {ret_std.max():.4f} |\n")
        f.write(f"| core stability (|∩|/|∪|)| {core.mean():.3f} | {core.std():.3f} | {core.min():.3f} | {core.max():.3f} |\n")

        f.write("\n## Per-image details\n\n")
        f.write("| image | |S| mean ± std | retention mean ± std | Jaccard mean | IoU mean | core stability |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['image']} | {r['size_mean']:.1f} ± {r['size_std']:.1f} | "
                    f"{r['retention_mean']:.3f} ± {r['retention_std']:.3f} | "
                    f"{r['jaccard_mean']:.3f} | {r['heatmap_iou_mean']:.3f} | "
                    f"{r['core_stability']:.3f} |\n")

        f.write(
            "\n## Interpretation\n\n"
            "- **Jaccard(minimal sets)** measures how much the 1-minimal channel sets agree across seeds.\n"
            "  Values close to 1 → DD-CAM converges to essentially the same subset regardless of ordering.\n"
            "- **core stability** = |∩seeds| / |∪seeds| is a stricter measure: the fraction of channels that\n"
            "  *every* seed agrees on, relative to the union.\n"
            "- **|S| CV** is the coefficient of variation of minimal-set size. Low CV → stable minimality.\n"
            "- **retention std** quantifies the spread of final faithfulness (prob retained on the kept set).\n"
            "- **IoU(heatmaps @0.5)** compares the *localized spatial attribution* between seeds, which matters\n"
            "  more to a human reader than raw channel-set identity.\n\n"
            "### Why stability is not free\n\n"
            "Delta debugging (Zeller 1999) is only guaranteed to return *a* 1-minimal subset — not *the* unique\n"
            "one. When the sufficiency predicate is non-monotone (as for deep networks, where feature\n"
            "interactions mean removing a channel can sometimes *increase* a competing class's probability),\n"
            "multiple 1-minimal sets can exist and different orderings can reach different ones. Empirical\n"
            "stability across random orderings is therefore the right validation: the heatmap-level IoU\n"
            "captures what the user actually sees, and core stability captures the invariant basis DD-CAM\n"
            "recovers across runs.\n"
        )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_glob", default="test_folder/*.JPEG",
                        help="Glob for images to analyze.")
    parser.add_argument("--n_seeds", type=int, default=12)
    parser.add_argument("--retention_ratio", type=float, default=0.7)
    parser.add_argument("--min_absolute", type=float, default=0.01)
    parser.add_argument("--max_iterations", type=int, default=150)
    parser.add_argument("--early_stop_size", type=int, default=30)
    parser.add_argument("--out_dir", default="results_stability")
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42 + i * 17 for i in range(args.n_seeds)]
    images = sorted(glob.glob(args.image_glob))
    if args.max_images:
        images = images[: args.max_images]
    if not images:
        raise SystemExit(f"No images matched glob: {args.image_glob}")

    print(f"[config] device={device}  seeds={seeds}  images={len(images)}")
    print(f"[config] α={args.retention_ratio}  min_abs={args.min_absolute}  "
          f"max_iter={args.max_iterations}  early_stop_size={args.early_stop_size}")

    model_name = "resnet50"
    model, target_layer = pick_model(device)

    cfg = dict(
        n_seeds=args.n_seeds,
        retention_ratio=args.retention_ratio,
        min_absolute=args.min_absolute,
        max_iterations=args.max_iterations,
        early_stop_size=args.early_stop_size,
    )

    rows = []
    for i, path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {path}")
        summary = analyze_image(
            path, model, target_layer, device, seeds,
            args.retention_ratio, args.min_absolute,
            args.max_iterations, args.early_stop_size,
            out_dir, model_name,
        )
        rows.append(summary)

    csv_path = write_summary_csv(out_dir, model_name, rows)
    report_path = write_report(out_dir, model_name, rows, cfg)
    with open(out_dir / f"config_{model_name}.json", "w") as f:
        json.dump({**cfg, "seeds": seeds, "device": device}, f, indent=2)

    print(f"\n✓ Summary CSV: {csv_path}")
    print(f"✓ Report:     {report_path}")
    print(f"✓ Grids saved to: {out_dir}")


if __name__ == "__main__":
    main()
