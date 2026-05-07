"""Sanity check: does DD-CAM produce identical results when run multiple times
with the *same* seed? If yes, all non-determinism comes from universe ordering
(which is what our stability analysis varies). If no, there's additional
nondeterminism in the pipeline and the stability numbers are a lower bound.
"""

from pathlib import Path
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from ddcam.multi_class import MultiClassMinimalChannelsCNN


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path, device, size=224):
    img = Image.open(path).convert("RGB")
    pre = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return pre(img).to(device)


def run_once(model, target_layer, x, device, seed):
    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        probs = F.softmax(logits[0], dim=0)
        target_class = int(torch.argmax(probs).item())
        baseline_prob = float(probs[target_class].item())

    ex = MultiClassMinimalChannelsCNN(
        model, target_layer, device=device,
        target_class=target_class,
        retention_ratio=0.7, min_absolute_prob=0.01,
    )
    ex._baseline_class = target_class
    ex._baseline_prob = baseline_prob

    kept, forwards, final_prob, retention = ex.find_minimal(
        x, verbose=False,
        max_iterations=150, early_stop_size=30,
        initial_partition_n=None, seed=seed,
    )
    heatmap = ex.build_heatmap(x, kept)
    return tuple(sorted(int(c) for c in kept)), heatmap, float(retention), int(forwards)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    target_layer = model.layer4[-1].conv3

    images = sorted(glob.glob("new_test/*.JPEG"))[:5]
    seed = 42
    n_repeats = 5

    print(f"Running {n_repeats} repeats with seed={seed} on {len(images)} images.\n")
    print(f"{'image':<40} {'run':>3}  {'|S|':>4}  {'retention':>9}  {'forwards':>8}  {'kept_hash':>10}  {'heatmap_hash':>14}")

    for img_path in images:
        x = load_image(img_path, device)
        results = []
        for r in range(n_repeats):
            kept, hm, ret, fwd = run_once(model, target_layer, x, device, seed)
            kept_hash = hash(kept) & 0xffffffff
            hm_hash = hash(hm.tobytes()) & 0xffffffff
            results.append((kept, hm, ret, fwd))
            print(f"{Path(img_path).stem:<40} {r:>3}  {len(kept):>4}  {ret:>9.6f}  {fwd:>8d}  {kept_hash:>10x}  {hm_hash:>14x}")

        # Check identity across runs
        all_same_kept = all(results[i][0] == results[0][0] for i in range(n_repeats))
        all_same_hm = all(np.allclose(results[i][1], results[0][1]) for i in range(n_repeats))
        max_hm_diff = max(float(np.max(np.abs(results[i][1] - results[0][1]))) for i in range(n_repeats))
        all_same_ret = all(abs(results[i][2] - results[0][2]) < 1e-8 for i in range(n_repeats))

        print(f"  → kept identical:    {all_same_kept}")
        print(f"  → heatmap identical: {all_same_hm}  (max abs diff = {max_hm_diff:.2e})")
        print(f"  → retention identical: {all_same_ret}\n")


if __name__ == "__main__":
    main()
