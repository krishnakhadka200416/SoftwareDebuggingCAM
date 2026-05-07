import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import torchxrayvision as xrv
from pytorch_grad_cam import (
    GradCAM,
    GradCAMPlusPlus,
    ScoreCAM,
    AblationCAM,
    XGradCAM,
    LayerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from ddcam import DDCAMExplainer
from reciprocam import ReciproCam
from torchray.attribution.extremal_perturbation import extremal_perturbation, PRESERVE_VARIANT, BLUR_PERTURBATION, simple_reward
from torchray.attribution.rise import rise_class
from captum.attr import IntegratedGradients, GradientShap
import lime.lime_image


def overlay_heatmap_on_image(img_array: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if img_array.max() > 1.0:
        img_array = img_array.astype(np.float32) / 255.0
    else:
        img_array = img_array.astype(np.float32)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if heatmap.shape != img_array.shape[:2]:
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_LINEAR)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    heatmap = np.clip(heatmap, 0, 1)
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
    overlay = np.clip(alpha * heatmap_color + (1 - alpha) * img_array, 0, 1)
    return (overlay * 255).astype(np.uint8)


def visualize(
    img: Image.Image,
    bbox,
    finding_label: str,
    heat_ddcam: np.ndarray,
    baseline_heatmaps: dict,
    heat_reciprocam: np.ndarray,
    out_dir: Path,
    image_name: str,
    heat_ep: np.ndarray = None,
    extra_heatmaps: dict = None,
) -> str:
    """
    Visualize with ordering: Original, DD-CAM, Grad-CAM, Grad-CAM++, XGRAD-CAM, Layer-CAM,
    Score-CAM, Ablation-CAM, ReciproCAM, EP, RISE, IG, SHAP, LIME
    """
    if extra_heatmaps is None:
        extra_heatmaps = {}
    img_np = np.array(img.convert('RGB'))
    h, w = img_np.shape[:2]
    x, y, bw, bh = bbox

    method_order = ['GradCAM', 'GradCAM++', 'XGradCAM', 'LayerCAM', 'ScoreCAM', 'AblationCAM']

    available_baselines = []
    for method in method_order:
        if method in baseline_heatmaps and isinstance(baseline_heatmaps[method], np.ndarray) and baseline_heatmaps[method].size > 0:
            available_baselines.append((method, baseline_heatmaps[method]))

    n_cols = 2 + len(available_baselines) + 1 + 1 + len(extra_heatmaps)  # +1 ReciproCAM, +1 EP, +extra
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    axes[0].imshow(img_np)
    from matplotlib.patches import Rectangle
    axes[0].add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor='lime', facecolor='none'))
    axes[0].set_title(f'Label: {finding_label}', fontsize=14)
    axes[0].axis('off')

    col = 1
    if heat_ddcam is not None:
        hm = cv2.resize(heat_ddcam, (w, h), interpolation=cv2.INTER_LINEAR)
        if hm.max() > 0:
            hm = hm / hm.max()
        axes[col].imshow(overlay_heatmap_on_image(img_np, hm, alpha=0.5))
        axes[col].add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor='lime', facecolor='none'))
        axes[col].set_title('DDCAM', fontsize=14)
    else:
        axes[col].imshow(img_np)
        axes[col].set_title('DDCAM (failed)', fontsize=14)
    axes[col].axis('off')
    col += 1

    for name, heat in available_baselines:
        try:
            hh = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
            if hh.max() > 0:
                hh = hh / hh.max()
            axes[col].imshow(overlay_heatmap_on_image(img_np, hh, alpha=0.5))
            axes[col].add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor='lime', facecolor='none'))
            axes[col].set_title(name, fontsize=14)
        except Exception:
            axes[col].imshow(img_np)
            axes[col].set_title(f'{name} (failed)', fontsize=14)
        axes[col].axis('off')
        col += 1

    if heat_reciprocam is not None:
        hr = cv2.resize(heat_reciprocam, (w, h), interpolation=cv2.INTER_LINEAR)
        if hr.max() > 0:
            hr = hr / hr.max()
        axes[col].imshow(overlay_heatmap_on_image(img_np, hr, alpha=0.5))
        axes[col].add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor='lime', facecolor='none'))
        axes[col].set_title('ReciproCAM', fontsize=14)
    else:
        axes[col].imshow(img_np)
        axes[col].set_title('ReciproCAM (failed)', fontsize=14)
    axes[col].axis('off')
    col += 1

    if heat_ep is not None:
        he = cv2.resize(heat_ep, (w, h), interpolation=cv2.INTER_LINEAR)
        if he.max() > 0:
            he = he / he.max()
        axes[col].imshow(overlay_heatmap_on_image(img_np, he, alpha=0.5))
        axes[col].add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor='lime', facecolor='none'))
        axes[col].set_title('EP', fontsize=14)
    else:
        axes[col].imshow(img_np)
        axes[col].set_title('EP (failed)', fontsize=14)
    axes[col].axis('off')
    col += 1

    # Extra method panels (RISE, IG, SHAP, LIME)
    for method_name, heat_extra in extra_heatmaps.items():
        if col >= n_cols:
            break
        if heat_extra is not None:
            hx = cv2.resize(heat_extra, (w, h), interpolation=cv2.INTER_LINEAR)
            if hx.max() > 0:
                hx = hx / hx.max()
            axes[col].imshow(overlay_heatmap_on_image(img_np, hx, alpha=0.5))
            axes[col].add_patch(Rectangle((x, y), bw, bh, linewidth=2, edgecolor='lime', facecolor='none'))
            axes[col].set_title(method_name, fontsize=14)
        else:
            axes[col].imshow(img_np)
            axes[col].set_title(f'{method_name} (failed)', fontsize=14)
        axes[col].axis('off')
        col += 1

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = image_name.replace('/', '_').replace('\\', '_')
    safe_finding = finding_label.replace('/', '_').replace('\\', '_')
    out_path = out_dir / f"{safe_name}_{safe_finding}_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return str(out_path)


def compute_metrics(heatmap: np.ndarray, bbox, orig_size, threshold=0.5):
    H, W = heatmap.shape
    oh, ow = orig_size
    x, y, bw, bh = bbox
    xs = int(x * W / ow)
    ys = int(y * H / oh)
    ws = max(1, int(bw * W / ow))
    hs = max(1, int(bh * H / oh))
    xs = max(0, min(xs, W - 1))
    ys = max(0, min(ys, H - 1))
    ws = min(ws, W - xs)
    hs = min(hs, H - ys)

    mask = np.zeros((H, W), dtype=bool)
    mask[ys:ys + hs, xs:xs + ws] = True

    pred = (heatmap > threshold)
    tp = np.logical_and(pred, mask).sum()
    fp = np.logical_and(pred, ~mask).sum()
    fn = np.logical_and(~pred, mask).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    inter = np.logical_and(pred, mask).sum()
    union = np.logical_or(pred, mask).sum()
    iou = inter / union if union > 0 else 0.0
    pct = pred.sum() / (H * W)
    try:
        num_labels, _ = cv2.connectedComponents(pred.astype(np.uint8), connectivity=8)
        num_regions = max(0, int(num_labels) - 1)
    except Exception:
        num_regions = 0
    return {
        'precision': precision,
        'recall': recall,
        'iou': iou,
        'pct_highlighted': pct,
        'num_regions': num_regions,
    }


def get_pathology_index(finding_label: str, model_labels) -> int:
    fl = finding_label.lower().strip()
    for i, name in enumerate(model_labels):
        if fl == name.lower().strip():
            return i
    return None


def resolve_image_path(root: Path, image_name: str) -> Path | None:
    cand = root / image_name
    if cand.exists():
        return cand
    lower = image_name.lower()
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        for p in root.rglob(ext):
            if p.name.lower() == lower:
                return p
    return None


def main():
    parser = argparse.ArgumentParser(description='Explain 10 random NIH images with minimal method and baselines')
    parser.add_argument('--dataset-path', type=str, default='med_img', help='Path to NIH Chest X-ray images directory (default: med_img)')
    parser.add_argument('--bbox-csv', type=str, default='BBox_List_2017.csv', help='Path to bounding box CSV')
    parser.add_argument('--random-samples', type=int, default=10, help='Number of random images to process')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold for metrics')
    parser.add_argument('--save-visualizations', action='store_true', help='Save side-by-side visualizations')
    parser.add_argument('--viz-output-dir', type=str, default='local_viz', help='Directory to save visuals')
    args = parser.parse_args()

    print('Loading chest X-ray model from torchxrayvision...')
    model = xrv.models.DenseNet(weights='densenet121-res224-all')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = True
    labels = model.pathologies

    tl_min = model.features[-1]
    tl = None
    for m in reversed(list(model.features.modules())):
        if isinstance(m, torch.nn.Conv2d):
            tl = m
            break
    if tl is None:
        tl = tl_min
    print(f'Target layer (minimal): {type(tl_min).__name__}  |  Target layer (CAM): {type(tl).__name__}')

    explainer_min = DDCAMExplainer(model, tl_min, device=device, is_vit=False)

    cam_classes = {
        'GradCAM': GradCAM,
        'GradCAM++': GradCAMPlusPlus,
        'ScoreCAM': ScoreCAM,
        'AblationCAM': AblationCAM,
        'XGradCAM': XGradCAM,
        'LayerCAM': LayerCAM,
    }

    df = pd.read_csv(args.bbox_csv)
    if len(df.columns) >= 6:
        colmap = {df.columns[0]: 'Image Index', df.columns[1]: 'Finding Label', df.columns[2]: 'x',
                  df.columns[3]: 'y', df.columns[4]: 'w', df.columns[5]: 'h'}
        df = df.rename(columns=colmap)
    df = df.dropna(subset=['x', 'y', 'w', 'h'])
    print(f'Loaded {len(df)} bounding box annotations')

    root = Path(args.dataset_path)
    candidates = []
    for _, r in df.iterrows():
        p = resolve_image_path(root, str(r['Image Index']))
        if p is not None:
            candidates.append((r, p))
    if not candidates:
        print('No matching images found in dataset path.')
        return
    random.shuffle(candidates)
    candidates = candidates[:max(1, int(args.random_samples))]

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 2048.0 - 1024.0),
    ])

    results = {'DDCAM': []}
    for m in cam_classes.keys():
        results[m] = []
    results['ReciproCAM'] = []
    results['EP'] = []
    results['RISE'] = []
    results['IG'] = []
    results['SHAP'] = []
    results['LIME'] = []

    print('\n' + '=' * 60)
    print('Processing images...')
    print('=' * 60)

    for (row, img_path) in tqdm(candidates, total=len(candidates), desc='Processing images'):
        image_name = str(row['Image Index'])
        finding = str(row['Finding Label'])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'Failed to open {img_path}: {e}')
            continue
        ow, oh = img.size
        bbox = [float(row['x']), float(row['y']), float(row['w']), float(row['h'])]

        x = preprocess(img).to(device)

        target_idx = get_pathology_index(finding, labels)
        if target_idx is None:
            with torch.no_grad():
                logits = model(x.unsqueeze(0))
                target_idx = int(torch.argmax(logits if logits.ndim == 1 else logits[0]).item())

        heat_ddcam, _ = None, None
        try:
            heat_ddcam, _ = explainer_min.explain(x, target_class=None)
        except Exception:
            heat_ddcam = None
        if isinstance(heat_ddcam, np.ndarray) and heat_ddcam.size > 0:
            try:
                m = compute_metrics(heat_ddcam, bbox, (oh, ow), threshold=args.threshold)
                m.update({'finding': finding, 'image': image_name})
                results['DDCAM'].append(m)
            except Exception:
                pass

        baseline_heatmaps = {}
        for method_name, CamClass in cam_classes.items():
            heat = None
            try:
                cam = CamClass(model=model, target_layers=[tl])
                if method_name in ['ScoreCAM', 'AblationCAM'] and hasattr(cam, 'batch_size'):
                    cam.batch_size = 32

                xt = x.unsqueeze(0).clone(); xt.requires_grad_(True)
                tg = [ClassifierOutputTarget(target_idx)]

                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                try:
                    with cam:
                        cam_out = cam(input_tensor=xt, targets=tg, aug_smooth=False, eigen_smooth=False)
                finally:
                    sys.stdout = old_stdout
                if isinstance(cam_out, torch.Tensor):
                    cam_out = cam_out.cpu().numpy()
                if isinstance(cam_out, np.ndarray) and cam_out.size > 0:
                    heat = cam_out[0] if cam_out.ndim == 3 else cam_out
            except Exception:
                heat = None
            baseline_heatmaps[method_name] = heat
            if isinstance(heat, np.ndarray) and heat.size > 0:
                try:
                    m = compute_metrics(heat, bbox, (oh, ow), threshold=args.threshold)
                    m.update({'finding': finding, 'image': image_name})
                    results[method_name].append(m)
                except Exception:
                    pass

        heat_reciprocam = None
        try:
            reciprocam_explainer = ReciproCam(model, device=device, target_layer_name='features')
            cam_output, _ = reciprocam_explainer(x.unsqueeze(0), index=target_idx)
            if isinstance(cam_output, torch.Tensor):
                cam_output = cam_output.cpu().numpy()
            if cam_output.shape != (224, 224):
                cam_output = cv2.resize(cam_output, (224, 224), interpolation=cv2.INTER_LINEAR)
            if cam_output.max() > cam_output.min():
                heat_reciprocam = (cam_output - cam_output.min()) / (cam_output.max() - cam_output.min())
            else:
                heat_reciprocam = cam_output
        except Exception:
            heat_reciprocam = None
        
        if isinstance(heat_reciprocam, np.ndarray) and heat_reciprocam.size > 0:
            try:
                m = compute_metrics(heat_reciprocam, bbox, (oh, ow), threshold=args.threshold)
                m.update({'finding': finding, 'image': image_name})
                results['ReciproCAM'].append(m)
            except Exception:
                pass

        # Extremal Perturbation
        heat_ep = None
        try:
            # EP needs the model with gradients enabled on params
            for p in model.parameters():
                p.requires_grad_(True)

            x_ep = x.unsqueeze(0)
            masks, _ = extremal_perturbation(
                model, x_ep, target_idx,
                areas=[0.1],
                perturbation=BLUR_PERTURBATION,
                max_iter=800,
                num_levels=8,
                step=7,
                sigma=21,
                jitter=True,
                variant=PRESERVE_VARIANT,
                reward_func=simple_reward,
                resize=False,
                smooth=0,
            )
            ep_cam = masks.sum(dim=0).squeeze(0).detach().cpu().numpy()
            ep_min, ep_max = ep_cam.min(), ep_cam.max()
            if ep_max - ep_min > 1e-8:
                heat_ep = (ep_cam - ep_min) / (ep_max - ep_min)
            else:
                heat_ep = np.zeros_like(ep_cam)
            if heat_ep.shape != (224, 224):
                heat_ep = cv2.resize(heat_ep, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:
            heat_ep = None

        if isinstance(heat_ep, np.ndarray) and heat_ep.size > 0:
            try:
                m = compute_metrics(heat_ep, bbox, (oh, ow), threshold=args.threshold)
                m.update({'finding': finding, 'image': image_name})
                results['EP'].append(m)
            except Exception:
                pass

        # RISE
        heat_rise = None
        try:
            with torch.no_grad():
                x_rise = x.unsqueeze(0)
                saliency = rise_class(model, x_rise, target=[target_idx],
                                      num_masks=1000, num_cells=7, batch_size=128)
                cam_rise = saliency[0, 0].detach().cpu().numpy()
                cam_min, cam_max = cam_rise.min(), cam_rise.max()
                if cam_max - cam_min > 1e-8:
                    heat_rise = (cam_rise - cam_min) / (cam_max - cam_min)
                else:
                    heat_rise = np.zeros_like(cam_rise)
                if heat_rise.shape != (224, 224):
                    heat_rise = cv2.resize(heat_rise, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:
            heat_rise = None

        if isinstance(heat_rise, np.ndarray) and heat_rise.size > 0:
            try:
                m = compute_metrics(heat_rise, bbox, (oh, ow), threshold=args.threshold)
                m.update({'finding': finding, 'image': image_name})
                results['RISE'].append(m)
            except Exception:
                pass

        # Integrated Gradients
        heat_ig = None
        try:
            x_ig = x.unsqueeze(0)
            x_ig.requires_grad_(True)
            baseline_ig = torch.zeros_like(x_ig)
            ig = IntegratedGradients(model)
            attributions = ig.attribute(x_ig, baselines=baseline_ig, target=target_idx, n_steps=50)
            cam_ig = attributions[0].abs().sum(dim=0).detach().cpu().numpy()
            cam_min, cam_max = cam_ig.min(), cam_ig.max()
            if cam_max - cam_min > 1e-8:
                heat_ig = (cam_ig - cam_min) / (cam_max - cam_min)
            else:
                heat_ig = np.zeros_like(cam_ig)
            if heat_ig.shape != (224, 224):
                heat_ig = cv2.resize(heat_ig, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:
            heat_ig = None

        if isinstance(heat_ig, np.ndarray) and heat_ig.size > 0:
            try:
                m = compute_metrics(heat_ig, bbox, (oh, ow), threshold=args.threshold)
                m.update({'finding': finding, 'image': image_name})
                results['IG'].append(m)
            except Exception:
                pass

        # GradientSHAP
        heat_shap = None
        try:
            x_shap = x.unsqueeze(0)
            x_shap.requires_grad_(True)
            baselines_shap = torch.cat([
                torch.zeros_like(x_shap),
                torch.randn_like(x_shap) * 0.1,
            ], dim=0)
            gs = GradientShap(model)
            attributions = gs.attribute(x_shap, baselines=baselines_shap, target=target_idx,
                                        n_samples=50, stdevs=0.0001)
            cam_shap = attributions[0].abs().sum(dim=0).detach().cpu().numpy()
            cam_min, cam_max = cam_shap.min(), cam_shap.max()
            if cam_max - cam_min > 1e-8:
                heat_shap = (cam_shap - cam_min) / (cam_max - cam_min)
            else:
                heat_shap = np.zeros_like(cam_shap)
            if heat_shap.shape != (224, 224):
                heat_shap = cv2.resize(heat_shap, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:
            heat_shap = None

        if isinstance(heat_shap, np.ndarray) and heat_shap.size > 0:
            try:
                m = compute_metrics(heat_shap, bbox, (oh, ow), threshold=args.threshold)
                m.update({'finding': finding, 'image': image_name})
                results['SHAP'].append(m)
            except Exception:
                pass

        # LIME
        heat_lime = None
        try:
            def batch_predict_lime(images):
                """images: list of numpy arrays (H, W, C) in [0, 1]."""
                batch = []
                for img_arr in images:
                    # Convert RGB to grayscale (LIME provides RGB)
                    gray = np.mean(img_arr, axis=2) if img_arr.ndim == 3 else img_arr
                    t = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0)  # 1×H×W
                    # Apply medical preprocessing: x * 2048.0 - 1024.0
                    t = t * 2048.0 - 1024.0
                    batch.append(t)
                batch = torch.stack(batch).to(device)
                with torch.no_grad():
                    logits = model(batch)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                return probs.cpu().numpy()

            # Prepare grayscale image in [0, 1] as 3-channel for LIME
            img_gray = np.array(img.convert('L').resize((224, 224))).astype(np.float32) / 255.0
            img_rgb_for_lime = np.stack([img_gray] * 3, axis=-1)

            lime_explainer = lime.lime_image.LimeImageExplainer()
            explanation = lime_explainer.explain_instance(
                img_rgb_for_lime,
                batch_predict_lime,
                labels=(target_idx,),
                hide_color=0,
                num_samples=1000,
            )
            local_exp = explanation.local_exp[target_idx]
            segments = explanation.segments
            cam_lime = np.zeros(segments.shape, dtype=np.float32)
            for seg_id, weight in local_exp:
                cam_lime[segments == seg_id] = weight
            cam_lime = np.abs(cam_lime)
            cam_min, cam_max = cam_lime.min(), cam_lime.max()
            if cam_max - cam_min > 1e-8:
                heat_lime = (cam_lime - cam_min) / (cam_max - cam_min)
            else:
                heat_lime = np.zeros_like(cam_lime)
            if heat_lime.shape != (224, 224):
                heat_lime = cv2.resize(heat_lime, (224, 224), interpolation=cv2.INTER_LINEAR)
        except Exception:
            heat_lime = None

        if isinstance(heat_lime, np.ndarray) and heat_lime.size > 0:
            try:
                m = compute_metrics(heat_lime, bbox, (oh, ow), threshold=args.threshold)
                m.update({'finding': finding, 'image': image_name})
                results['LIME'].append(m)
            except Exception:
                pass

        extra_heatmaps = {
            'RISE': heat_rise,
            'IG': heat_ig,
            'SHAP': heat_shap,
            'LIME': heat_lime,
        }

        if args.save_visualizations:
            try:
                visualize(img, bbox, finding, heat_ddcam, baseline_heatmaps, heat_reciprocam,
                          Path(args.viz_output_dir), image_name, heat_ep=heat_ep,
                          extra_heatmaps=extra_heatmaps)
            except Exception:
                pass

    for k, vals in results.items():
        if not vals:
            print(f'\n{k}: No results')
            continue
        dfk = pd.DataFrame(vals)
        print(f"\n{k}:")
        print(f"  Precision:     {dfk['precision'].mean():.4f} ± {dfk['precision'].std():.4f}")
        print(f"  Recall:        {dfk['recall'].mean():.4f} ± {dfk['recall'].std():.4f}")
        print(f"  IoU:           {dfk['iou'].mean():.4f} ± {dfk['iou'].std():.4f}")
        print(f"  % Highlighted: {dfk['pct_highlighted'].mean()*100:.2f}% ± {dfk['pct_highlighted'].std()*100:.2f}%")
        if 'num_regions' in dfk:
            print(f"  Num regions:   {dfk['num_regions'].mean():.2f} ± {dfk['num_regions'].std():.2f}")
            print(f"  Samples:       {len(dfk)}")

    rows = []
    for method, vals in results.items():
        for r in vals:
            rr = dict(r)
            rr['method'] = method
            rows.append(rr)
    if rows:
        pd.DataFrame(rows).to_csv('minimality_localization_detailed_results.csv', index=False)
        print('\nSaved: minimality_localization_detailed_results.csv')

    try:
        import scipy.stats as stats
        ddcam_df = pd.DataFrame(results.get('DDCAM', []))
        gc_df = pd.DataFrame(results.get('GradCAM', []))
        if not ddcam_df.empty and not gc_df.empty and 'image' in ddcam_df and 'image' in gc_df:
            ddcam_by = {r['image']: r for _, r in ddcam_df.iterrows()}
            gc_by = {r['image']: r for _, r in gc_df.iterrows()}
            common = sorted(set(ddcam_by.keys()) & set(gc_by.keys()))
            if len(common) >= 2:
                a = [ddcam_by[i]['iou'] for i in common]
                b = [gc_by[i]['iou'] for i in common]
                t, p = stats.ttest_rel(a, b)
                print("\nPaired t-test (DDCAM vs GradCAM) on IoU:")
                print(f"  n={len(common)}, t={t:.3f}, p={p:.4f}")
    except Exception:
        pass


if __name__ == '__main__':
    main()