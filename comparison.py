import os
import glob
import csv
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from torchvision import transforms
from torchvision.models import (
    vgg11, VGG11_Weights, 
    vgg16, VGG16_Weights, 
    vgg19, VGG19_Weights, 
    vit_b_16, ViT_B_16_Weights,
    resnet50, ResNet50_Weights,
    vit_l_16, ViT_L_16_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    inception_v3, Inception_V3_Weights,
    )

from pytorch_grad_cam import (
    GradCAM,
    HiResCAM,
    ScoreCAM,
    GradCAMPlusPlus,
    AblationCAM,
    XGradCAM,
    LayerCAM,
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget, ClassifierOutputReST
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.metrics.road import (
    ROADCombined,
    ROADMostRelevantFirstAverage,
    ROADLeastRelevantFirstAverage,
)
from pytorch_grad_cam.metrics.ADCC import ADCC
from pytorch_grad_cam.ablation_layer import AblationLayerVit

from ddcam import DDCAMExplainer
from reciprocam import ReciproCam, ViTReciproCam
from torchray.attribution.extremal_perturbation import extremal_perturbation, PRESERVE_VARIANT, BLUR_PERTURBATION, simple_reward
from torchray.attribution.rise import rise_class
from captum.attr import IntegratedGradients, GradientShap
import lime.lime_image


def load_image_as_tensor(image_path: str, device: str, image_size: int = 224):
    img = Image.open(image_path).convert("RGB")
    resize_size = int(image_size * 256 / 224)
    preprocess = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img).to(device)

    vis_img = Image.open(image_path).convert("RGB")
    vis_img = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])(vis_img)
    rgb_img = vis_img.numpy().transpose(1, 2, 0)
    rgb_img = np.clip(rgb_img, 0.0, 1.0)

    return input_tensor, rgb_img


def get_top1_class(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1)
        top1 = int(torch.argmax(probs, dim=1).item())
    return top1


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _vgg_conv_layers(model):
    """Extract all Conv2d layers from VGG-like models."""
    convs = []
    if hasattr(model, 'features'):
        for m in model.features:
            if isinstance(m, torch.nn.Conv2d):
                convs.append(m)
    return convs


def run_cam_methods(model: torch.nn.Module, target_layers, input_tensor: torch.Tensor, rgb_img: np.ndarray, target_class: int, reshape_transform=None, allowed_methods: list = None, is_vit: bool = False):
    cam_results = {}

    cam_classes = {
        "GradCAM": GradCAM,

        "ScoreCAM": ScoreCAM,
        "GradCAM++": GradCAMPlusPlus,
        "AblationCAM": AblationCAM,
        "XGradCAM": XGradCAM,

        "LayerCAM": LayerCAM,

    }

    iterable = cam_classes.items() if not allowed_methods else [(n, cam_classes[n]) for n in allowed_methods if n in cam_classes]

    for name, CamClass in iterable:
        try:
            if name == "AblationCAM" and is_vit:
                cam = CamClass(model=model, target_layers=target_layers, reshape_transform=reshape_transform, ablation_layer=AblationLayerVit())
            else:
                cam = CamClass(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
            
            with cam:
                if name in ["ScoreCAM", "AblationCAM"]:
                    cam.batch_size = 32
                
                if name == "ScoreCAM":
                    model.eval()
                
                t0 = time.perf_counter()
                grayscale_cam = cam(
                    input_tensor=input_tensor.unsqueeze(0),
                    targets=[ClassifierOutputTarget(target_class)]
                )
                elapsed = time.perf_counter() - t0
                grayscale_cam = grayscale_cam[0, :]
                
                if name == "ScoreCAM":
                    if np.isnan(grayscale_cam).any() or np.isinf(grayscale_cam).any():
                        print(f"[WARN] {name} produced NaN or Inf values")
                    cam_range = float(grayscale_cam.max() - grayscale_cam.min())
                    cam_std = float(np.std(grayscale_cam))
                    if cam_range < 1e-6 or cam_std < 1e-6:
                        print(f"[WARN] {name} produced near-constant heatmap (range={cam_range:.3e}, std={cam_std:.3e}). Trying fallback layer...")
                        convs = _vgg_conv_layers(model)
                        if len(convs) >= 2:
                            fallback_layer = convs[-2]
                            try:
                                cam_fb = CamClass(model=model, target_layers=[fallback_layer], reshape_transform=reshape_transform)
                                with cam_fb:
                                    if hasattr(cam, "batch_size"):
                                        cam_fb.batch_size = cam.batch_size
                                    model.eval()
                                    fb = cam_fb(
                                        input_tensor=input_tensor.unsqueeze(0),
                                        targets=[ClassifierOutputTarget(target_class)]
                                    )[0, :]
                                fb_range = float(fb.max() - fb.min())
                                fb_std = float(np.std(fb))
                                if fb_range >= 1e-6 and fb_std >= 1e-6:
                                    grayscale_cam = fb
                                    print("[INFO] ScoreCAM fallback layer improved heatmap variability.")
                                else:
                                    print(f"[WARN] ScoreCAM fallback also near-constant (range={fb_range:.3e}, std={fb_std:.3e}).")
                            except Exception as e_fb:
                                print(f"[WARN] ScoreCAM fallback failed: {e_fb}")
                
                vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                cam_results[name] = {
                    "grayscale": grayscale_cam,
                    "vis": vis,
                    "time": elapsed,
                    "cam_class": CamClass,  # Store class to recreate CAM object
                    "cam_config": {
                        "name": name,
                        "is_vit": is_vit,
                        "target_layers": target_layers,
                        "reshape_transform": reshape_transform,
                    }
                }
        except Exception as e:
            cam_results[name] = None
            print(f"[WARN] {name} failed: {e}")

    return cam_results


def run_ddcam_explainer(model: torch.nn.Module, target_layer, input_tensor: torch.Tensor, rgb_img: np.ndarray, 
                           target_class: int, device: str, is_vit: bool = False, verbose: bool = False):
    """
    Run DDCAM explainer (combines individual feature testing and DD-CAM).
    Uses individual feature testing mode for non-interacting features, DD mode for interacting features.
    Explains the target class using relative minimal sufficiency.
    """
    try:
        explainer = DDCAMExplainer(
            model, target_layer, device=device, is_vit=is_vit,
            apply_vit_smoothing=True,  # Apply 4x4 smoothing for ViT
            strategy="auto"  # Auto-detect non-interacting vs interacting feature testing based on feature interactions
        )
        
        t0 = time.perf_counter()
        
        heatmap, overlay = explainer.explain(
            input_tensor,
            target_class=target_class,
            retention_ratio=1.0,
            min_absolute_prob=0.01
        )
        
        elapsed = time.perf_counter() - t0
        
        if heatmap is None:
            return None, None, float('nan')
        
        return heatmap, overlay, elapsed
    except Exception as e:
        print(f"[WARN] DDCAM explainer failed: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None, float('nan')


def run_reciprocam(model: torch.nn.Module, input_tensor: torch.Tensor, rgb_img: np.ndarray, 
                   target_class: int, device: str, target_layer_name: str = None, model_name: str = None):
    """
    Run ReciproCAM for CNN models.
    Note: ReciproCAM requires 4D feature maps from convolutional layers.
    """
    try:
        if target_layer_name is None:
            if model_name == 'inception_v3':
                target_layer_name = 'Mixed_7c'
            elif hasattr(model, 'features'):
                target_layer_name = 'features'
            elif hasattr(model, 'layer4'):
                target_layer_name = 'layer4'
            elif model_name and 'vgg' in model_name.lower():
                target_layer_name = 'features'
            elif model_name and 'efficientnet' in model_name.lower():
                target_layer_name = 'features'
            else:
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        parts = name.split('.')
                        if len(parts) > 1:
                            target_layer_name = parts[0]
                        break
        
        expl = ReciproCam(model, device=device, target_layer_name=target_layer_name)
        
        t0 = time.perf_counter()
        cam, pred_class = expl(input_tensor.unsqueeze(0), index=target_class)
        elapsed = time.perf_counter() - t0
        
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
        
        if cam.shape != rgb_img.shape[:2]:
            from scipy.ndimage import zoom
            zoom_factors = (rgb_img.shape[0] / cam.shape[0], rgb_img.shape[1] / cam.shape[1])
            cam = zoom(cam, zoom_factors, order=1)
        
        # Normalize to [0, 1] if needed
        if cam.max() > 1.0 or cam.min() < 0.0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
        return cam, vis, elapsed
    except ValueError as e:
        if "4D feature maps" in str(e):
            print(f"[WARN] ReciproCAM skipped: Requires 4D conv features, but hook captured unsuitable output")
        else:
            print(f"[WARN] ReciproCAM failed: {e}")
        return None, None, float('nan')
    except Exception as e:
        print(f"[WARN] ReciproCAM failed: {e}")
        return None, None, float('nan')


def run_reciprocam_vit(model: torch.nn.Module, input_tensor: torch.Tensor, rgb_img: np.ndarray,
                       target_class: int, device: str, block_index: int = -2):
    """
    Run ViTReciproCAM for Vision Transformer models.
    """
    try:
        expl = ViTReciproCam(model, device=device, is_gaussian=True, block_index=block_index, cls_token=True)
        t0 = time.perf_counter()
        cam, pred_class = expl(input_tensor.unsqueeze(0), index=target_class)
        elapsed = time.perf_counter() - t0
        
        if isinstance(cam, torch.Tensor):
            cam = cam.cpu().numpy()
        
        if cam.shape != rgb_img.shape[:2]:
            from scipy.ndimage import zoom
            zoom_factors = (rgb_img.shape[0] / cam.shape[0], rgb_img.shape[1] / cam.shape[1])
            cam = zoom(cam, zoom_factors, order=1)
        
        # Normalize to [0, 1] if needed
        if cam.max() > 1.0 or cam.min() < 0.0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
        return cam, vis, elapsed
    except Exception as e:
        print(f"[WARN] ViTReciproCAM failed: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None, float('nan')


def run_extremal_perturbation(model: torch.nn.Module, input_tensor: torch.Tensor, rgb_img: np.ndarray,
                               target_class: int, device: str, areas=None, max_iter=800):
    """
    Run Extremal Perturbation (Fong et al., ICCV 2019) via TorchRay.
    Returns (grayscale_cam, vis, elapsed_time).
    """
    if areas is None:
        areas = [0.1]
    try:
        # EP needs requires_grad=True on parameters for some internals,
        # but it disables grad itself. Ensure model is in eval mode.
        model.eval()

        # EP expects 4D input: 1×C×H×W
        x = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        x = x.to(device)

        t0 = time.perf_counter()
        masks, _ = extremal_perturbation(
            model, x, target_class,
            areas=areas,
            perturbation=BLUR_PERTURBATION,
            max_iter=max_iter,
            num_levels=8,
            step=7,
            sigma=21,
            jitter=True,
            variant=PRESERVE_VARIANT,
            reward_func=simple_reward,
            resize=False,
            smooth=0,
        )
        elapsed = time.perf_counter() - t0

        # masks shape: K×1×H×W (K = len(areas))
        # Sum across areas and squeeze to get a single 2D heatmap
        cam = masks.sum(dim=0).squeeze(0).detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to match input image if needed
        if cam.shape != rgb_img.shape[:2]:
            cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        vis = show_cam_on_image(rgb_img, cam, use_rgb=True)

        # Re-enable gradients for model params (EP disables them)
        for p in model.parameters():
            p.requires_grad_(True)

        return cam, vis, elapsed
    except Exception as e:
        print(f"[WARN] Extremal Perturbation failed: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        # Re-enable gradients in case of failure too
        for p in model.parameters():
            p.requires_grad_(True)
        return None, None, float('nan')


def run_rise(model: torch.nn.Module, input_tensor: torch.Tensor, rgb_img: np.ndarray,
             target_class: int, device: str, num_masks=1000, num_cells=7, batch_size=128):
    """
    Run RISE (Randomized Input Sampling for Explanation) via TorchRay.
    Returns (grayscale_cam, vis, elapsed_time).
    """
    try:
        model.eval()
        x = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor
        x = x.to(device)

        t0 = time.perf_counter()
        saliency = rise_class(model, x, target=[target_class],
                              num_masks=num_masks, num_cells=num_cells,
                              batch_size=batch_size)
        elapsed = time.perf_counter() - t0

        # saliency shape: 1×1×H×W
        cam = saliency[0, 0].detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if cam.shape != rgb_img.shape[:2]:
            cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
        return cam, vis, elapsed
    except Exception as e:
        print(f"[WARN] RISE failed: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None, float('nan')


def run_integrated_gradients(model: torch.nn.Module, input_tensor: torch.Tensor, rgb_img: np.ndarray,
                             target_class: int, device: str, n_steps=50):
    """
    Run Integrated Gradients via Captum.
    Returns (grayscale_cam, vis, elapsed_time).
    """
    try:
        model.eval()
        x = input_tensor.unsqueeze(0).to(device)
        baseline = torch.zeros_like(x)

        ig = IntegratedGradients(model)

        t0 = time.perf_counter()
        attributions = ig.attribute(x, baselines=baseline, target=target_class, n_steps=n_steps)
        elapsed = time.perf_counter() - t0

        # attributions shape: 1×C×H×W → sum abs across channels for 2D heatmap
        cam = attributions[0].abs().sum(dim=0).detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if cam.shape != rgb_img.shape[:2]:
            cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
        return cam, vis, elapsed
    except Exception as e:
        print(f"[WARN] Integrated Gradients failed: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None, float('nan')


def run_gradient_shap(model: torch.nn.Module, input_tensor: torch.Tensor, rgb_img: np.ndarray,
                      target_class: int, device: str, n_samples=50, stdevs=0.0001):
    """
    Run GradientSHAP via Captum.
    Returns (grayscale_cam, vis, elapsed_time).
    """
    try:
        model.eval()
        x = input_tensor.unsqueeze(0).to(device)
        # Use a distribution of baselines (zeros + random noise)
        baselines = torch.cat([
            torch.zeros_like(x),
            torch.randn_like(x) * 0.1,
        ], dim=0)

        gs = GradientShap(model)

        t0 = time.perf_counter()
        attributions = gs.attribute(x, baselines=baselines, target=target_class,
                                    n_samples=n_samples, stdevs=stdevs)
        elapsed = time.perf_counter() - t0

        # attributions shape: 1×C×H×W → sum abs across channels
        cam = attributions[0].abs().sum(dim=0).detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if cam.shape != rgb_img.shape[:2]:
            cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
        return cam, vis, elapsed
    except Exception as e:
        print(f"[WARN] GradientSHAP failed: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None, float('nan')


def run_lime_explanation(model: torch.nn.Module, input_tensor: torch.Tensor, rgb_img: np.ndarray,
                         target_class: int, device: str, num_samples=1000):
    """
    Run LIME (Local Interpretable Model-agnostic Explanations).
    Returns (grayscale_cam, vis, elapsed_time).
    """
    try:
        model.eval()

        # LIME needs a batch prediction function operating on numpy images
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        def batch_predict(images):
            """images: list of numpy arrays (H, W, C) in [0, 1]."""
            batch = []
            for img in images:
                t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
                t = transforms.Normalize(mean=mean.tolist(), std=std.tolist())(t)
                batch.append(t)
            batch = torch.stack(batch).to(device)
            with torch.no_grad():
                logits = model(batch)
                probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.cpu().numpy()

        explainer = lime.lime_image.LimeImageExplainer()

        t0 = time.perf_counter()
        explanation = explainer.explain_instance(
            rgb_img,
            batch_predict,
            labels=(target_class,),
            hide_color=0,
            num_samples=num_samples,
        )
        elapsed = time.perf_counter() - t0

        # Get heatmap for target class
        local_exp = explanation.local_exp[target_class]
        segments = explanation.segments

        # Build heatmap from superpixel weights
        cam = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, weight in local_exp:
            cam[segments == seg_id] = weight

        # Use absolute values for importance, then normalize to [0, 1]
        cam = np.abs(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        if cam.shape != rgb_img.shape[:2]:
            cam = cv2.resize(cam, (rgb_img.shape[1], rgb_img.shape[0]), interpolation=cv2.INTER_LINEAR)

        vis = show_cam_on_image(rgb_img, cam, use_rgb=True)
        return cam, vis, elapsed
    except Exception as e:
        print(f"[WARN] LIME failed: {e}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        return None, None, float('nan')


def save_individual_images(image_name: str, model_name: str, rgb_img: np.ndarray,
                           cam_results: dict, ddcam_gray: np.ndarray, ddcam_vis: np.ndarray,
                           reciprocam_gray: np.ndarray, reciprocam_vis: np.ndarray,
                           ep_gray: np.ndarray, ep_vis: np.ndarray,
                           extra_methods: dict, out_dir: str):
    """
    Save each method's overlay and grayscale heatmap as individual images.
    Structure: {out_dir}/individual/{image_name}/{model_name}/{method}.png
               {out_dir}/individual/{image_name}/{model_name}/{method}_gray.npy
    """
    base = os.path.splitext(image_name)[0]
    ind_dir = os.path.join(out_dir, "individual", base, model_name)
    os.makedirs(ind_dir, exist_ok=True)

    # Save original image
    Image.fromarray((rgb_img * 255).astype(np.uint8) if rgb_img.max() <= 1.0 else rgb_img.astype(np.uint8)).save(
        os.path.join(ind_dir, "original.png"))

    # Helper to save a single method
    def _save(name, gray, vis):
        if vis is not None:
            Image.fromarray(vis).save(os.path.join(ind_dir, f"{name}.png"))
        if gray is not None:
            np.save(os.path.join(ind_dir, f"{name}_gray.npy"), gray)

    # DDCAM
    _save("DDCAM", ddcam_gray, ddcam_vis)

    # 6 CAM baselines
    for method_name, res in cam_results.items():
        if res is None:
            continue
        _save(method_name, res.get("grayscale"), res.get("visualization"))

    # ReciproCAM
    _save("ReciproCAM", reciprocam_gray, reciprocam_vis)

    # EP
    _save("EP", ep_gray, ep_vis)

    # RISE, IG, SHAP, LIME
    if extra_methods:
        for method_name, data in extra_methods.items():
            _save(method_name, data.get("gray"), data.get("vis"))


def compute_cam_scores(model: torch.nn.Module, input_tensor: torch.Tensor, grayscale_cam: np.ndarray, target_class: int):
    """Compute three ROAD-family scores: combined, deletion (MoRF avg), insertion (LeRF avg).

    Returns a dict with keys 'ROAD', 'ROAD_Del', 'ROAD_Ins'.
    Higher is better for all three (LeRF: keeping the important pixels and removing the
    least relevant ones should preserve the prediction; MoRF subtracts the most-relevant-first
    erasure score, so a more faithful explanation yields a higher combined value).
    """
    input_batch = input_tensor.unsqueeze(0)
    cams_batch = grayscale_cam[np.newaxis, ...]
    targets_logits = [ClassifierOutputTarget(target_class)]
    pcts = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    out = {'ROAD': float('nan'), 'ROAD_Del': float('nan'), 'ROAD_Ins': float('nan')}
    try:
        out['ROAD'] = float(np.array(
            ROADCombined(percentiles=pcts)(input_batch, cams_batch, targets_logits, model)
        ).reshape(-1)[0])
    except Exception as e:
        print(f"[WARN] ROADCombined failed: {e}")
    try:
        out['ROAD_Del'] = float(np.array(
            ROADMostRelevantFirstAverage(percentiles=pcts)(input_batch, cams_batch, targets_logits, model)
        ).reshape(-1)[0])
    except Exception as e:
        print(f"[WARN] ROAD_Del failed: {e}")
    try:
        out['ROAD_Ins'] = float(np.array(
            ROADLeastRelevantFirstAverage(percentiles=pcts)(input_batch, cams_batch, targets_logits, model)
        ).reshape(-1)[0])
    except Exception as e:
        print(f"[WARN] ROAD_Ins failed: {e}")

    return out


def format_model_name(model_name: str) -> str:
    """Format model name for display."""
    model_name_map = {
        'resnet50': 'ResNet-50',
        'efficientnet_b0': 'EfficientNet-B0',
        'inception_v3': 'Inception-v3',
        'vgg11': 'VGG-11',
        'vgg16': 'VGG-16',
        'vgg19': 'VGG-19',
        'vit_b_16': 'ViT-B/16',
        'vit_l_16': 'ViT-L/16',
    }
    return model_name_map.get(model_name, model_name)


def plot_and_save(image_path: str, rgb_img: np.ndarray, cam_results: dict,
                  ddcam_gray: np.ndarray, ddcam_vis: np.ndarray,
                  reciprocam_gray: np.ndarray, reciprocam_vis: np.ndarray,
                  scores_map: dict, out_dir: str, model_name: str,
                  ep_gray: np.ndarray = None, ep_vis: np.ndarray = None,
                  extra_methods: dict = None):
    """
    extra_methods: dict of {"MethodName": {"gray": np.ndarray, "vis": np.ndarray}} for RISE, IG, SHAP, LIME etc.
    """
    methods = ["GradCAM", "GradCAM++", "XGradCAM", "LayerCAM", "ScoreCAM", "AblationCAM"]
    if extra_methods is None:
        extra_methods = {}

    # +1 original, +1 DDCAM, +len(methods) baselines, +1 ReciproCAM, +1 EP, +len(extra_methods)
    n_panels = 1 + 1 + len(methods) + 1 + 1 + len(extra_methods)
    n_cols = n_panels
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, 4))
    if n_cols == 1:
        axes = [axes]
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes

    axes[0].imshow(rgb_img)
    axes[0].set_title("Original", fontsize=14)
    display_name = format_model_name(model_name)
    axes[0].text(0.5, 0.02, display_name, fontsize=16, fontweight='bold', 
                 ha='center', va='bottom', transform=axes[0].transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5))
    axes[0].axis('off')

    idx = 1
    if ddcam_vis is not None:
        s = scores_map.get("DDCAM", {})
        if 'adcc' in s:
            ddcam_scores = [s.get('adcc', 0), s.get('drop', 0), s.get('coherency', 0),
                                s.get('complexity', 0), s.get('inc', 0), s.get('dropindeletion', 0)]
            ddcam_vis_with_scores = visualize_score(ddcam_vis.copy(), 'DDCAM', ddcam_scores)
            axes[idx].imshow(ddcam_vis_with_scores)
        else:
            axes[idx].imshow(ddcam_vis)
        axes[idx].set_title("DDCAM", fontsize=14)
    else:
        axes[idx].imshow(rgb_img)
        axes[idx].set_title("DDCAM (failed)", fontsize=14)
    axes[idx].axis('off')
    idx += 1

    for m in methods:
        res = cam_results.get(m)
        if res is not None:
            vis = res.get("vis")
            s = scores_map.get(m, {})
            
            if 'adcc' in s:
                method_scores = [s.get('adcc', 0), s.get('drop', 0), s.get('coherency', 0),
                               s.get('complexity', 0), s.get('inc', 0), s.get('dropindeletion', 0)]
                vis_with_scores = visualize_score(vis.copy(), m, method_scores)
                axes[idx].imshow(vis_with_scores)
            else:
                axes[idx].imshow(vis)
            
            axes[idx].set_title(m, fontsize=14)
        else:
            axes[idx].imshow(rgb_img)
            axes[idx].set_title(f"{m} (failed)", fontsize=14)
        axes[idx].axis('off')
        idx += 1
    
    if reciprocam_vis is not None:
        s = scores_map.get("ReciproCAM", {})
        
        if 'adcc' in s:
            reciprocam_scores = [s.get('adcc', 0), s.get('drop', 0), s.get('coherency', 0),
                                s.get('complexity', 0), s.get('inc', 0), s.get('dropindeletion', 0)]
            reciprocam_vis_with_scores = visualize_score(reciprocam_vis.copy(), 'ReciproCAM', reciprocam_scores)
            axes[idx].imshow(reciprocam_vis_with_scores)
        else:
            axes[idx].imshow(reciprocam_vis)
        
        axes[idx].set_title("ReciproCAM", fontsize=14)
    else:
        axes[idx].imshow(rgb_img)
        axes[idx].set_title("ReciproCAM (failed)", fontsize=14)
    axes[idx].axis('off')
    idx += 1

    # Extremal Perturbation panel
    if ep_vis is not None:
        s = scores_map.get("EP", {})
        if 'adcc' in s:
            ep_scores = [s.get('adcc', 0), s.get('drop', 0), s.get('coherency', 0),
                         s.get('complexity', 0), s.get('inc', 0), s.get('dropindeletion', 0)]
            ep_vis_with_scores = visualize_score(ep_vis.copy(), 'EP', ep_scores)
            axes[idx].imshow(ep_vis_with_scores)
        else:
            axes[idx].imshow(ep_vis)
        axes[idx].set_title("EP", fontsize=14)
    else:
        axes[idx].imshow(rgb_img)
        axes[idx].set_title("EP (failed)", fontsize=14)
    axes[idx].axis('off')
    idx += 1

    # Extra method panels (RISE, IG, SHAP, LIME)
    for method_name, method_data in extra_methods.items():
        if idx >= len(axes):
            break
        m_vis = method_data.get("vis")
        if m_vis is not None:
            s = scores_map.get(method_name, {})
            if 'adcc' in s:
                m_scores = [s.get('adcc', 0), s.get('drop', 0), s.get('coherency', 0),
                            s.get('complexity', 0), s.get('inc', 0), s.get('dropindeletion', 0)]
                m_vis_with_scores = visualize_score(m_vis.copy(), method_name, m_scores)
                axes[idx].imshow(m_vis_with_scores)
            else:
                axes[idx].imshow(m_vis)
            axes[idx].set_title(method_name, fontsize=14)
        else:
            axes[idx].imshow(rgb_img)
            axes[idx].set_title(f"{method_name} (failed)", fontsize=14)
        axes[idx].axis('off')
        idx += 1

    for j in range(idx, len(axes)):
        axes[j].axis('off')

    base = os.path.splitext(os.path.basename(image_path))[0]
    out_path = os.path.join(out_dir, f"{base}_{model_name}_cam_comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def apply_gaussian_patch_smoothing(heatmap: np.ndarray, sigma: float = 1.0, mix: float = 0.4, gamma: float = 1.2) -> np.ndarray:
    """Apply 3x3 Gaussian smoothing to a patch-level heatmap for ViT."""
    from scipy.ndimage import gaussian_filter
    
    smoothed = gaussian_filter(heatmap, sigma=sigma, mode='constant', cval=0)
    blended = (1.0 - mix) * heatmap + mix * smoothed
    
    hmin, hmax = blended.min(), blended.max()
    if (hmax - hmin) > 1e-8:
        blended = (blended - hmin) / (hmax - hmin)
    blended = np.power(np.clip(blended, 0.0, 1.0), 1.0 / max(gamma, 1e-6))
    
    return blended


def visualize_score(visualization, name, scores):
    """Add score annotations to visualization."""
    y_pos = 15
    score_labels = ['ADCC', 'Drop', 'Coherency', 'Complexity', 'Inc', 'DropInDel']
    
    for label, value in zip(score_labels, scores):
        text = f"{label}: {value:.3f}"
        visualization = cv2.putText(visualization, text, (5, y_pos),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        y_pos += 15
        
    return visualization


def load_model(model_name: str, device: str):
    """Load a fresh model instance. Returns (model, target_layers, reshape_transform, is_vit, image_size)."""
    if model_name == "vgg11":
        model = vgg11(weights=VGG11_Weights.DEFAULT).to(device).eval()
        last_conv = None
        for m in reversed(model.features):
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
                break
        target_layers = [last_conv]
        reshape_transform = None
        is_vit = False
        image_size = 224
    elif model_name == "vgg16":
        model = vgg16(weights=VGG16_Weights.DEFAULT).to(device).eval()
        last_conv = None
        for m in reversed(model.features):
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
                break
        target_layers = [last_conv]
        reshape_transform = None
        is_vit = False
        image_size = 224
    elif model_name == "vgg19":
        model = vgg19(weights=VGG19_Weights.DEFAULT).to(device).eval()
        last_conv = None
        for m in reversed(model.features):
            if isinstance(m, torch.nn.Conv2d):
                last_conv = m
                break
        target_layers = [last_conv]
        reshape_transform = None
        is_vit = False
        image_size = 224
    elif model_name == "vit_b_16":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
        target_layers = [model.encoder.layers[-1].ln_1]
        reshape_transform = vit_reshape_transform
        is_vit = True
        image_size = 224
    elif model_name == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
        target_layers = [model.layer4]
        reshape_transform = None
        is_vit = False
        image_size = 224
    elif model_name == "vit_l_16":
        model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device).eval()
        target_layers = [model.encoder.layers[-1].ln_1]
        reshape_transform = vit_reshape_transform
        is_vit = True
        image_size = 224
    elif model_name == "vit_b_32":
        from torchvision.models import vit_b_32, ViT_B_32_Weights
        model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1).to(device).eval()
        # ViT-B/32 has a 7x7 token grid (224/32) instead of 14x14; use a matching reshape
        def _vit_b_32_reshape(tensor, height=7, width=7):
            return vit_reshape_transform(tensor, height=height, width=width)
        target_layers = [model.encoder.layers[-1].ln_1]
        reshape_transform = _vit_b_32_reshape
        is_vit = True
        image_size = 224
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).to(device).eval()
        target_layers = [model.features[-1]]
        reshape_transform = None
        is_vit = False
        image_size = 224
    elif model_name == "inception_v3":
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device).eval()
        target_layers = [model.Mixed_7c]
        reshape_transform = None
        is_vit = False
        image_size = 224
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model, target_layers, reshape_transform, is_vit, image_size


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default=os.path.join(os.path.dirname(__file__), "test_folder"),
                        help="Directory of images to run comparison on.")
    parser.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "test_img_result_comparison"),
                        help="Directory to write comparison outputs.")
    parser.add_argument("--models", default="resnet50,efficientnet_b0,inception_v3,vgg11,vgg16,vgg19,vit_b_16,vit_l_16",
                        help="Comma-separated list of model names.")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Cap on number of images to process (default: all in the directory).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    images_dir = args.images_dir
    out_dir = args.out_dir
    ensure_dir(out_dir)

    csv_rows = []

    image_paths = []
    image_paths += glob.glob(os.path.join(images_dir, "*.jpg"))
    image_paths += glob.glob(os.path.join(images_dir, "*.jpeg"))
    image_paths += glob.glob(os.path.join(images_dir, "*.JPEG"))
    image_paths += glob.glob(os.path.join(images_dir, "*.png"))
    image_paths = sorted(set(os.path.normcase(os.path.abspath(p)) for p in image_paths))
    if args.max_images:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        print(f"No images found in {images_dir}")
        return
    print(f"[config] device={device} models={model_names} images={len(image_paths)} out={out_dir}")

    for model_name in model_names:

        for idx, img_path in enumerate(image_paths):
            print(f"Processing [{model_name}]: {img_path}")

            model, target_layers, reshape_transform, is_vit, image_size = load_model(model_name, device)

            input_tensor, rgb_img = load_image_as_tensor(img_path, device, image_size=image_size)

            target_class = get_top1_class(model, input_tensor)

            cam_metric = ADCC()
            targets = [ClassifierOutputReST(target_class)]
            metric_targets = [ClassifierOutputSoftmaxTarget(target_class)]

            allowed = None
            if is_vit:
                allowed = ["GradCAM", "ScoreCAM", "GradCAM++", "AblationCAM", "XGradCAM", "LayerCAM"]
            cam_results = run_cam_methods(model, target_layers, input_tensor, rgb_img, target_class, reshape_transform, allowed_methods=allowed, is_vit=is_vit)

            verbose_ddcam = (idx == 0)
            ddcam_gray, ddcam_vis, ddcam_time = run_ddcam_explainer(
                model, target_layers[0], input_tensor, rgb_img, target_class, device, 
                is_vit=is_vit, verbose=verbose_ddcam
            )
            
            if is_vit:
                reciprocam_gray, reciprocam_vis, reciprocam_time = run_reciprocam_vit(
                    model, input_tensor, rgb_img, target_class, device, block_index=-2
                )
            else:
                reciprocam_gray, reciprocam_vis, reciprocam_time = run_reciprocam(
                    model, input_tensor, rgb_img, target_class, device, target_layer_name=None,                     model_name=model_name
                )

            # Run Extremal Perturbation
            ep_gray, ep_vis, ep_time = run_extremal_perturbation(
                model, input_tensor, rgb_img, target_class, device,
                areas=[0.1], max_iter=800
            )

            # Run RISE
            rise_gray, rise_vis, rise_time = run_rise(
                model, input_tensor, rgb_img, target_class, device,
                num_masks=1000, num_cells=7, batch_size=128
            )

            # Run Integrated Gradients
            ig_gray, ig_vis, ig_time = run_integrated_gradients(
                model, input_tensor, rgb_img, target_class, device, n_steps=50
            )

            # Run GradientSHAP
            shap_gray, shap_vis, shap_time = run_gradient_shap(
                model, input_tensor, rgb_img, target_class, device,
                n_samples=50, stdevs=0.0001
            )

            # Run LIME
            lime_gray, lime_vis, lime_time = run_lime_explanation(
                model, input_tensor, rgb_img, target_class, device,
                num_samples=1000
            )

            scores_map = {}
            first_gray = None
            base = os.path.splitext(os.path.basename(img_path))[0]
            
            for m, res in cam_results.items():
                if res is None:
                    continue
                gray = res.get("grayscale")
                if first_gray is None:
                    first_gray = gray
                road_dict = compute_cam_scores(model, input_tensor, gray, target_class)
                road = road_dict['ROAD']
                t = res.get("time", float('nan')) if res is not None else float('nan')
                
                try:
                    cam_class = res.get("cam_class")
                    cam_config = res.get("cam_config")
                    if cam_class is not None and cam_config is not None:
                        CamClass = cam_class
                        config = cam_config.copy()
                        target_cls = target_class
                        orig_gray_shape = gray.shape
                        
                        def cam_method_fn(x, t):
                            try:
                                model_device = next(model.parameters()).device
                            except StopIteration:
                                model_device = torch.device("cpu")
                            x = x.to(model_device)
                            if x.dim() == 3:
                                x = x.unsqueeze(0)
                            if config["name"] != "ScoreCAM":
                                x = x.requires_grad_(True)
                            else:
                                x = x.requires_grad_(False)

                            if config["name"] == "AblationCAM" and config["is_vit"]:
                                cam_obj = CamClass(
                                    model=model, 
                                    target_layers=config["target_layers"], 
                                    reshape_transform=config["reshape_transform"], 
                                    ablation_layer=AblationLayerVit()
                                )
                            else:
                                cam_obj = CamClass(
                                    model=model, 
                                    target_layers=config["target_layers"], 
                                    reshape_transform=config["reshape_transform"]
                                )
                            
                            with cam_obj:
                                if config["name"] == "ScoreCAM":
                                    if config["name"] in ["ScoreCAM", "AblationCAM"]:
                                        cam_obj.batch_size = 32
                                    model.eval()
                                    result = cam_obj(
                                        input_tensor=x,
                                        targets=t,
                                        aug_smooth=False,
                                        eigen_smooth=False,
                                    )
                                else:
                                    with torch.enable_grad():
                                        if hasattr(cam_obj, "model") and hasattr(cam_obj.model, "zero_grad"):
                                            try:
                                                cam_obj.model.zero_grad(set_to_none=True)
                                            except TypeError:
                                                cam_obj.model.zero_grad()
                                        if config["name"] in ["AblationCAM"]:
                                            cam_obj.batch_size = 32
                                        result = cam_obj(
                                            input_tensor=x,
                                            targets=t,
                                            aug_smooth=False,
                                            eigen_smooth=False,
                                        )
                                if isinstance(result, torch.Tensor):
                                    result = result.detach().cpu().numpy()
                                if len(result.shape) == 4:
                                    result = result[0]
                                elif len(result.shape) == 3 and result.shape[0] == 1:
                                    result = result[0]
                                elif len(result.shape) == 2:
                                    pass
                                else:
                                    while len(result.shape) > 2:
                                        result = result[0]
                                if result.shape != orig_gray_shape:
                                    from scipy.ndimage import zoom
                                    zoom_factors = (orig_gray_shape[0] / result.shape[0], orig_gray_shape[1] / result.shape[1])
                                    result = zoom(result, zoom_factors, order=1)
                                return torch.from_numpy(result).unsqueeze(0)
                    else:
                        cam_method_fn = lambda x, t: torch.from_numpy(gray).unsqueeze(0)
                    
                    adcc_val, drop_val, coh_val, com_val, inc_val, dropindel_val = cam_metric(
                        input_tensor.unsqueeze(0), gray[np.newaxis,...], targets, metric_targets, model,
                        cam_method_fn
                    )
                    scores_map[m] = {
                        "road": road, "time": t,
                        "adcc": adcc_val[0], "drop": drop_val[0], "coherency": coh_val[0],
                        "complexity": com_val[0], "inc": inc_val[0], "dropindeletion": dropindel_val[0]
                    }
                    
                    # Save to CSV
                    csv_rows.append({
                        'Image': base, 'Model': model_name, 'Method': m,
                        'ROAD': road, 'ROAD_Del': road_dict['ROAD_Del'], 'ROAD_Ins': road_dict['ROAD_Ins'],
                        'ADCC': adcc_val[0], 'Drop': drop_val[0],
                        'Coherency': coh_val[0], 'Complexity': com_val[0],
                        'Inc': inc_val[0], 'DropInDeletion': dropindel_val[0], 'Time': t
                    })
                except Exception as e:
                    print(f"[WARN] ADCC failed for {m}: {e}")
                    scores_map[m] = {"road": road, "time": t}

            if ddcam_gray is not None:
                road_dict = compute_cam_scores(model, input_tensor, ddcam_gray, target_class)
                road = road_dict['ROAD']
                
                try:
                    adcc_val, drop_val, coh_val, com_val, inc_val, dropindel_val = cam_metric(
                        input_tensor.unsqueeze(0), ddcam_gray[np.newaxis,...], targets, metric_targets, model,
                        lambda x, t: torch.from_numpy(ddcam_gray).unsqueeze(0)
                    )
                    scores_map["DDCAM"] = {
                        "road": road, "time": ddcam_time,
                        "adcc": adcc_val[0], "drop": drop_val[0], "coherency": coh_val[0],
                        "complexity": com_val[0], "inc": inc_val[0], "dropindeletion": dropindel_val[0]
                    }
                    
                    # Save to CSV
                    csv_rows.append({
                        'Image': base, 'Model': model_name, 'Method': 'DDCAM',
                        'ROAD': road, 'ROAD_Del': road_dict['ROAD_Del'], 'ROAD_Ins': road_dict['ROAD_Ins'],
                        'ADCC': adcc_val[0], 'Drop': drop_val[0],
                        'Coherency': coh_val[0], 'Complexity': com_val[0],
                        'Inc': inc_val[0], 'DropInDeletion': dropindel_val[0], 'Time': ddcam_time
                    })
                except Exception as e:
                    import traceback
                    print(f"[WARN] ADCC failed for DDCAM: {e}")
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    scores_map["DDCAM"] = {"road": road, "time": ddcam_time}
            
            if reciprocam_gray is not None:
                road_dict = compute_cam_scores(model, input_tensor, reciprocam_gray, target_class)
                road = road_dict['ROAD']
                
                try:
                    adcc_val, drop_val, coh_val, com_val, inc_val, dropindel_val = cam_metric(
                        input_tensor.unsqueeze(0), reciprocam_gray[np.newaxis,...], targets, metric_targets, model,
                        lambda x, t: torch.from_numpy(reciprocam_gray).unsqueeze(0)
                    )
                    scores_map["ReciproCAM"] = {
                        "road": road, "time": reciprocam_time,
                        "adcc": adcc_val[0], "drop": drop_val[0], "coherency": coh_val[0],
                        "complexity": com_val[0], "inc": inc_val[0], "dropindeletion": dropindel_val[0]
                    }
                    
                    # Save to CSV
                    csv_rows.append({
                        'Image': base, 'Model': model_name, 'Method': 'ReciproCAM',
                        'ROAD': road, 'ROAD_Del': road_dict['ROAD_Del'], 'ROAD_Ins': road_dict['ROAD_Ins'],
                        'ADCC': adcc_val[0], 'Drop': drop_val[0],
                        'Coherency': coh_val[0], 'Complexity': com_val[0],
                        'Inc': inc_val[0], 'DropInDeletion': dropindel_val[0], 'Time': reciprocam_time
                    })
                except Exception as e:
                    import traceback
                    print(f"[WARN] ADCC failed for ReciproCAM: {e}")
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    scores_map["ReciproCAM"] = {"road": road, "time": reciprocam_time}

            if ep_gray is not None:
                road_dict = compute_cam_scores(model, input_tensor, ep_gray, target_class)
                road = road_dict['ROAD']

                try:
                    adcc_val, drop_val, coh_val, com_val, inc_val, dropindel_val = cam_metric(
                        input_tensor.unsqueeze(0), ep_gray[np.newaxis,...], targets, metric_targets, model,
                        lambda x, t: torch.from_numpy(ep_gray).unsqueeze(0)
                    )
                    scores_map["EP"] = {
                        "road": road, "time": ep_time,
                        "adcc": adcc_val[0], "drop": drop_val[0], "coherency": coh_val[0],
                        "complexity": com_val[0], "inc": inc_val[0], "dropindeletion": dropindel_val[0]
                    }

                    csv_rows.append({
                        'Image': base, 'Model': model_name, 'Method': 'EP',
                        'ROAD': road, 'ROAD_Del': road_dict['ROAD_Del'], 'ROAD_Ins': road_dict['ROAD_Ins'],
                        'ADCC': adcc_val[0], 'Drop': drop_val[0],
                        'Coherency': coh_val[0], 'Complexity': com_val[0],
                        'Inc': inc_val[0], 'DropInDeletion': dropindel_val[0], 'Time': ep_time
                    })
                except Exception as e:
                    import traceback
                    print(f"[WARN] ADCC failed for EP: {e}")
                    print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                    scores_map["EP"] = {"road": road, "time": ep_time}

            # Score the four new methods (RISE, IG, SHAP, LIME)
            new_methods = {
                "RISE": (rise_gray, rise_time),
                "IG": (ig_gray, ig_time),
                "SHAP": (shap_gray, shap_time),
                "LIME": (lime_gray, lime_time),
            }
            extra_methods_plot = {}
            for method_name, (m_gray, m_time) in new_methods.items():
                if m_gray is not None:
                    road_dict = compute_cam_scores(model, input_tensor, m_gray, target_class)
                    road = road_dict['ROAD']
                    try:
                        _m_gray = m_gray  # capture for lambda
                        adcc_val, drop_val, coh_val, com_val, inc_val, dropindel_val = cam_metric(
                            input_tensor.unsqueeze(0), m_gray[np.newaxis,...], targets, metric_targets, model,
                            lambda x, t, _g=_m_gray: torch.from_numpy(_g).unsqueeze(0)
                        )
                        scores_map[method_name] = {
                            "road": road, "time": m_time,
                            "adcc": adcc_val[0], "drop": drop_val[0], "coherency": coh_val[0],
                            "complexity": com_val[0], "inc": inc_val[0], "dropindeletion": dropindel_val[0]
                        }
                        csv_rows.append({
                            'Image': base, 'Model': model_name, 'Method': method_name,
                            'ROAD': road, 'ROAD_Del': road_dict['ROAD_Del'], 'ROAD_Ins': road_dict['ROAD_Ins'],
                            'ADCC': adcc_val[0], 'Drop': drop_val[0],
                            'Coherency': coh_val[0], 'Complexity': com_val[0],
                            'Inc': inc_val[0], 'DropInDeletion': dropindel_val[0], 'Time': m_time
                        })
                    except Exception as e:
                        import traceback
                        print(f"[WARN] ADCC failed for {method_name}: {e}")
                        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
                        scores_map[method_name] = {"road": road, "time": m_time}

            # Build extra_methods dict for plotting
            extra_methods_plot["RISE"] = {"gray": rise_gray, "vis": rise_vis}
            extra_methods_plot["IG"] = {"gray": ig_gray, "vis": ig_vis}
            extra_methods_plot["SHAP"] = {"gray": shap_gray, "vis": shap_vis}
            extra_methods_plot["LIME"] = {"gray": lime_gray, "vis": lime_vis}

            if first_gray is not None:
                empty_gray = np.zeros_like(first_gray, dtype=np.float32)
                empty_road_dict = compute_cam_scores(model, input_tensor, empty_gray, target_class)
                print(f"Empty CAM ROADCombined score (higher is better): {empty_road_dict['ROAD']:.6f} for {os.path.basename(img_path)} [{model_name}]")

            plot_and_save(img_path, rgb_img, cam_results,
                       ddcam_gray, ddcam_vis, reciprocam_gray, reciprocam_vis,
                       scores_map, out_dir, model_name,
                       ep_gray=ep_gray, ep_vis=ep_vis,
                       extra_methods=extra_methods_plot)

            save_individual_images(
                os.path.basename(img_path), model_name, rgb_img,
                cam_results, ddcam_gray, ddcam_vis,
                reciprocam_gray, reciprocam_vis,
                ep_gray, ep_vis,
                extra_methods_plot, out_dir
            )
    
    csv_path = os.path.join(out_dir, "comparison_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["Image", "Model", "Method", "ROAD", "ROAD_Del", "ROAD_Ins", "ADCC", "Drop", "Coherency", "Complexity", "Inc", "DropInDeletion", "Time"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"\nSaved metrics CSV: {csv_path}")
    
    if csv_rows:
        from collections import defaultdict
        metric_groups = defaultdict(list)
        
        for row in csv_rows:
            key = (row['Model'], row['Method'])
            metric_groups[key].append(row)
        
        avg_rows = []
        metric_fields = ["ROAD", "ROAD_Del", "ROAD_Ins", "ADCC", "Drop", "Coherency", "Complexity", "Inc", "DropInDeletion", "Time"]
        
        for (model, method), rows in sorted(metric_groups.items()):
            avg_row = {"Model": model, "Method": method}
            for field in metric_fields:
                values = []
                for r in rows:
                    try:
                        val = float(r[field])
                        if not np.isnan(val):
                            values.append(val)
                    except (ValueError, TypeError):
                        continue
                
                if values:
                    avg_row[field] = round(np.mean(values), 6)
                else:
                    avg_row[field] = float('nan')
            
            avg_row["NumImages"] = len(rows)
            avg_rows.append(avg_row)
        
        avg_csv_path = os.path.join(out_dir, "comparison_metrics_avg.csv")
        with open(avg_csv_path, "w", newline="") as f:
            fieldnames = ["Model", "Method", "NumImages", "ROAD", "ROAD_Del", "ROAD_Ins", "ADCC", "Drop", "Coherency", "Complexity", "Inc", "DropInDeletion", "Time"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in avg_rows:
                formatted_row = {}
                for key, value in row.items():
                    if isinstance(value, float) and np.isnan(value):
                        formatted_row[key] = 'nan'
                    else:
                        formatted_row[key] = value
                writer.writerow(formatted_row)
        print(f"Saved average metrics CSV: {avg_csv_path}")
        print(f"  Computed averages for {len(avg_rows)} model/method combinations")


if __name__ == "__main__":
    main()