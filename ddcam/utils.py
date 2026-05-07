import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple
from PIL import Image
import cv2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_device(x, device):
    """Move tensor to device."""
    return x.to(device, non_blocking=True)


def set_eval(model):
    """Set model to eval mode and disable gradients."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)


def top1(logits: torch.Tensor) -> Tuple[int, float]:
    """Get top-1 prediction and probability."""
    probs = F.softmax(logits, dim=-1)
    val, idx = torch.max(probs, dim=-1)
    return idx.item(), val.item()


def overlay_heatmap_on_tensor(
    input_tensor: torch.Tensor, 
    heatmap: np.ndarray, 
    alpha: float = 0.5, 
    use_rgb: bool = True
) -> np.ndarray:
    """Overlay heatmap on input tensor for visualization."""
    x = input_tensor.detach().cpu().float().numpy()
    x = np.transpose(x, (1, 2, 0))  # HWC
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    heat = np.uint8(255 * np.clip(heatmap, 0, 1))
    color = cv2.applyColorMap(heat, cv2.COLORMAP_JET).astype(np.float32) / 255.0
    if use_rgb:
        color = color[..., ::-1]
    
    out = color * alpha + x * (1 - alpha)
    out = out / np.maximum(out.max(), 1e-8)
    return np.uint8(255 * out)


def normalize(im: Image.Image, size: int = 224) -> torch.Tensor:
    """Normalize image for model input."""
    import torchvision.transforms as T
    tfm = T.Compose([
        T.Resize(size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(im)


def apply_gaussian_patch_smoothing(
    heatmap: np.ndarray, 
    sigma: float = 1.0, 
    mix: float = 0.35, 
    gamma: float = 1.3
) -> np.ndarray:
    """
    Apply Gaussian smoothing to spread patch importance to 4x4 neighborhood.
    
    Args:
        heatmap: 2D array of shape (H, W) with patch-level importance scores
        sigma: Standard deviation for Gaussian kernel (1.0 creates ~4x4 neighborhood effect)
        mix: Blend ratio between original and smoothed (0.35 = 35% smoothed, 65% original)
        gamma: Gamma correction for contrast (1.3 = brighten)
    
    Returns:
        Smoothed and normalized heatmap
    """
    from scipy.ndimage import gaussian_filter
    smoothed = gaussian_filter(heatmap, sigma=sigma, mode='constant', cval=0)
    blended = (1.0 - mix) * heatmap + mix * smoothed
    hmin, hmax = blended.min(), blended.max()
    if (hmax - hmin) > 1e-8:
        blended = (blended - hmin) / (hmax - hmin + 1e-8)
    blended = np.power(np.clip(blended, 0.0, 1.0), 1.0 / max(gamma, 1e-6))
    return blended

