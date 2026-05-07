"""
DDCAM: Delta Debugging CAM for Minimal Sufficient Explanations

A package for finding minimal sufficient sets of features (channels for CNNs, 
patches for ViTs) that preserve model predictions using delta debugging.
"""

from .core import (
    ddmin,
    DeltaDebuggingCAM,
    DeltaDebuggingViT,
    MinimalChannelsCNN,
    MinimalPatchesViT,
    ViTAdapter,
    AbstractExplainer,
)
from .multi_class import MultiClassDDCAM, ddmin_optimized
from .explainer import DDCAMExplainer
from .utils import (
    overlay_heatmap_on_tensor,
    normalize,
    set_eval,
    top1,
    to_device,
    apply_gaussian_patch_smoothing,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "ddmin",
    "DeltaDebuggingCAM",
    "DeltaDebuggingViT",
    "MinimalChannelsCNN",
    "MinimalPatchesViT",
    "ViTAdapter",
    "AbstractExplainer",
    # Multi-class
    "MultiClassDDCAM",
    "ddmin_optimized",
    # High-level interface
    "DDCAMExplainer",
    # Utilities
    "overlay_heatmap_on_tensor",
    "normalize",
    "set_eval",
    "top1",
    "to_device",
    "apply_gaussian_patch_smoothing",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]

