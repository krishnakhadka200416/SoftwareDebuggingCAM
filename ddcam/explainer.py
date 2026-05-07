import numpy as np
import torch
from typing import Optional, Tuple


from .utils import overlay_heatmap_on_tensor, apply_gaussian_patch_smoothing
from .multi_class import MultiClassDDCAM


def _is_non_interacting_head_resnet(model: torch.nn.Module) -> bool:
    """Check if ResNet has non-interacting features (GAP + single Linear layer)."""
    return hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear)


def _is_non_interacting_head_vgg(model: torch.nn.Module) -> bool:
    """VGG has interacting features (multi-layer classifier)."""
    return False


def _is_non_interacting_head_efficientnet(model: torch.nn.Module) -> bool:
    """Check if EfficientNet has non-interacting features."""
    return hasattr(model, 'classifier') and any(isinstance(m, torch.nn.Linear) for m in model.classifier.modules())


def _is_non_interacting_head_inception(model: torch.nn.Module) -> bool:
    """Check if Inception has non-interacting features (GAP + single Linear layer)."""
    return hasattr(model, 'fc') and isinstance(model.fc, torch.nn.Linear)


def _is_non_interacting_head_vit(model: torch.nn.Module) -> bool:
    """Check if ViT has non-interacting features (only Linear/Identity/Dropout/LayerNorm in head)."""
    head = getattr(model, 'heads', None)
    if head is None:
        head = getattr(model, 'head', None)
    if head is None:
        return False
    mods = list(head.modules())
    leafs = [m for m in mods[1:] if len(list(m.children())) == 0]
    allowed = (torch.nn.Linear, torch.nn.Identity, torch.nn.Dropout, torch.nn.LayerNorm)
    return all(isinstance(m, allowed) for m in leafs) and any(isinstance(m, torch.nn.Linear) for m in leafs)


class DDCAMExplainer:
    """
    DDCAM explanation interface that adapts strategy based on feature interactions.

    - Non-interacting features (e.g., GAP + single FC): Uses individual feature testing
      - Each feature contributes independently, so we can test removal individually
    - Interacting features (e.g., multi-layer FC): Uses Delta Debugging with relative sufficiency
      - Features interact through multiple layers, requiring joint optimization
    """

    def __init__(
        self,
        model: torch.nn.Module,
        target_layer,
        device: str = 'cuda',
        is_vit: bool = False,
        apply_vit_smoothing: bool = True,
        strategy: str = 'auto',  # 'auto' | 'individual' | 'dd'
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.is_vit = is_vit
        self.apply_vit_smoothing = apply_vit_smoothing
        assert strategy in ('auto', 'individual', 'dd'), "strategy must be 'auto', 'individual', or 'dd'"
        self.strategy = strategy

        self.has_non_interacting_features = self._detect_non_interacting_features()

    def _detect_non_interacting_features(self) -> bool:
        """Detect if model has non-interacting features (can test individually) or interacting features (need DD)."""
        if self.is_vit:
            return _is_non_interacting_head_vit(self.model)
        if _is_non_interacting_head_resnet(self.model):
            return True
        if _is_non_interacting_head_efficientnet(self.model):
            return True
        if _is_non_interacting_head_inception(self.model):
            return True
        if _is_non_interacting_head_vgg(self.model):
            return True
        return False

    @torch.no_grad()
    def _get_baseline(self, x: torch.Tensor) -> Tuple[int, float]:
        logits = self.model(x.unsqueeze(0))
        probs = torch.nn.functional.softmax(logits, dim=1)
        top1 = int(torch.argmax(probs, dim=1).item())
        return top1, float(probs[0, top1].item())

    def explain(
        self,
        x: torch.Tensor,
        target_class: Optional[int] = None,
        retention_ratio: float = 1.0,
        min_absolute_prob: float = 0.01,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate DDCAM explanation heatmap and overlay.

        Args:
            x: Input tensor [3,H,W]
            target_class: If None, uses top-1
            retention_ratio: α for relative sufficiency (used in DD backend)
            min_absolute_prob: safety floor for probability (DD backend)

        Returns: (grayscale_heatmap [H,W], overlay_rgb [H,W,3]) both in [0,1] if successful, else (None, None)
        """
        x = x.to(self.device)
        if target_class is None:
            target_class, _ = self._get_baseline(x)

        mode = self.strategy
        if mode == 'auto':
            mode = 'individual' if self.has_non_interacting_features else 'dd'

        if mode == 'individual':
            mc = MultiClassDDCAM(
                self.model,
                self.target_layer,
                device=self.device,
                is_vit=self.is_vit,
                retention_ratio=retention_ratio,
                min_absolute_prob=min_absolute_prob,
            )
            heatmap, _, _ = mc.explain_class(
                x, target_class, baseline_prob=None, verbose=False,
                max_iterations=150 if self.is_vit else 100,
                early_stop_size=10 if self.is_vit else 20,
                initial_partition_n=-1,
            )
            if heatmap is None:
                return None, None
            if self.is_vit and self.apply_vit_smoothing:
                from scipy.ndimage import zoom
                patch_size = 14
                patch_map = zoom(heatmap, (patch_size / heatmap.shape[0], patch_size / heatmap.shape[1]), order=1)
                patch_map = apply_gaussian_patch_smoothing(patch_map, sigma=1.0, mix=0.35, gamma=1.3)
                heatmap = zoom(patch_map, (heatmap.shape[0] / patch_size, heatmap.shape[1] / patch_size), order=1)
            overlay = overlay_heatmap_on_tensor(x, heatmap, alpha=0.5)
            return heatmap, overlay

        mc = MultiClassDDCAM(
            self.model,
            self.target_layer,
            device=self.device,
            is_vit=self.is_vit,
            retention_ratio=retention_ratio,
            min_absolute_prob=min_absolute_prob,
        )
        heatmap, _, _ = mc.explain_class(
            x, target_class, baseline_prob=None, verbose=False,
            max_iterations=150 if self.is_vit else 100,
            early_stop_size=10 if self.is_vit else 20,
            initial_partition_n=None,
        )
        if heatmap is None:
            return None, None

        if self.is_vit and self.apply_vit_smoothing:
            from scipy.ndimage import zoom
            patch_size = 14
            patch_map = zoom(heatmap, (patch_size / heatmap.shape[0], patch_size / heatmap.shape[1]), order=1)
            patch_map = apply_gaussian_patch_smoothing(patch_map, sigma=1.0, mix=0.35, gamma=1.3)
            heatmap = zoom(patch_map, (heatmap.shape[0] / patch_size, heatmap.shape[1] / patch_size), order=1)

        overlay = overlay_heatmap_on_tensor(x, heatmap, alpha=0.5)
        return heatmap, overlay


