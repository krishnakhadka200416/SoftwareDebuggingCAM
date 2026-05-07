"""
Delta Debugging CAM: Finding Minimal Sufficient Sets for Model Explanations
- ViT: minimal spatial patches that preserve top-1 prediction
- CNN: minimal feature channels at a target conv layer

This implementation properly applies delta debugging to find truly minimal sets
and supports multiple ViT architectures (torchvision, timm, HuggingFace).
"""

import math
from typing import Dict, Iterable, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

from .utils import (
    set_eval,
    top1,
    overlay_heatmap_on_tensor,
)

def ddmin(
    universe: List[int],
    test_sufficient,
    verbose: bool = False
) -> List[int]:
    """
    Classic delta debugging (Zeller 1999): find a 1-minimal sufficient subset.
    A subset is 1-minimal if removing any single element makes it insufficient.
    """
    current = list(universe)
    n = 2
    iteration = 0
    
    while len(current) >= 2:
        iteration += 1
        subset_size = math.ceil(len(current) / n)
        subsets = [current[i*subset_size:(i+1)*subset_size] for i in range(n)]
        
        reduced = False
        if verbose:
            print(f"[DD] iter={iteration} | |S|={len(current)} | n={n} | subset_size≈{subset_size}")
        
        for k, subset in enumerate(subsets, 1):
            complement = [i for i in current if i not in subset]
            
            if verbose:
                print(f"  - test complement {k}/{n}: keep {len(complement)}, remove {len(subset)}")
            
            if test_sufficient(complement):
                current = complement
                n = max(2, n - 1)
                reduced = True
                if verbose:
                    print(f"    ✓ reduction → |S|={len(current)}")
                break
        
        if not reduced:
            if n >= len(current):
                break
            n = min(len(current), n * 2)
    
    return current


class AbstractExplainer(ABC):
    """Abstract base class for model explainers using delta debugging."""
    
    @abstractmethod
    def find_minimal(self, x: torch.Tensor, **kwargs) -> Tuple:
        """Find minimal sufficient set for the input."""
        pass
    
    @abstractmethod
    def build_heatmap(self, x: torch.Tensor, kept: List[int], **kwargs) -> np.ndarray:
        """Build importance heatmap from the minimal set."""
        pass


class ViTAdapter:
    """Adapter to handle different ViT architectures uniformly."""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.architecture = self._detect_architecture()
        
    def _detect_architecture(self) -> str:
        """Detect the ViT architecture type."""
        model_class = self.model.__class__.__name__.lower()
        
        if hasattr(self.model, '_process_input') and hasattr(self.model, 'class_token'):
            return "torchvision"
        elif hasattr(self.model, 'patch_embed') and hasattr(self.model, 'cls_token'):
            return "timm"
        elif hasattr(self.model, 'embeddings') and hasattr(self.model, 'encoder'):
            return "huggingface"
        elif 'deit' in model_class:
            return "deit"
        else:
            if any('patch_embed' in name for name, _ in self.model.named_modules()):
                return "timm"
            return "unknown"
    
    def get_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """Get the patch grid dimensions."""
        if self.architecture == "torchvision":
            if hasattr(self.model, 'patch_size'):
                ps = self.model.patch_size
                if isinstance(ps, (tuple, list)):
                    ph, pw = ps[0], ps[1]
                else:
                    ph = pw = ps
                return H // ph, W // pw
        
        elif self.architecture in ["timm", "deit"]:
            if hasattr(self.model, 'patch_embed'):
                pe = self.model.patch_embed
                if hasattr(pe, 'grid_size'):
                    return pe.grid_size
                elif hasattr(pe, 'patch_size'):
                    ps = pe.patch_size
                    if isinstance(ps, (tuple, list)):
                        return H // ps[0], W // ps[1]
                    else:
                        return H // ps, W // ps
        
        elif self.architecture == "huggingface":
            if hasattr(self.model, 'config'):
                ps = self.model.config.patch_size
                return H // ps, W // ps
        
        return H // 16, W // 16
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings before positional encoding."""
        if self.architecture == "torchvision":
            return self.model._process_input(x)
        
        elif self.architecture in ["timm", "deit"]:
            if hasattr(self.model, 'patch_embed'):
                return self.model.patch_embed(x)
            elif hasattr(self.model, 'forward_features'):
                return self.model.patch_embed(x)
        
        elif self.architecture == "huggingface":
            embeddings = self.model.embeddings(x)
            if embeddings.shape[1] > (x.shape[-2] // 16) * (x.shape[-1] // 16):
                return embeddings[:, 1:]
            return embeddings
        
        else:
            raise NotImplementedError(f"Architecture {self.architecture} not supported")
    
    def forward_with_masked_patches(
        self,
        x: torch.Tensor,
        keep_indices: List[int],
        ablation_method: str = "zero"
    ) -> torch.Tensor:

        B, C, H, W = x.shape
        gh, gw = self.get_grid_size(H, W)
        n_patches = gh * gw
        keep_set = set(keep_indices)

        def hook_fn(module, input, output):
            tokens = output.clone()
            B_out, seq_len, D = tokens.shape
            N = seq_len - 1
            n_c = int(N ** 0.5)

            spatial_tokens = tokens[:, 1:, :].reshape(B_out, n_c, n_c, D)

            mask = torch.zeros(n_c, n_c, device=tokens.device, dtype=torch.bool)
            for p_idx in keep_set:
                i = p_idx // n_c
                j = p_idx % n_c
                if 0 <= i < n_c and 0 <= j < n_c:
                    mask[i, j] = True
            spatial_tokens = spatial_tokens * mask.view(1, n_c, n_c, 1).float()

            tokens[:, 1:, :] = spatial_tokens.reshape(B_out, N, D)
            return tokens

        target_layer = self.model.encoder.layers[0]
        hook = target_layer.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                logits = self.model(x)
        finally:
            hook.remove()

        return logits


class MinimalPatchesViT(AbstractExplainer):
    """Find minimal sufficient patches for ViT models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        min_prob: float = 0.15,
        min_margin: float = 0.05,
        rel_baseline: float = 0.50,
        min_patches: int = 4,
        min_fraction: float = 0.10,
        ablation_method: str = "zero",
        strategy: str = "dd",
        simple_mode: bool = False
    ):
        self.model = model
        self.device = device
        self.adapter = ViTAdapter(model, device)
        
        set_eval(self.model)
        self.model.to(self.device)
        
        # Thresholds for sufficiency test
        self.min_prob = min_prob
        self.min_margin = min_margin
        self.rel_baseline = rel_baseline
        self.min_patches = min_patches
        self.min_fraction = min_fraction
        self.ablation_method = ablation_method
        self.strategy = strategy
        self.simple_mode = simple_mode
        
        self._baseline_logits = None
        self._baseline_class = None
        self._baseline_prob = None
    
    def _test_sufficient(
        self, 
        x: torch.Tensor,
        keep_indices: List[int],
        verbose: bool = False
    ) -> bool:
        """Test if keeping only these patches maintains the prediction."""
        if len(keep_indices) == 0:
            return False
        
        if self.simple_mode:
            logits = self.adapter.forward_with_masked_patches(
                x, keep_indices, self.ablation_method
            )
            probs = F.softmax(logits[0], dim=0)
            prob_target = float(probs[self._baseline_class].item())
            is_sufficient = prob_target > 0.01
            
            if verbose:
                print(f"  Keep {len(keep_indices):3d} patches | "
                      f"p={prob_target:.4f} | sufficient={is_sufficient}")
            
            return is_sufficient
        
        if len(keep_indices) < self.min_patches:
            return False
        
        gh, gw = self.adapter.get_grid_size(x.shape[-2], x.shape[-1])
        n_patches = gh * gw
        
        if len(keep_indices) < math.ceil(self.min_fraction * n_patches):
            return False
        
        logits = self.adapter.forward_with_masked_patches(
            x, keep_indices, self.ablation_method
        )
        
        probs = F.softmax(logits[0], dim=0)
        pred = int(torch.argmax(probs))
        prob_target = float(probs[self._baseline_class].item())
        
        top2 = torch.topk(probs, k=min(2, probs.numel())).values
        margin = float((prob_target - top2[-1].item()) if len(top2) >= 2 else prob_target)
        rel_prob = prob_target / max(self._baseline_prob, 1e-8)
        
        is_sufficient = (
            pred == self._baseline_class and
            prob_target >= self.min_prob and
            margin >= self.min_margin and
            rel_prob >= self.rel_baseline
        )
        
        if verbose:
            print(f"  Keep {len(keep_indices):3d} patches | "
                  f"pred={pred} (want {self._baseline_class}) | "
                  f"p={prob_target:.3f} | margin={margin:.3f} | "
                  f"rel={rel_prob:.2f} | sufficient={is_sufficient}")
        
        return is_sufficient
    
    def find_minimal(
        self,
        x: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[List[int], int, int, float, Tuple[int, int]]:
        """
        Find minimal sufficient set of patches.
        Returns: (kept_patches, class_idx, num_forwards, baseline_prob, (gh, gw))
        """
        x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
        
        try:
            with torch.no_grad():
                self._baseline_logits = self.model(x)
        finally:
            pass
        self._baseline_class, self._baseline_prob = top1(self._baseline_logits)
        
        gh, gw = self.adapter.get_grid_size(x.shape[-2], x.shape[-1])
        n_patches = gh * gw
        
        if verbose:
            print(f"[ViT] Grid: {gh}x{gw} = {n_patches} patches")
            print(f"[ViT] Baseline: class={self._baseline_class}, prob={self._baseline_prob:.3f}")
            print(f"[ViT] Strategy: {self.strategy}")
        
        forwards = 1
        
        def sufficient_test(keep: List[int]) -> bool:
            nonlocal forwards
            forwards += 1
            return self._test_sufficient(x, keep, verbose=verbose)
        
        if self.strategy == "dd":
            universe = list(range(n_patches))
            
            if not sufficient_test(universe):
                raise ValueError("Keeping all patches doesn't preserve prediction - check model/preprocessing")
            
            kept = ddmin(universe, sufficient_test, verbose=verbose)
            
        elif self.strategy == "grow":
            kept = []
            remaining = list(range(n_patches))
            
            while remaining and not sufficient_test(kept):
                best_patch = None
                best_prob = -1
                
                for patch in remaining:
                    trial = kept + [patch]
                    forwards += 1
                    logits = self.adapter.forward_with_masked_patches(
                        x, trial, self.ablation_method
                    )
                    probs = F.softmax(logits[0], dim=0)
                    prob = float(probs[self._baseline_class].item())
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_patch = patch
                
                if best_patch is not None:
                    kept.append(best_patch)
                    remaining.remove(best_patch)
                    if verbose:
                        print(f"[Grow] Added patch {best_patch}, |S|={len(kept)}, prob={best_prob:.3f}")
            
            if len(kept) > 1:
                kept = ddmin(kept, sufficient_test, verbose=verbose)
        
        elif self.strategy == "hybrid":
            importance_scores = self._compute_importance_scores(x, gh, gw)
            forwards += n_patches
            
            sorted_patches = sorted(
                range(n_patches), 
                key=lambda i: importance_scores[i], 
                reverse=True
            )
            
            left, right = 1, n_patches
            best_k = n_patches
            
            while left <= right:
                mid = (left + right) // 2
                test_set = sorted_patches[:mid]
                
                if sufficient_test(test_set):
                    best_k = mid
                    right = mid - 1
                else:
                    left = mid + 1
            
            kept = sorted_patches[:best_k]
            kept = ddmin(kept, sufficient_test, verbose=verbose)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        if verbose:
            print(f"[ViT] Final: {len(kept)} patches, {forwards} forward passes")
        
        return kept, self._baseline_class, forwards, self._baseline_prob, (gh, gw)
    
    def _compute_importance_scores(
        self, 
        x: torch.Tensor, 
        gh: int, 
        gw: int
    ) -> np.ndarray:
        """Compute importance score for each patch using occlusion."""
        n_patches = gh * gw
        scores = np.zeros(n_patches)
        
        for i in range(n_patches):
            keep = [j for j in range(n_patches) if j != i]
            logits = self.adapter.forward_with_masked_patches(
                x, keep, self.ablation_method
            )
            probs = F.softmax(logits[0], dim=0)
            prob_without = float(probs[self._baseline_class].item())
            scores[i] = max(0, self._baseline_prob - prob_without)
        
        return scores
    
    def build_heatmap(
        self, 
        x: torch.Tensor, 
        kept: List[int], 
        grid: Tuple[int, int]
    ) -> np.ndarray:
        """Build spatial importance heatmap."""
        gh, gw = grid
        heatmap = np.zeros((gh, gw))
        
        if self.strategy in ["dd", "grow", "hybrid"]:
            if len(kept) > 0:
                x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
                
                for patch in kept:
                    test_kept = [p for p in kept if p != patch]
                    
                    if len(test_kept) > 0:
                        logits = self.adapter.forward_with_masked_patches(
                            x, test_kept, self.ablation_method
                        )
                        probs = F.softmax(logits[0], dim=0)
                        prob_without = float(probs[self._baseline_class].item())
                        importance = max(0, self._baseline_prob - prob_without)
                    else:
                        importance = 1.0
                    
                    row, col = patch // gw, patch % gw
                    heatmap[row, col] = importance
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=0.3)
        
        heatmap_resized = cv2.resize(
            heatmap, 
            (x.shape[-1], x.shape[-2]), 
            interpolation=cv2.INTER_LINEAR
        )
        
        return heatmap_resized


class MinimalChannelsCNN(AbstractExplainer):
    """Find minimal sufficient channels at a target conv layer."""
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: nn.Module, 
        device: str = "cuda"
    ):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        
        set_eval(self.model)
        self.model.to(self.device)
        
        self.last_activation = None
        self._baseline_class = None
        self._baseline_prob = None
    
    def _hook_and_mask_forward(
        self, 
        x: torch.Tensor, 
        keep_channels: List[int]
    ) -> torch.Tensor:
        """Forward pass with channel masking at target layer."""
        keep_set = set(keep_channels)
        
        def hook_fn(module, input, output):
            mask = torch.zeros_like(output)
            if len(keep_set) > 0:
                for c in keep_set:
                    if c < output.shape[1]:
                        mask[:, c, :, :] = 1.0
            return output * mask
        
        hook = self.target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                logits = self.model(x)
        finally:
            hook.remove()
        return logits

    def _capture_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Capture activations at target layer."""
        activation = {}
        
        def hook_fn(module, input, output):
            activation['output'] = output.detach()
        
        hook = self.target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = self.model(x)
        finally:
            hook.remove()
        self.last_activation = activation['output']
        return self.last_activation

    def find_minimal(
        self,
        x: torch.Tensor,
        min_prob: float = 0.15,
        min_margin: float = 0.05,
        rel_baseline: float = 0.50,
        min_channels: int = 1,
        verbose: bool = False
    ) -> Tuple[List[int], int, int, float]:
        """
        Find minimal sufficient set of channels.
        Returns: (kept_channels, class_idx, num_forwards, baseline_prob)
        """
        x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
        
        with torch.no_grad():
            baseline_logits = self.model(x)
        self._baseline_class, self._baseline_prob = top1(baseline_logits)
        
        act = self._capture_activation(x)
        num_channels = act.shape[1]
        
        if verbose:
            print(f"[CNN] Target layer has {num_channels} channels")
            print(f"[CNN] Baseline: class={self._baseline_class}, prob={self._baseline_prob:.3f}")
        
        forwards = 2
        
        def sufficient_test(keep: List[int]) -> bool:
            nonlocal forwards
            forwards += 1
            
            if len(keep) < min_channels:
                return False
            
            logits = self._hook_and_mask_forward(x, keep)
            probs = F.softmax(logits[0], dim=0)
            pred = int(torch.argmax(probs))
            prob_target = float(probs[self._baseline_class].item())
            
            top2 = torch.topk(probs, k=min(2, probs.numel())).values
            margin = float((prob_target - top2[-1].item()) if len(top2) >= 2 else prob_target)
            rel_prob = prob_target / max(self._baseline_prob, 1e-8)
            
            is_sufficient = (
                pred == self._baseline_class and
                prob_target >= min_prob and
                margin >= min_margin and
                rel_prob >= rel_baseline
            )
            
            if verbose:
                print(f"  Keep {len(keep):3d} channels | "
                      f"pred={pred} (want {self._baseline_class}) | "
                      f"p={prob_target:.3f} | margin={margin:.3f} | "
                      f"rel={rel_prob:.2f} | sufficient={is_sufficient}")
            
            return is_sufficient
        
        universe = list(range(num_channels))
        logits_all = self._hook_and_mask_forward(x, universe)
        probs_all = F.softmax(logits_all[0], dim=0)
        pred_all = int(torch.argmax(probs_all))
        if pred_all != self._baseline_class:
            print(f"[WARN] Keeping all channels changed prediction from {self._baseline_class} to {pred_all}")
        
        if not sufficient_test(universe):
            print(f"[WARN] All channels don't meet sufficiency thresholds - relaxing constraints")
        
        kept = ddmin(universe, sufficient_test, verbose=verbose)
        
        if verbose:
            print(f"[CNN] Final: {len(kept)} channels, {forwards} forward passes")
        
        return kept, self._baseline_class, forwards, self._baseline_prob
    
    def build_heatmap(
        self, 
        x: torch.Tensor, 
        kept: List[int]
    ) -> np.ndarray:
        """Build spatial importance heatmap from kept channels."""
        if self.last_activation is None:
            x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
            self._capture_activation(x)
        
        act = self.last_activation[0].cpu()
        
        if len(kept) == 0:
            heatmap = torch.zeros_like(act[0])
        else:
            weighted_acts = []
            for c in kept:
                if c < act.shape[0]:
                    channel_act = F.relu(act[c])
                    test_kept = [k for k in kept if k != c]
                    if len(test_kept) > 0:
                        logits = self._hook_and_mask_forward(
                            x.unsqueeze(0).to(self.device) if x.dim() == 3 else x, 
                            test_kept
                        )
                        probs = F.softmax(logits[0], dim=0)
                        prob_without = float(probs[self._baseline_class].item())
                        importance = max(0, self._baseline_prob - prob_without)
                    else:
                        importance = 1.0
                    
                    weighted_acts.append(channel_act * importance)
            
            if weighted_acts:
                heatmap = torch.stack(weighted_acts).sum(dim=0)
            else:
                heatmap = torch.zeros_like(act[0])
        
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        heatmap_resized = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),
            size=(x.shape[-2], x.shape[-1]),
            mode='bilinear',
            align_corners=False
        ).squeeze().numpy()
        
        return heatmap_resized


class DeltaDebuggingCAM:
    """Unified interface for CNN channel-based explanations."""
    
    def __init__(
        self, 
        model: nn.Module, 
        target_layer: nn.Module, 
        device: str = "cuda"
    ):
        self.explainer = MinimalChannelsCNN(model, target_layer, device)
        self.device = device
    
    def __call__(
        self, 
        x: torch.Tensor, 
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[int], int]:
        """Generate explanation."""
        kept, _, forwards, _ = self.explainer.find_minimal(x, verbose=verbose)
        heatmap = self.explainer.build_heatmap(x, kept)
        return heatmap, kept, forwards
    
    def overlay_on_image(
        self, 
        x: torch.Tensor, 
        heatmap: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create visualization overlay."""
        return overlay_heatmap_on_tensor(x, heatmap, alpha=alpha)


class DeltaDebuggingViT:
    """Unified interface for ViT patch-based explanations."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer = None,
        device: str = "cuda",
        strategy: str = "dd",
        simple_mode: bool = False,
        **kwargs
    ):
        if isinstance(target_layer, str):
            device = target_layer
        elif target_layer is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.explainer = MinimalPatchesViT(
            model,
            device=device,
            strategy=strategy,
            simple_mode=simple_mode,
            **kwargs
        )
        self.device = device
        self._last_grid = None
    
    def __call__(
        self,
        x: torch.Tensor,
        verbose: bool = False
    ) -> Tuple[np.ndarray, List[int], int]:
        """Generate explanation."""
        kept, _, forwards, _, grid = self.explainer.find_minimal(x, verbose=verbose)
        self._last_grid = grid
        heatmap = self.explainer.build_heatmap(x, kept, grid)
        return heatmap, kept, forwards
    
    def overlay_on_image(
        self, 
        x: torch.Tensor, 
        heatmap: np.ndarray, 
        alpha: float = 0.5
    ) -> np.ndarray:
        """Create visualization overlay."""
        return overlay_heatmap_on_tensor(x, heatmap, alpha=alpha)


def verify_minimal_set(
    explainer: Union[MinimalPatchesViT, MinimalChannelsCNN],
    x: torch.Tensor,
    kept_set: List[int],
    name: str = "Model"
) -> bool:
    """Verify that a set is truly 1-minimal."""
    x = x.unsqueeze(0).to(explainer.device) if x.dim() == 3 else x.to(explainer.device)
    
    if isinstance(explainer, MinimalPatchesViT):
        is_sufficient = explainer._test_sufficient(x, kept_set, verbose=False)
    else:
        logits = explainer._hook_and_mask_forward(x, kept_set)
        probs = F.softmax(logits[0], dim=0)
        pred = int(torch.argmax(probs))
        is_sufficient = (pred == explainer._baseline_class)
    
    if not is_sufficient:
        print(f"[{name}] WARNING: Kept set is not sufficient!")
        return False
    
    for elem in kept_set:
        subset = [k for k in kept_set if k != elem]
        
        if isinstance(explainer, MinimalPatchesViT):
            if explainer._test_sufficient(x, subset, verbose=False):
                print(f"[{name}] WARNING: Not 1-minimal! Element {elem} can be removed.")
                return False
        else:
            logits = explainer._hook_and_mask_forward(x, subset)
            probs = F.softmax(logits[0], dim=0)
            pred = int(torch.argmax(probs))
            if pred == explainer._baseline_class:
                print(f"[{name}] WARNING: Not 1-minimal! Channel {elem} can be removed.")
                return False
    
    print(f"[{name}] ✓ Set is 1-minimal and sufficient")
    return True
