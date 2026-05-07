from typing import List, Tuple
import torch
import torch.nn.functional as F
import numpy as np

from .core import (
    MinimalChannelsCNN,
    MinimalPatchesViT,
    ddmin,
)
from .utils import (
    normalize,
    overlay_heatmap_on_tensor,
    set_eval,
)


def ddmin_optimized(universe: List[int], test_sufficient, verbose: bool = False,
                    max_iterations: int = 200, early_stop_size: int = None,
                    forwards_counter = None, initial_partition_n: int = None):
    """
    Optimized delta debugging with early stopping and iteration limits.
    
    Args:
        universe: Full set to minimize
        test_sufficient: Function to test if a set is sufficient
        max_iterations: Maximum number of iterations
        early_stop_size: Stop early if we find a set this small or smaller
        forwards_counter: Function that returns current forward pass count
    """
    import math
    
    current = list(universe)
    if initial_partition_n is None:
        n = 2
    elif initial_partition_n == -1:
        n = max(2, len(current))
    else:
        n = max(2, min(len(current), int(initial_partition_n)))
    iteration = 0
    
    while len(current) >= 2 and iteration < max_iterations:
        iteration += 1
        
        if forwards_counter and forwards_counter() >= max_iterations:
            if verbose:
                print(f"[DD] Stopping early: reached max forward passes ({max_iterations})")
            break
        
        if early_stop_size and len(current) <= early_stop_size:
            if verbose:
                print(f"[DD] Early stopping: found set of size {len(current)} <= {early_stop_size}")
            break
        
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


class MultiClassDDCAM:
    """Generate DDCAM explanations for multiple target classes using Relative Minimal Sufficiency."""
    
    def __init__(self, model, target_layer, device="cuda", is_vit=False, 
                 retention_ratio=0.7, min_absolute_prob=0.01):
        """
        Args:
            model: PyTorch model
            target_layer: Target layer for feature extraction
            device: Device to run on
            is_vit: Whether model is Vision Transformer
            retention_ratio: α ∈ (0,1]. Keep this fraction of baseline probability.
            min_absolute_prob: Safety threshold to avoid noise (e.g., 0.01)
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.is_vit = is_vit
        self.alpha = retention_ratio
        self.min_absolute = min_absolute_prob
        
        set_eval(self.model)
        self.model.to(self.device)
    
    def get_top_k_predictions(self, x, k=2):
        """Get top-k predictions from the model."""
        x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
        
        probs = F.softmax(logits[0], dim=0)
        top_probs, top_indices = torch.topk(probs, k)
        
        return [(idx.item(), prob.item()) for idx, prob in zip(top_indices, top_probs)]
    
    def explain_class(self, x, target_class, baseline_prob=None, verbose=False, max_iterations=100, early_stop_size=20,
                      initial_partition_n: int = None):
        """Generate DDCAM explanation for a specific target class using relative sufficiency."""
        x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
        
        if baseline_prob is None:
            with torch.no_grad():
                baseline_logits = self.model(x)
            baseline_probs = F.softmax(baseline_logits[0], dim=0)
            baseline_prob = baseline_probs[target_class].item()
        
        target_prob = self.alpha * baseline_prob
        
        if verbose:
            print(f"  [Relative Minimality]")
            print(f"  Target class: {target_class}")
            print(f"  Baseline probability: {baseline_prob:.4f}")
            print(f"  Retention ratio α: {self.alpha}")
            print(f"  Target probability: ≥ {target_prob:.4f} (α * baseline)")
            print(f"  Absolute minimum: ≥ {self.min_absolute:.4f}")
        
        if self.is_vit:
            vit_strategy = "dd" if initial_partition_n is not None else "hybrid"
            explainer = MultiClassMinimalPatchesViT(
                self.model,
                device=self.device,
                strategy=vit_strategy,
                retention_ratio=self.alpha,
                min_absolute_prob=self.min_absolute
            )
            
            with torch.no_grad():
                baseline_logits = self.model(x)
            baseline_probs = F.softmax(baseline_logits[0], dim=0)
            
            explainer._baseline_logits = baseline_logits
            explainer._baseline_class = target_class
            explainer._baseline_prob = baseline_probs[target_class].item()
            
            try:
                kept, forwards, final_prob, retention = explainer.find_minimal(
                    x.squeeze(0), verbose=verbose, initial_partition_n=initial_partition_n
                )
                heatmap = explainer.build_heatmap(x.squeeze(0), kept, explainer._last_grid)
                return heatmap, kept, forwards
            except Exception as e:
                print(f"[WARN] Explanation failed for class {target_class}: {e}")
                return None, [], 0
        else:
            explainer = MultiClassMinimalChannelsCNN(
                self.model,
                self.target_layer,
                device=self.device,
                target_class=target_class,
                retention_ratio=self.alpha,
                min_absolute_prob=self.min_absolute
            )
            
            try:
                kept, forwards, final_prob, retention = explainer.find_minimal(
                    x.squeeze(0),
                    verbose=verbose,
                    max_iterations=max_iterations,
                    early_stop_size=early_stop_size,
                    initial_partition_n=initial_partition_n
                )
                heatmap = explainer.build_heatmap(x.squeeze(0), kept)
                return heatmap, kept, forwards
            except Exception as e:
                print(f"[WARN] Explanation failed for class {target_class}: {e}")
                return None, [], 0
    
    def explain_top_k(self, x, k=2, verbose=False):
        """Generate explanations for top-k predicted classes."""
        # Get top-k predictions
        top_predictions = self.get_top_k_predictions(x, k)
        
        print(f"\nTop-{k} predictions:")
        for i, (cls_idx, prob) in enumerate(top_predictions):
            print(f"  {i+1}. Class {cls_idx}: {prob:.3f}")
        
        # Generate explanation for each class
        results = []
        for cls_idx, prob in top_predictions:
            print(f"\nGenerating explanation for class {cls_idx} (prob={prob:.3f})...")
            heatmap, kept, forwards = self.explain_class(x, cls_idx, baseline_prob=prob, verbose=verbose)
            
            if heatmap is not None:
                results.append({
                    'class_idx': cls_idx,
                    'probability': prob,
                    'heatmap': heatmap,
                    'kept_features': kept,
                    'num_forwards': forwards
                })
                print(f"  ✓ Found {len(kept)} minimal features in {forwards} forward passes")
            else:
                print(f"  ✗ Failed to generate explanation")
        
        return results


class MultiClassMinimalChannelsCNN(MinimalChannelsCNN):
    """Extended CNN explainer using Relative Minimal Sufficiency."""
    
    def __init__(self, model, target_layer, device="cuda", target_class=None,
                 retention_ratio=0.7, min_absolute_prob=0.01):
        """
        Args:
            retention_ratio: α ∈ (0,1]. Keep this fraction of baseline probability.
            min_absolute_prob: Absolute minimum probability threshold.
        """
        super().__init__(model, target_layer, device)
        self.target_class = target_class
        self.alpha = retention_ratio
        self.min_absolute = min_absolute_prob
    
    def find_minimal(self, x, min_channels=1, verbose=False, max_iterations=100, early_stop_size=None,
                     initial_partition_n: int = None):
        """Find minimal α-sufficient channels for the target class."""
        x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
        
        with torch.no_grad():
            baseline_logits = self.model(x)
        
        baseline_probs = F.softmax(baseline_logits[0], dim=0)
        
        if self.target_class is not None:
            self._baseline_class = self.target_class
            self._baseline_prob = baseline_probs[self.target_class].item()
        else:
            self._baseline_class = int(torch.argmax(baseline_probs))
            self._baseline_prob = baseline_probs[self._baseline_class].item()
        
        target_prob = self.alpha * self._baseline_prob
        
        act = self._capture_activation(x)
        num_channels = act.shape[1]
        
        if verbose:
            print(f"[CNN] Target layer has {num_channels} channels")
            print(f"[CNN] Target class: {self._baseline_class}, prob={self._baseline_prob:.4f}")
            print(f"[CNN] Retention ratio α: {self.alpha}")
            print(f"[CNN] Target probability: ≥ {target_prob:.4f} (α * baseline)")
            print(f"[CNN] Absolute minimum: ≥ {self.min_absolute:.4f}")
            print(f"[CNN] Max iterations: {max_iterations}, Early stop size: {early_stop_size}")
        
        forwards = 2
        
        cache = {}
        
        def sufficient_test(keep):
            nonlocal forwards
            
            keep_key = frozenset(keep)
            if keep_key in cache:
                prob_target, is_sufficient = cache[keep_key]
                if verbose:
                    retention = prob_target / max(self._baseline_prob, 1e-8)
                    print(f"  |S|={len(keep):3d} | "
                          f"p={prob_target:.4f} | "
                          f"retention={retention:.3f} | "
                          f"sufficient={is_sufficient} [cached]")
                return is_sufficient
            
            if forwards >= max_iterations:
                return False
            
            forwards += 1
            
            if len(keep) < min_channels:
                cache[keep_key] = (0.0, False)
                return False
            
            logits = self._hook_and_mask_forward(x, keep)
            probs = F.softmax(logits[0], dim=0)
            prob_target = float(probs[self._baseline_class].item())
            
            retention = prob_target / max(self._baseline_prob, 1e-8)
            
            is_sufficient = (
                retention >= self.alpha and
                prob_target >= self.min_absolute
            )
            
            cache[keep_key] = (prob_target, is_sufficient)
            
            if verbose:
                print(f"  |S|={len(keep):3d} | "
                      f"p={prob_target:.4f} | "
                      f"retention={retention:.3f} | "
                      f"sufficient={is_sufficient} | "
                      f"passes={forwards}")
            
            return is_sufficient
        
        universe = list(range(num_channels))
        all_sufficient = sufficient_test(universe)
        
        if not all_sufficient:
            print(f"[WARN] Full model doesn't meet relative sufficiency!")
            print(f"  Baseline prob: {self._baseline_prob:.4f}")
            print(f"  Target prob (α * baseline): {target_prob:.4f}")
            print(f"  This may indicate the class is too weak or α is too high.")
            print(f"[INFO] Trying greedy approach to find best subset...")
            
            kept = []
            
            if forwards < max_iterations - 20:
                channel_scores = {}
                num_to_score = min(100, num_channels)
                channels_to_score = list(range(num_to_score))
                
                if verbose:
                    print(f"[INFO] Pre-scoring {num_to_score} channels...")
                
                for ch in channels_to_score:
                    if forwards >= max_iterations - 10:
                        break
                    trial = [ch]
                    logits = self._hook_and_mask_forward(x, trial)
                    probs = F.softmax(logits[0], dim=0)
                    prob = float(probs[self._baseline_class].item())
                    channel_scores[ch] = prob
                    forwards += 1
                
                sorted_channels = sorted(channel_scores.items(), key=lambda x: x[1], reverse=True)
                remaining = [ch for ch in range(num_channels) if ch not in [k for k, _ in sorted_channels[:20]]]
                remaining = [ch for k, _ in sorted_channels[:20]] + remaining
                
                best_prob = 0.0
                max_greedy_iter = min(30, num_channels)
                
                for i, ch in enumerate(remaining[:max_greedy_iter]):
                    if forwards >= max_iterations:
                        break
                    trial = kept + [ch]
                    
                    trial_key = frozenset(trial)
                    if trial_key in cache:
                        prob, _ = cache[trial_key]
                    else:
                        logits = self._hook_and_mask_forward(x, trial)
                        probs = F.softmax(logits[0], dim=0)
                        prob = float(probs[self._baseline_class].item())
                        cache[trial_key] = (prob, False)
                        forwards += 1
                    
                    if prob > best_prob:
                        kept.append(ch)
                        best_prob = prob
                        if verbose:
                            print(f"  Added channel {ch}, |S|={len(kept)}, prob={best_prob:.4f}")
                        
                        if sufficient_test(kept):
                            if verbose:
                                print(f"  ✓ Reached sufficiency with {len(kept)} channels")
                            break
            else:
                kept = []
            
            if not sufficient_test(kept) and verbose:
                print(f"  Could not reach sufficiency, keeping {len(kept)} best channels")
        else:
            kept = ddmin_optimized(
                universe, sufficient_test, verbose=verbose,
                max_iterations=max_iterations,
                early_stop_size=early_stop_size,
                forwards_counter=lambda: forwards,
                initial_partition_n=initial_partition_n,
            )
        
        logits_final = self._hook_and_mask_forward(x, kept)
        probs_final = F.softmax(logits_final[0], dim=0)
        final_prob = float(probs_final[self._baseline_class].item())
        retention = final_prob / max(self._baseline_prob, 1e-8)
        
        if verbose:
            print(f"[CNN] Final: {len(kept)} channels, {forwards} forward passes")
            print(f"[CNN] Final probability: {final_prob:.4f}")
            print(f"[CNN] Retention: {retention:.3f} ({retention*100:.1f}% of baseline)")
        
        return kept, forwards, final_prob, retention


class MultiClassMinimalPatchesViT(MinimalPatchesViT):
    """Extended ViT explainer using Relative Minimal Sufficiency."""
    
    def __init__(self, model, device="cuda", strategy="hybrid",
                 retention_ratio=0.7, min_absolute_prob=0.01,
                 ablation_method="zero"):
        """
        Args:
            retention_ratio: α ∈ (0,1]. Keep this fraction of baseline probability.
            min_absolute_prob: Absolute minimum probability threshold.
        """
        super().__init__(
            model, device=device, strategy=strategy,
            min_prob=0.01, min_margin=0.001, rel_baseline=0.5,
            ablation_method=ablation_method, simple_mode=False
        )
        self.alpha = retention_ratio
        self.min_absolute = min_absolute_prob
        self._last_grid = None
    
    def _test_sufficient(
        self, 
        x: torch.Tensor,
        keep_indices: List[int],
        verbose: bool = False
    ) -> bool:
        """Relative Sufficiency Test for ViT patches."""
        if len(keep_indices) == 0:
            return False
        
        logits = self.adapter.forward_with_masked_patches(
            x, keep_indices, self.ablation_method
        )
        probs = F.softmax(logits[0], dim=0)
        prob_target = float(probs[self._baseline_class].item())
        
        retention = prob_target / max(self._baseline_prob, 1e-8)
        
        is_sufficient = (
            retention >= self.alpha and
            prob_target >= self.min_absolute
        )
        
        if verbose:
            print(f"  |S|={len(keep_indices):3d} | "
                  f"p={prob_target:.4f} | "
                  f"retention={retention:.3f} | "
                  f"sufficient={is_sufficient}")
        
        return is_sufficient
    
    def find_minimal(
        self,
        x: torch.Tensor,
        verbose: bool = False,
        initial_partition_n: int = None
    ) -> Tuple[List[int], int, float, float]:
        """Find minimal α-sufficient patches for target class."""
        x = x.unsqueeze(0).to(self.device) if x.dim() == 3 else x.to(self.device)
        
        gh, gw = self.adapter.get_grid_size(x.shape[-2], x.shape[-1])
        n_patches = gh * gw
        self._last_grid = (gh, gw)
        
        target_prob = self.alpha * self._baseline_prob
        
        if verbose:
            print(f"[ViT] Grid: {gh}x{gw} = {n_patches} patches")
            print(f"[ViT] Target class: {self._baseline_class}, prob={self._baseline_prob:.4f}")
            print(f"[ViT] Retention ratio α: {self.alpha}")
            print(f"[ViT] Target probability: ≥ {target_prob:.4f} (α * baseline)")
            print(f"[ViT] Absolute minimum: ≥ {self.min_absolute:.4f}")
            print(f"[ViT] Strategy: {self.strategy}")
        
        forwards = 1
        
        def sufficient_test(keep: List[int]) -> bool:
            nonlocal forwards
            forwards += 1
            return self._test_sufficient(x, keep, verbose=verbose)
        
        universe = list(range(n_patches))
        all_sufficient = sufficient_test(universe)
        
        if not all_sufficient:
            print(f"[WARN] Full model doesn't meet relative sufficiency!")
            print(f"  Baseline prob: {self._baseline_prob:.4f}")
            print(f"  Target prob (α * baseline): {target_prob:.4f}")
            print(f"  This may indicate the class is too weak or α is too high.")
        
        if self.strategy == "dd":
            kept = ddmin_optimized(
                universe, sufficient_test, verbose=verbose,
                max_iterations=200, early_stop_size=None,
                forwards_counter=lambda: forwards,
                initial_partition_n=initial_partition_n,
            )
            
        elif self.strategy == "grow":
            kept = []
            remaining = list(range(n_patches))
            
            while remaining and not sufficient_test(kept):
                best_patch = None
                best_prob = -1
                
                for patch in remaining[:min(50, len(remaining))]:
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
        
        logits_final = self.adapter.forward_with_masked_patches(
            x, kept, self.ablation_method
        )
        probs_final = F.softmax(logits_final[0], dim=0)
        final_prob = float(probs_final[self._baseline_class].item())
        retention = final_prob / max(self._baseline_prob, 1e-8)
        
        if verbose:
            print(f"[ViT] Final: {len(kept)} patches, {forwards} forward passes")
            print(f"[ViT] Final probability: {final_prob:.4f}")
            print(f"[ViT] Retention: {retention:.3f} ({retention*100:.1f}% of baseline)")
        
        return kept, forwards, final_prob, retention


def load_model_and_layer(model_name, device):
    """Load model and identify target layer."""
    if model_name == "vgg11":
        model = models.vgg11(weights=models.VGG11_Weights.DEFAULT)
        target_layer = model.features[-2]  # Last conv layer
        is_vit = False
    elif model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        target_layer = model.features[-2]
        is_vit = False
    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        target_layer = model.features[-2]
        is_vit = False
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        target_layer = model.layer4[-1].conv3
        is_vit = False
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        target_layer = model.encoder.layers[-1].ln_1
        is_vit = True
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.eval()
    model.to(device)
    
    return model, target_layer, is_vit


def visualize_multi_class_results(image_path, results, labels=None, output_path="multi_class_explanation.png"):
    """Visualize explanations for multiple classes side by side."""
    # Load original image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img.resize((224, 224)))
    
    # Load the preprocessed tensor for overlay
    x = normalize(img)
    
    # Create figure
    n_classes = len(results)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(5 * (n_classes + 1), 5))
    
    if n_classes == 0:
        axes[0].text(0.5, 0.5, 'No results', ha='center', va='center')
        axes[0].axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        return
    
    # Original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Each class explanation
    for i, result in enumerate(results):
        cls_idx = result['class_idx']
        prob = result['probability']
        heatmap = result['heatmap']
        n_features = len(result['kept_features'])
        n_forwards = result['num_forwards']
        
        # Create overlay
        overlay = overlay_heatmap_on_tensor(x, heatmap, alpha=0.5)
        
        axes[i + 1].imshow(overlay)
        
        # Get class label if available
        if labels and cls_idx < len(labels):
            class_name = labels[cls_idx]
            title = f"{class_name}\n(Class {cls_idx})"
        else:
            title = f"Class {cls_idx}"
        
        title += f"\nProb: {prob:.3f}"
        title += f"\nFeatures: {n_features}"
        title += f"\nPasses: {n_forwards}"
        
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    plt.close()



