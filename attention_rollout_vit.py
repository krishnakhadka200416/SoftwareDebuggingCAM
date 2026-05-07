"""Attention Rollout (Abnar & Zuidema 2020) baseline for torchvision ViT models.

Adapted from jacobgil/vit-explain (https://github.com/jacobgil/vit-explain)
to work with torchvision's vit_b_16 / vit_b_32 by capturing attention weights
from each EncoderBlock's nn.MultiheadAttention via a forward wrapper.

Algorithm (verbatim from the reference):
  - For each transformer block, fuse multi-head attention (mean / max / min).
  - Drop the lowest discard_ratio attentions per head-fused row (preserving CLS).
  - Add residual: A = (A + I) / 2, normalize rows to sum to 1.
  - Cumulatively multiply across blocks.
  - Take the row of the CLS token, drop the CLS column, reshape to spatial grid.
  - Normalize to [0, 1] by dividing by max.

Then upsample the spatial grid to (224, 224) for compatibility with the
existing metrics pipeline.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


def _rollout(attentions: List[torch.Tensor], discard_ratio: float, head_fusion: str) -> np.ndarray:
    """Verbatim from the reference, modulo torch.eye device handling."""
    result = torch.eye(attentions[0].size(-1))
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise ValueError("head_fusion must be mean / max / min")

            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0 * I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)

    mask = result[0, 0, 1:]
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).numpy()
    mx = float(mask.max())
    if mx > 0:
        mask = mask / mx
    return mask


class TorchvisionViTAttentionRollout:
    """Attention rollout for torchvision ViTs.

    Captures attention weights from each EncoderBlock by replacing
    `block.self_attention.forward` with a wrapper that forces
    `need_weights=True` and `average_attn_weights=False`, then stores the
    weights tensor of shape (B, num_heads, seq_len, seq_len).
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean", discard_ratio: float = 0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        self._captured = {"attentions": []}

        # Replace each transformer block's self_attention forward with a capturing wrapper.
        self._patched = []
        for block in self.model.encoder.layers:
            mha = block.self_attention
            original_forward = mha.forward

            def make_wrapped(orig, capture):
                def wrapped(query, key, value, *args, **kwargs):
                    # Force need_weights=True and per-head weights.
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = False
                    out, attn = orig(query, key, value, *args, **kwargs)
                    if attn is not None:
                        capture["attentions"].append(attn.detach().cpu())
                    return out, attn
                return wrapped

            mha.forward = make_wrapped(original_forward, self._captured)
            self._patched.append((mha, original_forward))

    def restore(self):
        """Undo the monkey-patches."""
        for mha, original_forward in self._patched:
            mha.forward = original_forward
        self._patched = []

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Returns a (gh, gw) spatial mask in [0, 1]."""
        self._captured["attentions"] = []
        with torch.no_grad():
            _ = self.model(input_tensor)
        if not self._captured["attentions"]:
            raise RuntimeError("No attention weights captured. Check model architecture.")
        mask = _rollout(self._captured["attentions"], self.discard_ratio, self.head_fusion)
        return mask


def run_attention_rollout(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_size: int = 224,
    head_fusion: str = "mean",
    discard_ratio: float = 0.9,
) -> Tuple[np.ndarray, float]:
    """One-shot helper. Returns (heatmap of shape (target_size, target_size), elapsed_seconds).

    The attention-rollout module is constructed and torn down inside this call,
    so it does not leave the model patched after returning.
    """
    rollout = TorchvisionViTAttentionRollout(model, head_fusion=head_fusion,
                                             discard_ratio=discard_ratio)
    try:
        t0 = time.perf_counter()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        mask = rollout(input_tensor)
        elapsed = time.perf_counter() - t0
        # Upsample to target image resolution for compatibility with metrics expecting HxW maps.
        mask_up = cv2.resize(mask.astype(np.float32), (target_size, target_size),
                             interpolation=cv2.INTER_LINEAR)
        return mask_up, float(elapsed)
    finally:
        rollout.restore()


if __name__ == "__main__":
    # Tiny smoke test
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, 224, 224, device=device)
    mask, t = run_attention_rollout(model, x, target_size=224)
    print(f"shape={mask.shape}, dtype={mask.dtype}, min={mask.min():.4f}, max={mask.max():.4f}, mean={mask.mean():.4f}, t={t:.2f}s")
