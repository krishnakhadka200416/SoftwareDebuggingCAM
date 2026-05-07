"""Sufficient Input Subsets (Carter et al., 2018), PyTorch / image-friendly port.

Adapted from the reference implementation at
https://github.com/google-research/google-research/tree/master/sufficient_input_subsets
(Apache 2.0).

Differences from the reference:
  - We mask at the level of a coarse spatial patch grid (default 7x7 = 49
    patches of 32x32 pixels each on 224x224 inputs). This is the standard
    granularity used in image SIS experiments and is a closer apples-to-apples
    comparison to LIME/RISE/extremal-perturbation than per-pixel masking.
  - The "function f" wraps a PyTorch model and returns the predicted-class
    softmax probability (a scalar per input). Masking replaces patch pixels
    with the masking value (default 0 in the normalized input space, matching
    DD-CAM's masking convention).
  - Backselect's per-iteration "test all unmasked positions" is computed in a
    single batched forward pass per iteration for efficiency.

Public API:
    sis_collection_image(model, x, fully_masked_input, threshold, target_class,
                         grid_h, grid_w)    -> list of SISResult-like dicts
    backselect_image(model, x, fully_masked_input, target_class,
                     grid_h, grid_w)        -> list of (patch_idx, score) tuples
    backselect_to_heatmap(backselect_stack, grid_h, grid_w, image_h, image_w)
                                            -> (image_h, image_w) float heatmap

`grid_h` and `grid_w` define the spatial mask grid; the input is divided into
grid_h x grid_w patches that are masked together.
"""

from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _patch_mask_to_pixel_mask(patch_mask: np.ndarray, image_h: int, image_w: int) -> np.ndarray:
    """Upsample a (grid_h, grid_w) boolean patch mask to a (image_h, image_w) pixel mask."""
    gh, gw = patch_mask.shape
    ph, pw = image_h // gh, image_w // gw
    pixel = np.zeros((image_h, image_w), dtype=bool)
    for i in range(gh):
        for j in range(gw):
            if patch_mask[i, j]:
                pixel[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw] = True
    return pixel


def _build_masked_inputs(
    x: torch.Tensor,                      # (C, H, W) - the original normalized image
    fully_masked_input: torch.Tensor,     # (C, H, W) - the masked-value image (typically zeros)
    patch_masks: np.ndarray,              # (B, gh, gw) bool, True = visible (unmasked)
) -> torch.Tensor:
    """Apply each of B patch masks to x, falling back to fully_masked_input where False."""
    B, gh, gw = patch_masks.shape
    C, H, W = x.shape
    out = torch.empty(B, C, H, W, dtype=x.dtype, device=x.device)
    for b in range(B):
        pixel_mask = _patch_mask_to_pixel_mask(patch_masks[b], H, W)  # (H, W) bool
        m = torch.from_numpy(pixel_mask).to(x.device)
        out[b] = torch.where(m.unsqueeze(0), x, fully_masked_input)
    return out


@torch.no_grad()
def _predicted_prob(
    model: torch.nn.Module,
    x_batch: torch.Tensor,                # (B, C, H, W)
    target_class: int,
    batch_size: int = 32,
    score: str = "softmax",                # "softmax" (default) or "sigmoid"
) -> np.ndarray:
    """Return target_class probability for each input in the batch.

    `score`:
      - "softmax" (default): F.softmax(logits, dim=1)[:, target_class]
        (correct for single-label classifiers, e.g. ImageNet ResNet-50).
      - "sigmoid": torch.sigmoid(logits)[:, target_class]
        (correct for multi-label classifiers, e.g. torchxrayvision DenseNet-121).
    """
    out = []
    model.eval()
    for i in range(0, x_batch.shape[0], batch_size):
        chunk = x_batch[i : i + batch_size]
        logits = model(chunk)
        # Some torchvision models return tuples (e.g., InceptionOutputs in train mode);
        # in eval mode they return a tensor.
        if isinstance(logits, tuple):
            logits = logits[0]
        if score == "sigmoid":
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)
        out.append(probs[:, target_class].detach().cpu().numpy())
    return np.concatenate(out)


def backselect_image(
    model: torch.nn.Module,
    x: torch.Tensor,                       # (C, H, W) - the original normalized image
    fully_masked_input: torch.Tensor,      # (C, H, W) - the masked-value image
    target_class: int,
    grid_h: int = 7,
    grid_w: int = 7,
    batch_size: int = 32,
    score: str = "softmax",                # "softmax" or "sigmoid"
) -> List[Tuple[Tuple[int, int], float]]:
    """Backward selection over a (grid_h, grid_w) patch grid.

    At each iteration, test masking each remaining unmasked patch; mask the patch
    whose removal preserves the prediction the most, record its index and the
    resulting predicted-class probability, repeat until all patches are masked.
    Returns the backselect stack: [(patch_idx_tuple, prob_after_masking), ...].
    """
    n_patches = grid_h * grid_w
    visible = np.ones((grid_h, grid_w), dtype=bool)  # True = unmasked
    stack: List[Tuple[Tuple[int, int], float]] = []

    while visible.any():
        candidate_idxs = np.argwhere(visible)              # (R, 2) row,col of remaining patches
        R = candidate_idxs.shape[0]
        # Build a batch of R masks: each masks one of the remaining patches.
        cand_masks = np.broadcast_to(visible, (R, grid_h, grid_w)).copy()
        for r in range(R):
            i, j = candidate_idxs[r]
            cand_masks[r, i, j] = False                     # mask this patch

        masked_inputs = _build_masked_inputs(x, fully_masked_input, cand_masks)
        scores = _predicted_prob(model, masked_inputs, target_class, batch_size, score=score)

        # The optimal-to-mask is the one whose masking *retains the highest*
        # prediction (minimum loss). Argmax in the SIS paper.
        best_r = int(np.argmax(scores))
        best_idx = (int(candidate_idxs[best_r, 0]), int(candidate_idxs[best_r, 1]))
        best_score = float(scores[best_r])
        stack.append((best_idx, best_score))
        visible[best_idx[0], best_idx[1]] = False

    return stack


def find_sis_from_backselect(
    backselect_stack: List[Tuple[Tuple[int, int], float]],
    threshold: float,
) -> List[Tuple[int, int]]:
    """SIS = the patches masked LAST in backselect, before the score first
    drops below the threshold (read the stack in reverse).
    """
    sis: List[Tuple[int, int]] = []
    for idx, val in reversed(backselect_stack):
        if val >= threshold:
            break
        sis.append(idx)
    return sis


def backselect_to_heatmap(
    backselect_stack: List[Tuple[Tuple[int, int], float]],
    grid_h: int, grid_w: int,
    image_h: int, image_w: int,
) -> np.ndarray:
    """Convert the full backselect ordering into a continuous (image_h, image_w) heatmap.

    Patches masked LATER have higher importance. Used as a fallback when no SIS
    is found at the chosen threshold. Returns a heatmap covering all patches.
    """
    D = grid_h * grid_w
    importance = np.zeros((grid_h, grid_w), dtype=np.float32)
    for rank, (idx, _val) in enumerate(backselect_stack):
        importance[idx[0], idx[1]] = rank / max(D - 1, 1)  # in [0, 1]

    ph, pw = image_h // grid_h, image_w // grid_w
    pixel = np.zeros((image_h, image_w), dtype=np.float32)
    for i in range(grid_h):
        for j in range(grid_w):
            pixel[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw] = importance[i, j]
    return pixel


def sis_collection_to_heatmap(
    backselect_stack: List[Tuple[Tuple[int, int], float]],
    sis_idxs: List[Tuple[int, int]],
    grid_h: int, grid_w: int,
    image_h: int, image_w: int,
) -> np.ndarray:
    """SIS-faithful heatmap (Carter et al. 2018, Section 4 visualization convention).

    Highlights ONLY the patches in the SIS subset, shaded by backselect order
    within the SIS (later-masked ⇒ darker/higher value, matching the paper:
    "Depictions of each SIS are shaded based on the feature order in R
    (darker = later), which can indicate relative feature importance within the SIS.").

    Patches not in the SIS subset get value 0 (no color when overlaid).
    """
    importance = np.zeros((grid_h, grid_w), dtype=np.float32)
    if sis_idxs:
        # The backselect stack lists patches from first-masked (least important) to last-masked (most important).
        # Patches in the SIS were among the LAST to be masked, i.e. they appear toward the END of the stack.
        # We shade each SIS patch by its rank in the backselect order, normalized within the SIS itself.
        sis_set = set(tuple(i) for i in sis_idxs)
        # Get the ranks of SIS patches in the backselect stack (last-masked has highest rank).
        sis_ranks = []
        for rank, (idx, _val) in enumerate(backselect_stack):
            if tuple(idx) in sis_set:
                sis_ranks.append((tuple(idx), rank))
        if sis_ranks:
            min_rank = min(r for _, r in sis_ranks)
            max_rank = max(r for _, r in sis_ranks)
            for (idx, rank) in sis_ranks:
                if max_rank > min_rank:
                    importance[idx[0], idx[1]] = (rank - min_rank) / (max_rank - min_rank)
                else:
                    importance[idx[0], idx[1]] = 1.0
            # Floor non-zero values so the lowest-importance SIS patch is still visible.
            for (idx, _) in sis_ranks:
                if importance[idx[0], idx[1]] < 0.3:
                    importance[idx[0], idx[1]] = max(importance[idx[0], idx[1]], 0.3)

    ph, pw = image_h // grid_h, image_w // grid_w
    pixel = np.zeros((image_h, image_w), dtype=np.float32)
    for i in range(grid_h):
        for j in range(grid_w):
            pixel[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw] = importance[i, j]
    return pixel


def sis_collection_image_full(
    model: torch.nn.Module,
    x: torch.Tensor,
    fully_masked_input: torch.Tensor,
    threshold: float,
    target_class: int,
    grid_h: int = 7,
    grid_w: int = 7,
    batch_size: int = 32,
    score: str = "softmax",
):
    """Find the FULL SIS-collection (multiple disjoint SIS subsets) following
    Carter et al. 2018 SIScollection algorithm. Returns:

      backselect_stacks: list of backselect_stack from each iteration
      sis_collection:    list of SIS index lists (one per iteration)
      union_idxs:        flat list of all patch indices in the union of SIS subsets
    """
    # Currently-unmasked patches start as all True
    visible = np.ones((grid_h, grid_w), dtype=bool)
    backselect_stacks = []
    sis_collection = []
    current_input = x

    while visible.any():
        # If the prediction on currently-unmasked features is below threshold, no more SIS exist.
        # Build current masked input from `visible`.
        gh, gw = visible.shape
        ph, pw = x.shape[-2] // gh, x.shape[-1] // gw
        pixel_mask = np.zeros((x.shape[-2], x.shape[-1]), dtype=bool)
        for i in range(gh):
            for j in range(gw):
                if visible[i, j]:
                    pixel_mask[i * ph:(i + 1) * ph, j * pw:(j + 1) * pw] = True
        m = torch.from_numpy(pixel_mask).to(x.device)
        masked_x = torch.where(m.unsqueeze(0), x, fully_masked_input).unsqueeze(0)
        score_now = float(_predicted_prob(model, masked_x, target_class, batch_size=1, score=score)[0])
        if score_now < threshold:
            break

        # Backselect over the currently-visible patches
        # We need a backselect that respects which patches are already masked.
        # Reuse backselect_image with a custom starting visible mask.
        stack = _backselect_with_visible(model, x, fully_masked_input, target_class,
                                          grid_h, grid_w, visible.copy(), batch_size, score)
        if not stack:
            break
        backselect_stacks.append(stack)
        sis_idxs = find_sis_from_backselect(stack, threshold)
        if not sis_idxs:
            break
        sis_collection.append(sis_idxs)
        # Mark these patches as masked for next iteration
        for (i, j) in sis_idxs:
            visible[i, j] = False

    union_idxs = [idx for sis in sis_collection for idx in sis]
    return backselect_stacks, sis_collection, union_idxs


def _backselect_with_visible(
    model: torch.nn.Module,
    x: torch.Tensor,
    fully_masked_input: torch.Tensor,
    target_class: int,
    grid_h: int, grid_w: int,
    visible: np.ndarray,
    batch_size: int = 32,
    score: str = "softmax",
) -> List[Tuple[Tuple[int, int], float]]:
    """Backselect starting from a custom visible mask (subset already masked)."""
    stack: List[Tuple[Tuple[int, int], float]] = []
    while visible.any():
        candidate_idxs = np.argwhere(visible)
        R = candidate_idxs.shape[0]
        cand_masks = np.broadcast_to(visible, (R, grid_h, grid_w)).copy()
        for r in range(R):
            i, j = candidate_idxs[r]
            cand_masks[r, i, j] = False
        masked_inputs = _build_masked_inputs(x, fully_masked_input, cand_masks)
        scores = _predicted_prob(model, masked_inputs, target_class, batch_size, score=score)
        best_r = int(np.argmax(scores))
        best_idx = (int(candidate_idxs[best_r, 0]), int(candidate_idxs[best_r, 1]))
        best_score = float(scores[best_r])
        stack.append((best_idx, best_score))
        visible[best_idx[0], best_idx[1]] = False
    return stack


def sis_collection_image(
    model: torch.nn.Module,
    x: torch.Tensor,
    fully_masked_input: torch.Tensor,
    threshold: float,
    target_class: int,
    grid_h: int = 7,
    grid_w: int = 7,
    batch_size: int = 32,
):
    """Convenience wrapper: run backselect, find first SIS, return both."""
    stack = backselect_image(model, x, fully_masked_input, target_class,
                             grid_h, grid_w, batch_size)
    sis_idxs = find_sis_from_backselect(stack, threshold)
    return stack, sis_idxs
