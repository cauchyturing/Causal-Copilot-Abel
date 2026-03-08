"""Evaluation metrics for causal discovery benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DiscoveryMetrics:
    """Metrics comparing discovered adjacency to ground truth."""

    # Edge-level (directed)
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    fdr: float
    shd: int  # Structural Hamming Distance
    # Skeleton-level (undirected)
    skeleton_tp: int
    skeleton_fp: int
    skeleton_fn: int
    skeleton_precision: float
    skeleton_recall: float
    skeleton_f1: float
    # Orientation
    orientation_accuracy: float

    def to_dict(self) -> dict:
        """Return all metrics as a plain dict."""
        return asdict(self)


def _skeleton(adj: np.ndarray) -> np.ndarray:
    """Convert directed adjacency to undirected upper-triangle matrix.

    Any edge (i,j) or (j,i) becomes a 1 in position (min(i,j), max(i,j)).
    """
    sym = ((adj > 0) | (adj.T > 0)).astype(int)
    return np.triu(sym, k=1)


def evaluate_adjacency(predicted: np.ndarray, ground_truth: np.ndarray) -> DiscoveryMetrics:
    """Compare predicted adjacency matrix against ground truth.

    Both matrices use convention: mat[i,j]=1 means j->i.
    Predicted values >0 are treated as edges (handles 1/2/3 encoding).
    """
    pred_binary = (predicted > 0).astype(int)
    gt_binary = (ground_truth > 0).astype(int)

    # --- Directed (edge-level) metrics ---
    tp = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
    fp = int(np.sum((pred_binary == 1) & (gt_binary == 0)))
    fn = int(np.sum((pred_binary == 0) & (gt_binary == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    # SHD = |edges in pred but not gt| + |edges in gt but not pred| + |reversed edges|
    shd = int(np.sum(pred_binary != gt_binary))

    # --- Skeleton (undirected) metrics ---
    skel_pred = _skeleton(pred_binary)
    skel_gt = _skeleton(gt_binary)

    skel_tp = int(np.sum((skel_pred == 1) & (skel_gt == 1)))
    skel_fp = int(np.sum((skel_pred == 1) & (skel_gt == 0)))
    skel_fn = int(np.sum((skel_pred == 0) & (skel_gt == 1)))

    skel_precision = skel_tp / (skel_tp + skel_fp) if (skel_tp + skel_fp) > 0 else 0.0
    skel_recall = skel_tp / (skel_tp + skel_fn) if (skel_tp + skel_fn) > 0 else 0.0
    skel_f1 = (
        2 * skel_precision * skel_recall / (skel_precision + skel_recall)
        if (skel_precision + skel_recall) > 0
        else 0.0
    )

    # --- Orientation accuracy ---
    # Of edges that are correct in the skeleton, how many have the correct direction?
    # For each undirected edge present in both skeletons, check if the directed
    # edges match exactly.
    shared_skeleton = (skel_pred == 1) & (skel_gt == 1)
    if np.sum(shared_skeleton) > 0:
        correct_orientation = 0
        total_shared = 0
        rows, cols = np.where(shared_skeleton)
        for r, c in zip(rows, cols):
            total_shared += 1
            # Check if directed edges between r and c match in both directions
            if pred_binary[r, c] == gt_binary[r, c] and pred_binary[c, r] == gt_binary[c, r]:
                correct_orientation += 1
        orientation_accuracy = correct_orientation / total_shared
    else:
        orientation_accuracy = 0.0

    return DiscoveryMetrics(
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        fdr=fdr,
        shd=shd,
        skeleton_tp=skel_tp,
        skeleton_fp=skel_fp,
        skeleton_fn=skel_fn,
        skeleton_precision=skel_precision,
        skeleton_recall=skel_recall,
        skeleton_f1=skel_f1,
        orientation_accuracy=orientation_accuracy,
    )
