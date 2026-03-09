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

    Both matrices use convention: mat[i,j]=1 means j->i (directed),
    2=undirected, 3=bidirected.

    Directed edge metrics use exact cell comparison: pred[i,j] must equal
    gt[i,j] for a true positive. An undirected prediction (2) where ground
    truth has a directed edge (1) counts as both FP and FN — the predicted
    edge type is wrong and the correct type is missing.

    Skeleton metrics ignore direction (any nonzero value counts as an edge).
    Orientation accuracy measures how many skeleton-correct edges also have
    the correct direction.
    """
    pred_has_edge = (predicted > 0).astype(int)
    gt_has_edge = (ground_truth > 0).astype(int)

    # --- Directed (edge-level) metrics ---
    # TP: edge exists in both AND has same type
    tp = int(np.sum((pred_has_edge == 1) & (gt_has_edge == 1) & (predicted == ground_truth)))
    # FP: edge in pred but not in gt
    fp_extra = int(np.sum((pred_has_edge == 1) & (gt_has_edge == 0)))
    # FP also includes wrong edge type (e.g. undirected where directed expected)
    fp_wrong_type = int(np.sum((pred_has_edge == 1) & (gt_has_edge == 1) & (predicted != ground_truth)))
    fp = fp_extra + fp_wrong_type
    # FN: edge in gt but not in pred, plus missed correct type
    fn_missing = int(np.sum((pred_has_edge == 0) & (gt_has_edge == 1)))
    fn = fn_missing + fp_wrong_type

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fdr = fp / (fp + tp) if (fp + tp) > 0 else 0.0

    # --- Skeleton (undirected) metrics ---
    skel_pred = _skeleton(pred_has_edge)
    skel_gt = _skeleton(gt_has_edge)

    skel_tp = int(np.sum((skel_pred == 1) & (skel_gt == 1)))
    skel_fp = int(np.sum((skel_pred == 1) & (skel_gt == 0)))
    skel_fn = int(np.sum((skel_pred == 0) & (skel_gt == 1)))

    skel_precision = skel_tp / (skel_tp + skel_fp) if (skel_tp + skel_fp) > 0 else 0.0
    skel_recall = skel_tp / (skel_tp + skel_fn) if (skel_tp + skel_fn) > 0 else 0.0
    skel_f1 = (
        2 * skel_precision * skel_recall / (skel_precision + skel_recall) if (skel_precision + skel_recall) > 0 else 0.0
    )

    # --- Orientation accuracy + SHD ---
    # SHD (Structural Hamming Distance) = extra + missing + reversed at
    # skeleton level.  A reversed edge counts as ONE edit, not two.
    # Orientation accuracy = fraction of shared-skeleton edges with correct
    # directed representation (compares actual cell values, not binary).
    shared_skeleton = (skel_pred == 1) & (skel_gt == 1)
    reversed_edges = 0
    correct_orientation = 0
    total_shared = 0
    if np.sum(shared_skeleton) > 0:
        rows, cols = np.where(shared_skeleton)
        for r, c in zip(rows, cols, strict=True):
            total_shared += 1
            if predicted[r, c] == ground_truth[r, c] and predicted[c, r] == ground_truth[c, r]:
                correct_orientation += 1
            else:
                reversed_edges += 1
        orientation_accuracy = correct_orientation / total_shared
    else:
        orientation_accuracy = 0.0

    shd = skel_fp + skel_fn + reversed_edges

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
