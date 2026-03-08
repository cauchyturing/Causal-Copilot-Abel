"""Standard benchmark datasets for causal discovery evaluation.

Provides well-known DAGs (Sachs, Asia) and a synthetic ALARM-scale graph,
each with linear Gaussian data generated from the ground-truth structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class StandardDataset:
    """A benchmark dataset with known ground truth."""

    name: str
    description: str
    data: pd.DataFrame
    ground_truth: np.ndarray
    columns: list[str]
    source: str


def _generate_from_dag(
    adj: np.ndarray,
    columns: list[str],
    n: int = 1000,
    seed: int = 42,
    weight_range: tuple[float, float] = (0.3, 0.8),
    noise_std: float = 0.3,
) -> pd.DataFrame:
    """Generate linear Gaussian data from a DAG adjacency matrix.

    Convention: adj[i, j] = 1 means j -> i (column causes row).
    Data is generated in topological order.
    """
    rng = np.random.default_rng(seed)
    p = adj.shape[0]

    # Assign deterministic edge weights from the adjacency
    weight_rng = np.random.default_rng(seed + 1)
    weights = np.zeros_like(adj, dtype=float)
    for i in range(p):
        for j in range(p):
            if adj[i, j] > 0:
                w = weight_rng.uniform(weight_range[0], weight_range[1])
                weights[i, j] = w

    # Topological sort (Kahn's algorithm)
    in_degree = (adj > 0).astype(int).sum(axis=1)  # number of parents per node
    queue = [i for i in range(p) if in_degree[i] == 0]
    order = []
    remaining = in_degree.copy()
    while queue:
        node = queue.pop(0)
        order.append(node)
        # Find children: adj[child, node] = 1
        for child in range(p):
            if adj[child, node] > 0:
                remaining[child] -= 1
                if remaining[child] == 0:
                    queue.append(child)

    assert len(order) == p, "Graph has a cycle — not a valid DAG"

    # Generate data in topological order
    data = np.zeros((n, p))
    for node in order:
        noise = rng.normal(0, noise_std, size=n)
        parents = np.where(adj[node] > 0)[0]
        if len(parents) > 0:
            data[:, node] = data[:, parents] @ weights[node, parents] + noise
        else:
            data[:, node] = rng.normal(0, 1, size=n)

    return pd.DataFrame(data, columns=columns)


def _build_sachs() -> StandardDataset:
    """Sachs et al. 2005 — 11 proteins, 17 edges."""
    columns = [
        "Raf", "Mek", "PLCg", "PIP2", "PIP3",
        "Erk", "Akt", "PKA", "PKC", "JNK", "P38",
    ]
    col_idx = {c: i for i, c in enumerate(columns)}
    n = len(columns)
    adj = np.zeros((n, n), dtype=int)

    # Edges: source -> target means adj[target, source] = 1
    # 17-edge consensus network from Sachs et al. 2005, Science 308(5721),
    # Fig. 3A.  Note: the bnlearn repository version differs (has Akt→Erk
    # instead of PIP3→Akt and PKA→Akt).  We follow the original paper.
    edges = [
        ("PLCg", "PIP2"), ("PLCg", "PIP3"), ("PIP3", "PIP2"),
        ("PIP3", "Akt"),
        ("PKC", "PKA"), ("PKC", "Raf"), ("PKC", "Mek"),
        ("PKC", "JNK"), ("PKC", "P38"),
        ("PKA", "Raf"), ("PKA", "Mek"), ("PKA", "Erk"),
        ("PKA", "Akt"), ("PKA", "JNK"), ("PKA", "P38"),
        ("Raf", "Mek"), ("Mek", "Erk"),
    ]
    for src, tgt in edges:
        adj[col_idx[tgt], col_idx[src]] = 1

    data = _generate_from_dag(adj, columns, n=1000, seed=42)
    return StandardDataset(
        name="sachs",
        description="Sachs et al. 2005 — 11-protein signaling network (17 edges)",
        data=data,
        ground_truth=adj,
        columns=columns,
        source="Sachs, K. et al. (2005). Causal protein-signaling networks derived from multiparameter single-cell data. Science 308(5721):523-529.",
    )


def _build_asia() -> StandardDataset:
    """Lauritzen & Spiegelhalter 1988 — 8 nodes, 8 edges."""
    columns = [
        "Asia", "Tuberculosis", "Smoking", "LungCancer",
        "TBorCancer", "Bronchitis", "XRay", "Dyspnea",
    ]
    col_idx = {c: i for i, c in enumerate(columns)}
    n = len(columns)
    adj = np.zeros((n, n), dtype=int)

    edges = [
        ("Asia", "Tuberculosis"),
        ("Smoking", "LungCancer"),
        ("Smoking", "Bronchitis"),
        ("Tuberculosis", "TBorCancer"),
        ("LungCancer", "TBorCancer"),
        ("TBorCancer", "XRay"),
        ("TBorCancer", "Dyspnea"),
        ("Bronchitis", "Dyspnea"),
    ]
    for src, tgt in edges:
        adj[col_idx[tgt], col_idx[src]] = 1

    data = _generate_from_dag(adj, columns, n=1000, seed=43)
    return StandardDataset(
        name="asia",
        description="Lauritzen & Spiegelhalter 1988 — 8-node chest clinic model (8 edges)",
        data=data,
        ground_truth=adj,
        columns=columns,
        source="Lauritzen, S.L. & Spiegelhalter, D.J. (1988). Local computations with probabilities on graphical structures and their application to expert systems. JRSS-B 50(2):157-194.",
    )


def _build_alarm() -> StandardDataset:
    """Synthetic sparse DAG — 37 nodes, 46 edges (ALARM-scale).

    NOTE: This is NOT the real ALARM network from Beinlich et al. 1989.
    It is a random sparse DAG of the same scale (37 nodes, 46 edges),
    useful for testing scalability of causal discovery algorithms.
    """
    columns = [f"V{i}" for i in range(37)]
    n = len(columns)
    adj = np.zeros((n, n), dtype=int)

    # Deterministic sparse DAG using a fixed seed
    rng = np.random.default_rng(1989)

    edges_added = 0
    target_edges = 46

    # Edge j -> i only if j < i (topological ordering by index)
    candidate_edges = []
    for i in range(1, n):
        for j in range(i):
            candidate_edges.append((j, i))

    rng.shuffle(candidate_edges)

    for src, tgt in candidate_edges:
        if edges_added >= target_edges:
            break
        adj[tgt, src] = 1  # adj[i,j]=1 means j->i
        edges_added += 1

    data = _generate_from_dag(adj, columns, n=1000, seed=44)
    return StandardDataset(
        name="alarm",
        description="Synthetic sparse DAG — 37 nodes, 46 edges (ALARM-scale, not the real ALARM network)",
        data=data,
        ground_truth=adj,
        columns=columns,
        source="Synthetic random DAG at ALARM scale. See Beinlich et al. 1989 for the real ALARM network.",
    )


# Pre-build all datasets at import time (deterministic, fast)
STANDARD_DATASETS: dict[str, StandardDataset] = {
    "sachs": _build_sachs(),
    "asia": _build_asia(),
    "alarm": _build_alarm(),
}
