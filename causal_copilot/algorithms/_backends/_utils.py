"""Shared utilities for causal-learn and other backend conversions."""

import numpy as np


def convert_causallearn_cpdag(adj_matrix: np.ndarray) -> np.ndarray:
    """Convert causal-learn's CPDAG encoding to our adjacency convention.

    causal-learn encodes edges as:
      adj[i,j]=1  and adj[j,i]=-1  =>  directed j -> i
      adj[i,j]=1  and adj[j,i]=1   =>  bidirected i <-> j
      adj[i,j]=-1 and adj[j,i]=-1  =>  undirected i -- j

    Our convention:
      mat[i,j]=1 => directed j -> i
      mat[i,j]=2 => undirected
      mat[i,j]=3 => bidirected
    """
    inferred_flat = np.zeros_like(adj_matrix)

    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1], strict=False):
        if adj_matrix[j, i] == -1:
            # directed edge: j -> i
            inferred_flat[i, j] = 1
        elif adj_matrix[j, i] == 1:
            # bidirected edge: j <-> i
            if inferred_flat[j, i] == 0:
                inferred_flat[i, j] = 3

    indices = np.where(adj_matrix == -1)
    for i, j in zip(indices[0], indices[1], strict=False):
        if adj_matrix[j, i] == -1:
            # undirected edge: j -- i
            if inferred_flat[j, i] == 0:
                inferred_flat[i, j] = 2

    return inferred_flat


def convert_causallearn_pag(adj_matrix: np.ndarray) -> np.ndarray:
    """Convert causal-learn's PAG encoding to our adjacency convention.

    PAG-specific edges beyond standard CPDAG:
      adj[i,j]=1  and adj[j,i]=2   =>  circle-arrow: j o-> i  (value 4)
      adj[i,j]=2  and adj[j,i]=2   =>  circle-circle: j o-o i (value 6)

    Our convention adds:
      mat[i,j]=4 => j o-> i (PAG)
      mat[i,j]=6 => j o-o i (PAG)
    """
    inferred_flat = np.zeros_like(adj_matrix)

    indices = np.where(adj_matrix == 1)
    for i, j in zip(indices[0], indices[1], strict=False):
        if adj_matrix[j, i] == -1:
            # directed edge: j -> i
            inferred_flat[i, j] = 1
        elif adj_matrix[j, i] == 2:
            # circle-arrow: j o-> i
            inferred_flat[i, j] = 4
        elif adj_matrix[j, i] == 1:
            # bidirected edge: j <-> i
            if inferred_flat[j, i] == 0:
                inferred_flat[i, j] = 3

    indices = np.where(adj_matrix == 2)
    for i, j in zip(indices[0], indices[1], strict=False):
        if adj_matrix[j, i] == 2:
            # circle-circle: j o-o i
            if inferred_flat[j, i] == 0:
                inferred_flat[i, j] = 6

    return inferred_flat


def convert_castle_matrix(causal_matrix: np.ndarray) -> np.ndarray:
    """Convert gcastle's causal_matrix to our convention.

    gcastle: causal_matrix[i,j] != 0 => i -> j
    Our convention: mat[i,j] = 1 => j -> i  (transpose + binarize)
    """
    binary = np.where(causal_matrix != 0, 1, 0)
    return binary.T


def convert_castle_cpdag(adj_matrix: np.ndarray) -> np.ndarray:
    """Convert gcastle PC's adjacency matrix (transposed) to our CPDAG convention.

    After transposing gcastle's matrix so that adj[i,j]=1 means j->i,
    mutual edges (adj[i,j]=1 and adj[j,i]=1) become undirected (value 2).
    Self-loops are cleared.
    """
    result = adj_matrix.copy()

    # Clear self-loops
    np.fill_diagonal(result, 0)

    n = result.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if result[i, j] == 1 and result[j, i] == 1:
                result[i, j] = 2
                result[j, i] = 0

    return result
