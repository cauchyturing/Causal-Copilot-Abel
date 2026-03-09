"""MCP resources: algorithm profiles, HP specs, and guide documents."""

from __future__ import annotations

from causal_copilot.mcp.bridge import PIPELINE_ROOT

_ALGO_DIR = PIPELINE_ROOT / "causal_discovery" / "context" / "algos"
_HP_DIR = PIPELINE_ROOT / "causal_discovery" / "context" / "hyperparameters"


def get_algorithm_resources():
    """List all algorithm resources."""
    resources = []
    if _ALGO_DIR.exists():
        for f in sorted(_ALGO_DIR.glob("*.txt")):
            resources.append(
                {
                    "name": f.stem,
                    "uri": f"causal://algorithms/{f.stem}",
                    "description": f"Profile for {f.stem} causal discovery algorithm",
                }
            )
    return resources


def get_algorithm_content(name):
    """Get algorithm profile content."""
    path = _ALGO_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text()
    for f in _ALGO_DIR.glob("*.txt"):
        if f.stem.lower() == name.lower():
            return f.read_text()
    return None


def get_hp_resources():
    """List all HP spec resources."""
    resources = []
    if _HP_DIR.exists():
        for f in sorted(_HP_DIR.glob("*.json")):
            resources.append(
                {
                    "name": f.stem,
                    "uri": f"causal://hyperparameters/{f.stem}",
                    "description": f"Hyperparameter specification for {f.stem}",
                }
            )
    return resources


def get_hp_content(name):
    """Get HP spec content."""
    path = _HP_DIR / f"{name}.json"
    if path.exists():
        return path.read_text()
    for f in _HP_DIR.glob("*.json"):
        if f.stem.lower() == name.lower():
            return f.read_text()
    return None


_CI_TEST_GUIDE = """# CI Test Selection Guide

| Data Characteristic | CI Test | Notes |
|---------------------|---------|-------|
| Missing values | mv_fisherz | Missing-value adapted Fisher's z |
| Discrete/Category/Binary | chisq | Chi-squared test |
| Mixed (Mixture/Mixed) | kci | Kernel CI handles mixed types |
| Linear + Continuous | fisherz | Fisher's z-transform (fastest) |
| Nonlinear, small (n<1500, p<10) | kci | Kernel CI test (exact, slow) |
| Nonlinear, medium (n<10K, p<100) | rcit | Randomized CI test (approximate) |
| Nonlinear, large | fastkci | Fast kernel CI test (scalable) |

Priority: missing > discrete > mixed > linear > nonlinear (size-based)
"""

_SCORE_FUNC_GUIDE = """# Score Function Selection Guide

| Data Characteristic | Score Function | Notes |
|---------------------|----------------|-------|
| Discrete/Category/Binary | local_score_BDeu | Bayesian Dirichlet equivalent uniform |
| Mixed (Mixture/Mixed) | local_score_CV_general | Cross-validated general score |
| Nonlinear + Continuous | local_score_CV_general | Cross-validated general score |
| Linear + Continuous | local_score_BIC | Bayesian Information Criterion |
| Linear + GRaSP algorithm | local_score_BIC_from_cov | Covariance-based BIC (faster) |
"""

_GRAPH_GUIDE = """# Interpreting Causal Graphs

## Edge Types
- **Directed (->)**: mat[i,j]=1 means j causes i
- **Undirected (--)**: mat[i,j]=mat[j,i]=2, causal direction unknown
- **Bidirected (<->)**: mat[i,j]=mat[j,i]=3, shared latent confounder

## Graph Types
- **DAG**: All edges directed. Full causal inference valid.
- **CPDAG**: Mix of directed and undirected. Some effects identifiable.
- **PAG**: Includes circle marks (values 4-7) or bidirected edges. Latent confounders possible.

## Roles
- **Root cause**: No incoming directed edges. Only outgoing.
- **Terminal effect**: No outgoing directed edges. Only incoming.
- **Mediator**: Both incoming and outgoing directed edges.

## Convention
Adjacency matrix: mat[i,j]=1 means column j -> row i (j causes i).
"""

_GUIDES = {
    "ci-tests": _CI_TEST_GUIDE,
    "score-functions": _SCORE_FUNC_GUIDE,
    "interpreting-graphs": _GRAPH_GUIDE,
}


def get_guide_content(guide_name):
    return _GUIDES.get(guide_name)


def get_all_guide_names():
    return list(_GUIDES.keys())
