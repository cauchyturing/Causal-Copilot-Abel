"""Deterministic CI test resolution based on data characteristics.

Replaces LLM guesswork with data-driven CI test selection.
Hierarchy: missing → discrete → linear → nonlinear (size-based).
"""


def resolve_ci_test(statistics):
    """Select CI test based on data characteristics.

    Args:
        statistics: GlobalState.statistics with linearity, data_type,
                   missingness, sample_size, feature_number attributes.

    Returns:
        str: CI test name compatible with causal-learn.
    """
    if statistics.missingness:
        return "mv_fisherz"

    if statistics.data_type in ("Discrete", "Category", "Binary"):
        return "chisq"

    if statistics.data_type in ("Mixture", "Mixed"):
        return "kci"  # KCI handles mixed types; fisherz assumes continuous

    if statistics.linearity:
        return "fisherz"

    # Nonlinear: pick based on computational budget
    n = statistics.sample_size or 0
    p = statistics.feature_number or 0

    if n < 1500 and p < 10:
        return "kci"
    if n < 10000 and p < 100:
        return "rcit"
    return "fastkci"
