"""Deterministic score function resolution based on data characteristics.

Maps data type + linearity to appropriate score function for score-based algorithms.
"""


def resolve_score_func(statistics, algorithm):
    """Select score function based on data characteristics.

    Args:
        statistics: GlobalState.statistics with data_type, linearity.
        algorithm: Algorithm name (str). GRaSP uses BIC_from_cov variant.

    Returns:
        str: Score function name compatible with causal-learn.
    """
    if statistics.data_type in ("Discrete", "Category", "Binary"):
        return "local_score_BDeu"

    if not statistics.linearity:
        return "local_score_CV_general"

    if algorithm == "GRaSP":
        return "local_score_BIC_from_cov"

    return "local_score_BIC"
