from typing import List
import numpy as np


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    relevance = np.asfarray(relevance)[:k]
    if relevance.size == 0:
        return 0.0

    discounts = np.log2(np.arange(2, relevance.size + 2))
    if method == "standard":
        return float(np.sum(relevance / discounts))
    if method == "industry":
        gains = np.power(2, relevance) - 1
        return float(np.sum(gains / discounts))
    raise ValueError()


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator

    Returns
    -------
    score : `float`
        Metric score
    """
    dcg = discounted_cumulative_gain(relevance, k, method)

    ideal_relevance = sorted(relevance, reverse=True)
    idcg = discounted_cumulative_gain(ideal_relevance, k, method)

    if idcg == 0:
        return 0.0

    return dcg / idcg
