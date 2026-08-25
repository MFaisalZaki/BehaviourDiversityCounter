"""The Wilcoxon signed-rank test, paired per task.

Written out rather than imported from scipy so that reading the results costs
nothing beyond the two packages the library itself needs. The implementation is
checked against ``scipy.stats.wilcoxon`` in ``tests/test_stats.py``, which skips
itself when scipy is absent.

The test is paired and non-parametric, which is what these comparisons need:
every selector sees the same task, the same pool and the same hidden-preference
draws, so the pairing is real, and nothing about a hit rate or a spread is
normally distributed.
"""

import math


def rank_average(values):
    """Ranks of ``values``, ties sharing their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1        # ranks are 1-based
        for index in order[i:j + 1]:
            ranks[index] = shared
        i = j + 1
    return ranks


def _exact_two_sided(statistic, n):
    """P(W+ <= statistic) * 2 under the null, by counting sign assignments.

    Valid only when no |difference| is tied and no pair is equal, which is what
    makes every rank the integer it needs to be here. Counts are built by
    dynamic programming over the achievable rank sums.
    """
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for value in range(total, rank - 1, -1):
            counts[value] += counts[value - rank]
    cumulative = sum(counts[:int(statistic) + 1])
    return min(1.0, 2.0 * cumulative / (2 ** n))


def wilcoxon(x, y=None):
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Returns ``(statistic, p_value, n_used)``. Pairs with a zero difference are
    dropped, as in the test's original formulation; ``n_used`` is what is left,
    and a p-value computed from three or four pairs should be read as the
    formality it is.
    """
    differences = list(x) if y is None else [a - b for a, b in zip(x, y)]
    non_zero = [d for d in differences if d != 0]
    n = len(non_zero)
    if n == 0:
        return 0.0, 1.0, 0

    magnitudes = [abs(d) for d in non_zero]
    ranks = rank_average(magnitudes)
    positive = sum(rank for rank, d in zip(ranks, non_zero) if d > 0)
    negative = sum(rank for rank, d in zip(ranks, non_zero) if d < 0)
    statistic = min(positive, negative)

    tied = len(set(magnitudes)) != len(magnitudes)
    if not tied and n <= 25:
        return statistic, _exact_two_sided(statistic, n), n

    # Normal approximation, with the continuity correction and the tie
    # correction to the variance.
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24
    counts = {}
    for magnitude in magnitudes:
        counts[magnitude] = counts.get(magnitude, 0) + 1
    variance -= sum(t ** 3 - t for t in counts.values()) / 48
    if variance <= 0:
        return statistic, 1.0, n
    z = (statistic - mean + 0.5) / math.sqrt(variance)
    p = 2.0 * 0.5 * math.erfc(-z / math.sqrt(2))
    return statistic, min(1.0, p), n


def holm_bonferroni(p_values):
    """Holm-Bonferroni adjusted p-values, in the order given.

    Several selectors are compared against the same baseline on the same tasks,
    so the family-wise error rate needs controlling; Holm is uniformly more
    powerful than Bonferroni and assumes nothing extra.
    """
    indexed = sorted(range(len(p_values)), key=lambda i: p_values[i])
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(indexed):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)     # keep them monotone
        adjusted[index] = running
    return adjusted


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return float('nan')
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2
