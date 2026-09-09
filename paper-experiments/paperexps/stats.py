"""The statistics the report stage uses, written out rather than imported so
that a compute node needs nothing beyond the standard library.

``wilcoxon`` is the two-sided signed-rank test, exact for small samples
without ties and a tie-corrected normal approximation otherwise; ``holm`` is
Holm's step-down correction; ``kendall_tau`` is tau-b.  ``tests/test_stats.py``
checks all three against scipy where scipy is installed.
"""

import math


def median(values):
    values = sorted(values)
    n = len(values)
    if n == 0:
        return None
    middle = n // 2
    return values[middle] if n % 2 else (values[middle - 1] + values[middle]) / 2


def quantile(values, fraction):
    """Linear interpolation between order statistics, as numpy's default."""
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    low, high = int(math.floor(position)), int(math.ceil(position))
    return values[low] + (values[high] - values[low]) * (position - low)


def iqr(values):
    return None if not values else quantile(values, 0.75) - quantile(values, 0.25)


def _ranks(values):
    """Average ranks, 1-based, ties sharing their mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        rank = (i + j + 2) / 2
        for position in range(i, j + 1):
            ranks[order[position]] = rank
        i = j + 1
    return ranks


def _exact_signed_rank_p(w_plus, n):
    """Two-sided p of W+ under the null, by enumerating the 2^n sign patterns
    through their rank-sum distribution."""
    total = n * (n + 1) // 2
    counts = [0] * (total + 1)
    counts[0] = 1
    for rank in range(1, n + 1):
        for s in range(total, rank - 1, -1):
            counts[s] += counts[s - rank]
    scale = 2.0 ** n
    w = int(round(w_plus))
    lower = sum(counts[:w + 1]) / scale
    upper = sum(counts[w:]) / scale
    return min(1.0, 2 * min(lower, upper))


def wilcoxon(x, y, exact_limit=25):
    """Two-sided Wilcoxon signed-rank test of paired samples ``x`` and ``y``.

    Zero differences are discarded (Wilcoxon's own treatment).  Returns a dict
    with the statistic ``W+``, the p-value, the number of pairs used, and the
    median and interquartile range of ``x - y``; ``p`` is None when fewer than
    one non-zero difference remains.
    """
    differences = [a - b for a, b in zip(x, y)]
    nonzero = [d for d in differences if d != 0]
    n = len(nonzero)
    result = {'n': len(differences), 'n_nonzero': n,
              'median_diff': median(differences), 'iqr_diff': iqr(differences),
              'statistic': None, 'p': None, 'method': None}
    if n == 0:
        return result
    magnitudes = [abs(d) for d in nonzero]
    ranks = _ranks(magnitudes)
    w_plus = sum(rank for rank, d in zip(ranks, nonzero) if d > 0)
    result['statistic'] = w_plus
    has_ties = len(set(magnitudes)) < n
    if n <= exact_limit and not has_ties:
        result['p'] = _exact_signed_rank_p(w_plus, n)
        result['method'] = 'exact'
        return result
    mean = n * (n + 1) / 4
    tie_counts = {}
    for rank in ranks:
        tie_counts[rank] = tie_counts.get(rank, 0) + 1
    tie_term = sum(t ** 3 - t for t in tie_counts.values()) / 48
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie_term
    if variance <= 0:
        result['p'] = 1.0
        result['method'] = 'approx'
        return result
    deviation = w_plus - mean
    # Continuity correction towards the mean.
    deviation -= 0.5 * (1 if deviation > 0 else -1 if deviation < 0 else 0)
    z = deviation / math.sqrt(variance)
    result['p'] = min(1.0, math.erfc(abs(z) / math.sqrt(2)))
    result['method'] = 'approx'
    return result


def holm(pvalues):
    """Holm's step-down adjusted p-values, in the input order; ``None`` entries
    stay ``None`` and do not count."""
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    m = len(indexed)
    adjusted = [None] * len(pvalues)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, (m - rank) * p)
        adjusted[i] = min(1.0, running)
    return adjusted


def kendall_tau(x, y):
    """Kendall's tau-b between two equally long sequences; None when either
    has no variation."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    concordant = discordant = ties_x = ties_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = (x[i] > x[j]) - (x[i] < x[j])
            dy = (y[i] > y[j]) - (y[i] < y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denominator == 0:
        return None
    return (concordant - discordant) / denominator
