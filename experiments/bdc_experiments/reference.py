"""An independent reference for the paper's diversity definitions.

The indicators B-Coverage, B-MaxSum, B-MaxMin and B-Novelty and the stability
distance of the literature's model, written straight from the definitions:
the reports read every indicator value off a behaviour dump through these,
and the audit in tests/experiments/test_audit.py holds the library to them.
A behaviour is the tuple (extract_1(pi), ..., extract_n(pi)) and ``d`` is
psi_M. Sums use math.fsum: exactly rounded, hence independent of the order
the behaviours happen to arrive in.
"""

import math
from itertools import combinations


def ref_distinct(behaviours):
    """B_M(Psi): the distinct behaviours, in first-occurrence order."""
    return list(dict.fromkeys(behaviours))


def ref_bcoverage(behaviours):
    """B-Coverage: |B_M(Psi)|."""
    return len(ref_distinct(behaviours))


def ref_bmaxsum(behaviours, d):
    """B-MaxSum: the sum of psi_M over the unordered pairs of U_M(Psi)."""
    u = ref_distinct(behaviours)
    if len(u) < 2:
        return 0.0
    return math.fsum(d(x, y) for x, y in combinations(u, 2))


def ref_bmaxmin(behaviours, d):
    """B-MaxMin: the minimum psi_M over the unordered pairs of U_M(Psi)."""
    u = ref_distinct(behaviours)
    if len(u) < 2:
        return 0.0
    return float(min(d(x, y) for x, y in combinations(u, 2)))


def ref_bnovelty(behaviours, d, kappa):
    """B-Novelty(kappa): the mean over U_M(Psi) of the mean psi_M to the
    kappa' = min(kappa, b - 1) nearest neighbours."""
    if kappa is None or kappa < 1:
        raise ValueError('B-Novelty needs kappa >= 1, got %r' % (kappa,))
    u = ref_distinct(behaviours)
    b = len(u)
    if b < 2:
        return 0.0
    k_prime = min(kappa, b - 1)
    per = []
    for i, x in enumerate(u):
        near = sorted(d(x, y) for j, y in enumerate(u) if j != i)
        per.append(math.fsum(near[:k_prime]) / k_prime)
    return math.fsum(per) / b


def ref_indicator(name, behaviours, d, kappa=None):
    """Dispatch to the named indicator; B-Coverage is returned as a float."""
    if name == 'bcoverage':
        return float(ref_bcoverage(behaviours))
    if name == 'bmaxsum':
        return ref_bmaxsum(behaviours, d)
    if name == 'bmaxmin':
        return ref_bmaxmin(behaviours, d)
    if name == 'bnovelty':
        if kappa is None:
            raise ValueError("indicator 'bnovelty' needs kappa")
        return ref_bnovelty(behaviours, d, kappa)
    raise ValueError('unknown indicator: %r' % (name,))


def ref_stability(actions_a, actions_b):
    """The stability distance of Srivastava et al. (2007): one minus the
    Jaccard measure of the two plans' action sets."""
    a, b = set(actions_a), set(actions_b)
    return 1.0 - len(a & b) / len(a | b) if a | b else 0.0

