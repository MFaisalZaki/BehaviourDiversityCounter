"""An independent reference for the paper's diversity definitions.

The indicators B-Coverage, B-MaxSum, B-MaxMin and B-Novelty, their exhaustive
optima, and the phase-two rules extract_lambda(M, C, k), written straight from
the definitions. A behaviour is the tuple (extract_1(pi), ..., extract_n(pi));
``d`` is psi_M; a plan triple is ``(index, cost, behaviour)`` and pools arrive
cost-sorted, so the paper's "arbitrary" tie-break is made deterministic as the
earliest position given. Sums use math.fsum: exactly rounded, hence independent
of the order the behaviours happen to arrive in.
"""

import math
from itertools import combinations


def ref_distinct(behaviours):
    """B_M(Psi): the distinct behaviours, in first-occurrence order."""
    out = []
    for b in behaviours:
        if b not in out:
            out.append(b)
    return out


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


def ref_optimum(behaviours, d, k, indicator, kappa=None):
    """The exhaustive optimum over the subsets of exactly k behaviours."""
    u = ref_distinct(behaviours)
    if k < 1 or k > len(u):
        return (None, None)
    best_value, best_subset = None, None
    for subset in combinations(range(len(u)), k):
        value = ref_indicator(indicator, [u[i] for i in subset], d, kappa)
        if best_value is None or value > best_value:
            best_value, best_subset = value, subset
    return (best_value, best_subset)


def ref_optimum_at_most(behaviours, d, k, indicator, kappa=None):
    """The optimum over the subsets of size 1..min(k, b), smaller first;
    (None, None) when there is no such subset, i.e. b == 0 or k < 1."""
    u = ref_distinct(behaviours)
    best_value, best_subset = None, None
    for size in range(1, min(k, len(u)) + 1):
        for subset in combinations(range(len(u)), size):
            value = ref_indicator(indicator, [u[i] for i in subset], d, kappa)
            if best_value is None or value > best_value:
                best_value, best_subset = value, subset
    return (best_value, best_subset)


def _pad(plans, held, target):
    """Fill a selection up to target with further plans, in pool order."""
    for pos in range(len(plans)):
        if len(held) >= target:
            break
        if pos not in held:
            held.append(pos)
    return held


def _open(plans, d, k):
    """The shared opening: the first pair, i < j, maximising psi_M."""
    n = len(plans)
    if k <= 0 or n == 0:
        return []
    if k == 1 or n == 1:
        return list(range(min(k, n)))
    best_value, best_pair = None, None
    for i, j in combinations(range(n), 2):
        value = d(plans[i][2], plans[j][2])
        if best_value is None or value > best_value:
            best_value, best_pair = value, [i, j]
    return best_pair


def _greedy(plans, d, k, score):
    """Open on the best pair, then repeatedly add the lowest-position candidate
    of maximal score (score None skips it), and pad to min(k, |C|)."""
    held = _open(plans, d, k)
    target = min(k, len(plans))
    while len(held) < target:
        best_value, best_pos = None, None
        for pos in range(len(plans)):
            if pos in held:
                continue
            value = score(pos, held)
            if value is None:
                continue
            if best_value is None or value > best_value:
                best_value, best_pos = value, pos
        if best_pos is None:
            break
        held.append(best_pos)
    return [plans[pos][0] for pos in _pad(plans, held, target)]


def ref_extract_bcoverage(plans, d, k, kappa=None):
    """extract_BCoverage (exact): the cheapest plan of each behaviour, in
    first-occurrence order, up to k, then padding."""
    if k <= 0:
        return []
    seen, cheapest = [], []
    for pos, (_index, cost, behaviour) in enumerate(plans):
        if behaviour in seen:
            j = seen.index(behaviour)
            if cost < plans[cheapest[j]][1]:
                cheapest[j] = pos
        else:
            seen.append(behaviour)
            cheapest.append(pos)
    held = _pad(plans, cheapest[:k], min(k, len(plans)))
    return [plans[pos][0] for pos in held]


def ref_extract_bmaxsum(plans, d, k, kappa=None):
    """extract_BMaxSum: greedy on the B-MaxSum gain -- the sum of psi_M to the
    held plans for a new behaviour, 0 for a duplicate."""
    def gain(pos, held):
        behaviour = plans[pos][2]
        if any(behaviour == plans[h][2] for h in held):
            return 0.0
        return math.fsum(d(behaviour, plans[h][2]) for h in held)
    return _greedy(plans, d, k, gain)


def ref_extract_bmaxmin(plans, d, k, kappa=None):
    """extract_BMaxMin: farthest-first on the minimum psi_M to the held."""
    def gain(pos, held):
        return min(d(plans[pos][2], plans[h][2]) for h in held)
    return _greedy(plans, d, k, gain)


def ref_extract_bnovelty(plans, d, k, kappa):
    """extract_BNovelty: greedy on B-Novelty of the combined set over the
    new-behaviour candidates, then padding."""
    if kappa is None or kappa < 1:
        raise ValueError("extract 'bnovelty' needs kappa >= 1, got %r" % (kappa,))

    def gain(pos, held):
        behaviour = plans[pos][2]
        if any(behaviour == plans[h][2] for h in held):
            return None
        combined = [plans[h][2] for h in held] + [behaviour]
        return ref_bnovelty(combined, d, kappa)
    return _greedy(plans, d, k, gain)


def ref_extract(name, plans, d, k, kappa=None):
    """Dispatch to the named extraction rule."""
    rules = {'bcoverage': ref_extract_bcoverage,
             'bmaxsum': ref_extract_bmaxsum,
             'bmaxmin': ref_extract_bmaxmin,
             'bnovelty': ref_extract_bnovelty}
    if name not in rules:
        raise ValueError('unknown extraction rule: %r' % (name,))
    return rules[name](plans, d, k, kappa)
