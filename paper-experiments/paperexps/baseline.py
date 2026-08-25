"""The plan-level baseline: Stability, and greedy MaxSum over it.

Named ``baseline`` rather than ``selectors``: a runner in this directory puts
it on ``sys.path`` ahead of the standard library, and a module called
``selectors`` there shadows the stdlib one that ``subprocess`` imports, which
breaks ``unified_planning`` at import time with an error naming neither.

This is what the diverse-planning literature does, and the only condition in
the three experiments that never touches the behaviour space. It lives here
rather than in the library because it is not part of the behaviour-space model
the paper proposes -- it is what that model is being compared against.

Lifted out of the deleted ``run_pool.py``, which imported ``Stability`` from
``plandiversity``. That implementation reduces a plan to
``frozenset(str(action) for action in plan.actions)`` and says so: "Repeated
actions are collapsed: the metric compares sets, not multisets or sequences."
Katz and Sohrabi define A(p) as a multiset, so that reading hands the baseline
a weaker notion of difference than its own authors gave it -- a plan that
drives back and forth four times becomes identical to one that drives once --
and every comparison against it would flatter the behaviour space by exactly
that much. The multiset reading is therefore the default here, with the set
reading kept behind a flag so the two can be compared if a reviewer asks.
"""

import collections
import itertools


class Stability:
    """``1 - |A(p) & A(q)| / |A(p) | A(q)|`` over the actions of two plans.

    ``A(p)`` is a multiset by default, following Katz and Sohrabi (2020): an
    action occurring three times contributes three times, so intersection and
    union are taken with multiplicities,

        |A(p) & A(q)| = sum_a min(count_p(a), count_q(a))
        |A(p) | A(q)| = sum_a max(count_p(a), count_q(a))

    ``multiset=False`` restores the set reading, where each action counts once
    however often it occurs.

    Features are extracted once per plan and cached by plan identity; the plan
    itself is kept alongside so CPython cannot recycle its ``id()`` onto an
    unrelated object, and so a hit can be confirmed rather than assumed.

    References
    ----------
    M. Katz and S. Sohrabi, "Reshaping diverse planning", AAAI 2020.
    """

    name = 'Stability'

    def __init__(self, plans=None, multiset=True):
        self.multiset = multiset
        self._cache = {}
        for plan in (plans or []):
            self._feature_of(plan)

    def feature(self, plan):
        """The multiset (or set) of ground actions the plan uses."""
        names = [str(action) for action in plan.actions]
        return collections.Counter(names) if self.multiset else frozenset(names)

    def _feature_of(self, plan):
        entry = self._cache.get(id(plan))
        if entry is not None and entry[0] is plan:
            return entry[1]
        feature = self.feature(plan)
        self._cache[id(plan)] = (plan, feature)
        return feature

    def distance(self, plan_a, plan_b):
        return self._distance(self._feature_of(plan_a), self._feature_of(plan_b))

    def _distance(self, a, b):
        if not self.multiset:
            union = len(a | b)
            return 0.0 if union == 0 else 1.0 - len(a & b) / union
        # sum(min) + sum(max) = |p| + |q|, so the union follows from the
        # intersection and only the smaller multiset needs walking.
        smaller, larger = (a, b) if len(a) <= len(b) else (b, a)
        intersection = sum(min(count, larger[action])
                           for action, count in smaller.items() if action in larger)
        total = sum(a.values()) + sum(b.values())
        union = total - intersection
        return 0.0 if union == 0 else 1.0 - intersection / union

    def maxsum(self, plans):
        """The MaxSum objective: the total distance over unordered plan pairs.

        Unordered, matching :meth:`BehaviourDiversityCounter.b_maxsum`, so the
        cross-scoring table of Experiment B compares like with like.
        """
        return sum(self.distance(a, b) for a, b in itertools.combinations(plans, 2))


def greedy_maxsum_stability(plans, k, stability=None):
    """Katz and Sohrabi's greedy selection, run on Stability.

    Their section 5.2, quoted because the algorithm is its description: order
    the plans by cost, find the pair with the largest diversity score, then
    iteratively add the plan that maximises the diversity of the resulting set,
    stopping at k.

    Cost here is plan length -- every pool in this sweep was generated at
    q = 1.0, so all its plans are optimal and the ordering reduces to pool
    order, which is what breaks the ties.

    Note the asymmetry with ``extract(..., 'bmaxsum')``, which is deliberate:
    that one opens on an arbitrary plan because every singleton scores zero,
    while this one opens on the farthest pair. Each rule is left as its own
    authors define it, and where that helps a condition it helps the baseline.
    """
    plans = list(plans)
    if k <= 0 or not plans:
        return []
    stability = stability if stability is not None else Stability(plans)
    order = sorted(range(len(plans)), key=lambda i: (len(plans[i].actions), i))
    if len(plans) == 1 or k == 1:
        return [plans[order[0]]]

    # The opening pair. `order` is in preference order and the test is strict,
    # so ties fall to the cheapest plans and then to the lowest pool index.
    best_pair, best_distance = (order[0], order[1]), None
    for a, b in itertools.combinations(order, 2):
        distance = stability.distance(plans[a], plans[b])
        if best_distance is None or distance > best_distance:
            best_pair, best_distance = (a, b), distance

    chosen = sorted(best_pair)
    taken = set(chosen)
    # Each candidate's total distance to the chosen set, carried along so that
    # adding a plan costs one pass over the pool rather than one per pair.
    totals = {i: sum(stability.distance(plans[i], plans[j]) for j in chosen)
              for i in order if i not in taken}
    while len(chosen) < k and len(taken) < len(plans):
        best, best_total = None, None
        for i in order:
            if i in taken:
                continue
            if best_total is None or totals[i] > best_total:
                best, best_total = i, totals[i]
        chosen.append(best)
        taken.add(best)
        for i in order:
            if i not in taken:
                totals[i] += stability.distance(plans[i], plans[best])
    return [plans[i] for i in chosen]
