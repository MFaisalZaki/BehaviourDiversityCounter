import math

import numpy as np

from functools import partial

from unified_planning.shortcuts import SequentialSimulator

from behaviour_diversity_counter.simulation import InapplicablePlanError, simulate
from behaviour_diversity_counter.dimensions.goal_predicate_ordering import GoalPredicatesOrderingDimension
from behaviour_diversity_counter.dimensions.cost_bound_makespan_optimal import MakespanOptimalCostDimension
from behaviour_diversity_counter.dimensions.resources import (
    ResourceCountDimension, ResourceNumberDimension, ResourceUsedDimension)
from behaviour_diversity_counter.dimensions.cost_bin import CostBinDimension
from behaviour_diversity_counter.dimensions.utility_value import UtilityValueDimension
from behaviour_diversity_counter.dimensions.functions import NumericFunctionDimension
from behaviour_diversity_counter.dimensions.stability import StabilityDimension

dimensions_map = {
    'go': GoalPredicatesOrderingDimension,
    'cb': MakespanOptimalCostDimension,
    'rc': ResourceCountDimension,
    'ru': ResourceUsedDimension,
    'rn': ResourceNumberDimension,
    'uv': UtilityValueDimension,
    'fn': NumericFunctionDimension,
    'cbin': CostBinDimension,
    'stability': StabilityDimension,
}

#: The neighbourhood size kappa of B-Novelty (Def. bnovelty) when the caller
#: names none. The paper fixes no value and evaluates kappa in {1, 2, 3}.
#:
#: Novelty search takes 15 (Lehman and Stanley) and NSLC 20, but those count
#: neighbours in a population and archive of thousands, where 15 is a genuinely
#: local neighbourhood. Here the neighbours are drawn from the *distinct
#: behaviours* of one plan pool -- tens, not thousands -- and ``kappa`` is
#: clamped to ``b - 1``. At 15, every pool with 16 or fewer behaviours has every
#: behaviour averaging over all the others, which is the mean pairwise
#: dissimilarity: B-MaxSum over C(b, 2), reported under another name. At 3 that
#: does not happen for any pool with more than four behaviours.
DEFAULT_KAPPA = 3

#: Scores agreeing to this many decimals count as tied.
#:
#: Greedy compares sums of the same dissimilarities taken in different orders,
#: so two mathematically equal candidates routinely differ in the last bit.
#: Letting that decide the pick is reproducible but not stable: an unrelated
#: change to how a score is accumulated, or a different numpy, silently returns
#: a different selection.
#:
#: Nine decimals rather than three. The noise being absorbed is of order 1e-15;
#: three decimals declared two candidates tied whenever they came within 5e-4 of
#: each other, which let the greedy take a plan that is not the maximiser the
#: paper's rule names -- the Phase 0 audit found such a case in a random
#: behaviour space, where the greedy gave away 8.8e-05 of B-MaxSum. On the
#: benchmark's own spaces, whose dissimilarities are rationals with small
#: denominators, the two settings select identically (648 selections compared,
#: none changed); see docs/AUDIT.md.
TIE_DECIMALS = 9


def best_index(scores):
    """The highest-scoring entry, ties falling to the earliest.

    The paper breaks ties arbitrarily; the implementation takes the lowest plan
    index. Callers pass scores in ascending plan-index order, and ``np.argmax``
    takes the first maximum, so the earliest entry of a tied group wins.
    """
    return int(np.argmax(np.round(scores, TIE_DECIMALS)))


class BehaviourDiversityCounter:
    """A diversity model over one task, and the paper's four indicators on it.

    ``dimensions`` is an iterable of ``(key, addinfo)`` pairs, one feature
    ``<Delta, extract, psi, w>`` each in the sense of Def. feature: the key
    names the dimension and its extracting function, the dimension class
    supplies the per-dimension dissimilarity, and ``addinfo`` may declare the
    weight (``{'weight': w}``) next to whatever else the dimension needs.
    Together they are the diversity model M of Def. diversity-model, whose
    dissimilarity is ``psi_M(a, b) = sum_i w_i * psi_i(a[i], b[i])``.

    Weights are declared for every dimension or for none. Declared weights lie
    in ``(0, 1]`` and sum to one, as Def. feature and Def. diversity-model
    require, which keeps ``psi_M`` in ``[0, 1]``. With none declared they
    default to the uniform ``1/n``, the weights of the paper's rover example.
    """

    def __init__(self, task, dimensions, trace_cache=None):
        """``trace_cache`` is an optional ``{id(plan): (states, cost)}`` mapping
        shared between counters over the same task: a plan is simulated by
        whichever counter meets it first and replayed by none of the others.
        The caller keeps the plans alive, as the mapping is keyed by identity.
        """
        dimensions = list(dimensions)
        unknown = [name for name, _ in dimensions if name not in dimensions_map]
        if unknown:
            raise ValueError(f'unknown dimension(s) {unknown}; valid keys: {sorted(dimensions_map)}')
        self.task = task
        self.dimensions = {name: dimensions_map[name](task, addinfo) for name, addinfo in dimensions}
        self._apply_weight_convention()
        self._simulator = SequentialSimulator(problem=task)
        self._trace_cache = trace_cache
        self._behaviour_cache = {}
        self._cost_cache = {}
        self._dissimilarity_cache = {}

    def _apply_weight_convention(self):
        """Declared weights for all dimensions or for none; none means uniform."""
        declared = [name for name, dim in self.dimensions.items() if dim.declared_weight]
        missing  = [name for name, dim in self.dimensions.items() if not dim.declared_weight]
        if declared and missing:
            raise ValueError(f'no weight given for dimension(s): {missing}; '
                             f'declare a weight for every dimension or for none')
        if not declared:
            for dim in self.dimensions.values():
                dim.weight = 1.0 / len(self.dimensions)
            return
        invalid = [name for name, dim in self.dimensions.items() if not 0 < dim.weight <= 1]
        if invalid:
            raise ValueError(f'weights must lie in (0, 1] (Def. feature); got {invalid}')
        total = sum(dim.weight for dim in self.dimensions.values())
        if not math.isclose(total, 1.0):
            raise ValueError(f'weights must sum to one (Def. diversity-model); got {total}')

    # ------------------------------------------------------------------
    # Indicators (Def. bc, Def. maxsum, Def. bmaxmin, Def. bnovelty)
    # ------------------------------------------------------------------

    def behaviours(self, plans):
        """B_M(plans): the set of distinct behaviours the plans exhibit."""
        return set(self._plan_behaviours(plans))

    def b_coverage(self, plans):
        """B-Coverage: the number of distinct behaviours, |B_M(plans)|."""
        return len(self.behaviours(plans))

    def b_maxsum(self, plans):
        """B-MaxSum: the sum of psi_M over the unordered pairs of distinct
        behaviours; 0 below two."""
        return float(self._pairwise(plans).sum())

    def b_maxmin(self, plans):
        """B-MaxMin: the smallest psi_M over the unordered pairs of distinct
        behaviours; 0 below two."""
        pairs = self._pairwise(plans)
        return float(pairs.min()) if pairs.size else 0.0

    def b_novelty(self, plans, kappa=DEFAULT_KAPPA):
        """B-Novelty: the mean, over the distinct behaviours, of each
        behaviour's mean dissimilarity to its neighbourhood N_kappa, the
        ``min(kappa, b - 1)`` closest other behaviours; 0 below two.
        """
        unique = self._unique_behaviours(plans)
        if len(unique) < 2: return 0.0
        return self._novelty(self._dissimilarity_matrix(unique), kappa)

    @staticmethod
    def _novelty(psi, kappa):
        """Def. bnovelty on ``psi``, the b x b matrix of psi_M over b >= 2
        distinct behaviours."""
        psi = psi.copy()
        # A behaviour is not its own neighbour; inf keeps the diagonal out of
        # every k-smallest without excluding it index by index.
        np.fill_diagonal(psi, np.inf)
        k_prime = min(kappa, len(psi) - 1)
        nearest = np.partition(psi, k_prime - 1, axis=1)[:, :k_prime]
        return float(nearest.mean(axis=1).mean())

    def _pairwise(self, plans):
        """psi_M over the unordered pairs of distinct behaviours, as a vector."""
        unique = self._unique_behaviours(plans)
        return self._dissimilarity_matrix(unique)[np.triu_indices(len(unique), k=1)]

    # ------------------------------------------------------------------
    # Selection: extract_lambda(M, C, k) of Alg. two-phase
    # ------------------------------------------------------------------

    def extract(self, plans, k, indicator='bcoverage', kappa=DEFAULT_KAPPA):
        """Select k plans from the candidate pool for the chosen indicator.

        k plans come back whenever the pool holds that many, however the
        indicator moves across the steps -- B-MaxMin can only fall as the
        selection grows, a minimum over pairs never rising when a pair is
        added. The paper returns the k plans the greedy selects rather than
        truncating to the best-scoring prefix: the indicator is to certify a
        set of the requested size, not to choose that size.
        """
        plans = list(plans)
        selections = {
            'bcoverage': self._extract_bcoverage,
            'bmaxsum':   partial(self._extract_greedy, aggregate=np.sum),
            'bmaxmin':   partial(self._extract_greedy, aggregate=np.min),
            'bnovelty':  partial(self._extract_bnovelty, kappa=kappa),
        }
        if indicator not in selections:
            raise ValueError(f"unknown indicator '{indicator}'; valid indicators: {sorted(selections)}")
        # Every plan of the pool gets its `behaviour` attribute, whichever rule
        # runs and whichever early return it takes.
        self._plan_behaviours(plans)
        return selections[indicator](plans, k)

    def _extract_bcoverage(self, plans, k):
        """Alg. extract-bc: one plan per behaviour, the cheapest exhibiting it,
        then fill.

        A pass over the pool that keeps, for each behaviour, the cheapest plan
        exhibiting it, as MAP-Elites retains the best solution per cell; ties
        fall to the earliest plan. No dissimilarity is read, so this never
        builds the matrix the other three rules open on. Behaviours are taken
        in first-occurrence order, and once every behaviour is held the
        remaining slots are filled in pool order.
        """
        if k <= 0 or not plans:
            return []

        cheapest = {}   # behaviour -> index of its cheapest plan, in first-occurrence order
        costs = self._costs_of(plans)
        for idx, (behaviour, cost) in enumerate(zip(self._plan_behaviours(plans), costs)):
            if behaviour not in cheapest or cost < costs[cheapest[behaviour]]:
                cheapest[behaviour] = idx
        covered = np.fromiter(cheapest.values(), dtype=np.intp, count=len(cheapest))[:k]

        # A repeat leaves the count untouched, so once the behaviours run out
        # the tail is padded in plan-index order, as the greedy rules pad theirs.
        rest = np.setdiff1d(np.arange(len(plans)), covered)
        return [plans[idx] for idx in np.concatenate([covered, rest])[:k]]

    def _extract_greedy(self, plans, k, *, aggregate):
        """Alg. extract-bmaxsum (``aggregate=np.sum``, the greedy of Katz et
        al. 2020) and Alg. extract-bmaxmin (``np.min``, the farthest-first of
        Ravi et al. 1994): open on the farthest pair, then add the candidate
        whose dissimilarities to the selection aggregate highest.

        The seed needs no split: over two behaviours there is one pair, so sum
        and min are the same number and the farthest pair opens both.
        """
        if k < 2 or len(plans) < 2:
            return plans[:max(k, 0)]              # every such set scores 0

        matrix, codes = self._plan_dissimilarity_matrix(plans)
        selected, candidates = self._farthest_pair(matrix)

        while len(selected) < k and len(candidates):
            # A held behaviour leaves the indicator where it stands (its gain
            # is 0 under the sum and 0 under the min), so candidates are ranked
            # among the fresh behaviours until none remains, then padded in
            # plan order.
            scores = aggregate(matrix[np.ix_(candidates, selected)], axis=1)
            fresh = ~np.isin(codes[candidates], codes[selected])
            scores = np.where(fresh, scores, -np.inf) if fresh.any() else np.zeros(len(candidates))
            position = best_index(scores)
            selected.append(int(candidates[position]))
            candidates = np.delete(candidates, position)

        return [plans[idx] for idx in selected]

    def _extract_bnovelty(self, plans, k, kappa=DEFAULT_KAPPA):
        """Alg. extract-bnov: open on the farthest pair; while a behaviour of
        the pool is unheld, add the plan maximising B-Novelty over the
        combined set, evaluated once per unheld behaviour; then fill.

        B-Novelty is not monotone, so the fresh behaviour is taken even when
        it lowers the value, as the paper's rule says; the indicator reported
        for the returned set makes that fall visible.
        """
        if k < 2 or len(plans) < 2:
            return plans[:max(k, 0)]              # every such set scores 0

        matrix, codes = self._plan_dissimilarity_matrix(plans)
        selected, candidates = self._farthest_pair(matrix)
        # psi_M over the distinct behaviours, indexed by behaviour code: the
        # first plan exhibiting each behaviour stands for it.
        reps = np.unique(codes, return_index=True)[1]
        psi = matrix[np.ix_(reps, reps)]

        while len(selected) < k:
            held   = np.unique(codes[selected])                    # B_M(selected)
            unheld = np.setdiff1d(codes[candidates], held)         # B_M(pool) \ B_M(selected)
            if not unheld.size:
                break
            value = {b: self._novelty(psi[np.ix_(np.append(held, b), np.append(held, b))], kappa)
                     for b in unheld.tolist()}
            # A candidate scores its behaviour's value; a held behaviour is out
            # of the running. Ties fall to the lowest plan index.
            scores = [value.get(code, -np.inf) for code in codes[candidates].tolist()]
            position = best_index(scores)
            selected.append(int(candidates[position]))
            candidates = np.delete(candidates, position)

        # Every behaviour held: fill the remaining slots in plan order.
        selected += candidates[:k - len(selected)].tolist()
        return [plans[idx] for idx in selected]

    def _farthest_pair(self, matrix):
        """The opening pair of the three greedy rules: two plans maximising
        psi_M, ties to the earliest pair; the rest of the pool is the candidates."""
        rows, cols = np.triu_indices(len(matrix), k=1)
        best = best_index(matrix[rows, cols])
        selected = [int(rows[best]), int(cols[best])]
        candidates = np.setdiff1d(np.arange(len(matrix)), selected)
        return selected, candidates

    # ------------------------------------------------------------------
    # Behaviours, costs and dissimilarities
    # ------------------------------------------------------------------

    def _plan_behaviours(self, plans):
        """PBehaviour_M of each plan (Def. plan-beh): the value every
        dimension extracts from it, one token each, joined into one string."""
        result = []
        for plan in plans:
            if id(plan) not in self._behaviour_cache:
                states, cost = self._simulate(plan)
                setattr(plan, 'states', states)
                setattr(plan, 'cost', cost)
                behaviour = ' $$ '.join(dim.extract(plan) for dim in self.dimensions.values())
                delattr(plan, 'states')  # the trace is only needed while the tokens are built
                setattr(plan, 'behaviour', behaviour)
                self._behaviour_cache[id(plan)] = behaviour
                self._cost_cache[id(plan)] = cost
            result.append(self._behaviour_cache[id(plan)])
        return result

    def _unique_behaviours(self, plans):
        """B_M(plans) in first-occurrence order. It stands in for U_M(plans)
        of Def. plan-beh: psi_M reads a plan only through its behaviour, so
        every indicator is computed on the behaviours themselves."""
        return list(dict.fromkeys(self._plan_behaviours(plans)))

    def _costs_of(self, plans):
        """Each plan's cost (Def. plans), simulated alongside its behaviour."""
        self._plan_behaviours(plans)
        return [self._cost_cache[id(plan)] for plan in plans]

    def _simulate(self, plan):
        """``(states, cost)``; raises InapplicablePlanError when the plan
        cannot be replayed against the task."""
        if self._trace_cache is None:
            return simulate(self.task, plan, self._simulator)
        if id(plan) not in self._trace_cache:
            self._trace_cache[id(plan)] = simulate(self.task, plan, self._simulator)
        return self._trace_cache[id(plan)]

    def _dissimilarity(self, b1, b2):
        """psi_M(b1, b2) = sum_i w_i * psi_i(b1[i], b2[i]) (Def.
        diversity-model); each dimension applies its own weight."""
        if len(self.dimensions) == 0: return 0.0
        if (b1, b2) not in self._dissimilarity_cache:
            self._dissimilarity_cache[(b1, b2)] = sum(dim.dissimilarity(b1, b2) for dim in self.dimensions.values())
            self._dissimilarity_cache[(b2, b1)] = self._dissimilarity_cache[(b1, b2)]
        return self._dissimilarity_cache[(b1, b2)]

    def _dissimilarity_matrix(self, unique):
        """psi_M over the given distinct behaviours, as a fresh b x b array:
        callers write into it -- b_novelty fills the diagonal with inf -- and
        the values themselves are cached, so a second call re-reads them."""
        rows, cols = np.triu_indices(len(unique), k=1)
        compact    = np.zeros((len(unique), len(unique)))
        compact[rows, cols] = np.fromiter(
            (self._dissimilarity(unique[i], unique[j])
             for i, j in zip(rows.tolist(), cols.tolist())),
            dtype=float, count=rows.size)
        # Out-of-place: `compact += compact.T` reads cells the same statement
        # is writing.
        return compact + compact.T

    def _plan_dissimilarity_matrix(self, plans):
        """psi_M over the plans of the pool, n x n, with each plan's behaviour
        code so the greedy rules can tell a held behaviour from a fresh one."""
        behaviours = self._plan_behaviours(plans)
        unique     = list(dict.fromkeys(behaviours))
        code_of    = {behaviour: code for code, behaviour in enumerate(unique)}
        codes      = np.fromiter(map(code_of.get, behaviours), dtype=np.intp, count=len(behaviours))
        return self._dissimilarity_matrix(unique)[np.ix_(codes, codes)], codes
