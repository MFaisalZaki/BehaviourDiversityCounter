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

dimensions_map = {
    'go': GoalPredicatesOrderingDimension,
    'cb': MakespanOptimalCostDimension,
    'rc': ResourceCountDimension,
    'ru': ResourceUsedDimension,
    'rn': ResourceNumberDimension,
    'uv': UtilityValueDimension,
    'fn': NumericFunctionDimension,
    'cbin': CostBinDimension,
}

#: How many nearest neighbours B-Novelty averages over.
#:
#: Novelty search takes 15 (Lehman and Stanley) and NSLC 20, but those count
#: neighbours in a population and archive of thousands, where 15 is a genuinely
#: local neighbourhood. Here the neighbours are drawn from the *distinct
#: behaviours* of one plan pool -- tens, not thousands -- and ``k_nn`` is
#: clamped to ``b - 1``. At 15, every pool with 16 or fewer behaviours has every
#: behaviour averaging over all the others, which is the mean pairwise distance:
#: B-MaxSum over C(b, 2), reported under another name. At 3 that does not happen
#: for any pool with more than four behaviours.
DEFAULT_K_NN = 3

#: Scores agreeing to this many decimals count as tied.
#:
#: Greedy compares sums of the same distances taken in different orders, so two
#: mathematically equal candidates routinely differ in the last bit. Letting
#: that decide the pick is reproducible but not stable: an unrelated change to
#: how a score is accumulated, or a different numpy, silently returns a
#: different selection.
TIE_DECIMALS = 3


def best_index(scores):
    """The highest-scoring entry, ties falling to the earliest.

    Callers pass scores in ascending plan-index order, and ``np.argmax`` takes
    the first maximum, so the earliest entry of a tied group wins.
    """
    return int(np.argmax(np.round(scores, TIE_DECIMALS)))


class BehaviourDiversityCounter:
    """A behaviour space over one task, and the paper's four indicators on it.

    ``dimensions`` is an iterable of ``(key, addinfo)`` pairs, one feature each
    in the sense of the paper's Def. feature: the key names the dimension and
    its extracting function, the dimension class supplies the per-dimension
    distance, and ``addinfo`` may declare the weight (``{'weight': w}``) next
    to whatever else the dimension needs.

    The behaviour distance is the additively separable one of Def.
    separable-distance, ``d(b, b') = sum_i w_i * d_i(b_i, b'_i)``. Weights are
    declared for every dimension or for none: with none declared they default
    to the uniform ``1/n``, under which the distance is the mean of the
    per-dimension distances and lies in ``[0, 1]`` (Prop. separable), as in the
    paper's rover example with weights 1/2, 1/2.
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
        self._behaviour_distance_cache  = {}

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
        invalid = [name for name, dim in self.dimensions.items() if not dim.weight > 0]
        if invalid:
            raise ValueError(f'weights must be positive (Def. feature); got non-positive for {invalid}')

    # ------------------------------------------------------------------
    # Indicators
    # ------------------------------------------------------------------

    def behaviours(self, plans):
        """The distinct behaviours exhibited by the given plans."""
        return set(self._behaviours_of(plans))

    def b_coverage(self, plans):
        """The B-Coverage indicator: the number of distinct behaviours the
        given plans cover."""
        return len(self.behaviours(plans))

    def b_maxsum(self, plans):
        """The B-MaxSum indicator: the sum of pairwise distances between the
        distinct behaviours. Duplicates are discarded before aggregation."""
        distinct = self._distinct_behaviours(plans)
        return sum(self._pair_distance(distinct[i], distinct[j])
                   for i in range(len(distinct)) for j in range(i + 1, len(distinct)))

    def b_maxmin(self, plans):
        """The B-MaxMin indicator: the smallest pairwise distance between the
        distinct behaviours.
        """
        distinct = self._distinct_behaviours(plans)
        if len(distinct) < 2: return 0.0
        return min(self._pair_distance(distinct[i], distinct[j])
                   for i in range(len(distinct)) for j in range(i + 1, len(distinct)))

    def b_novelty(self, plans, k_nn=DEFAULT_K_NN):
        """The B-Novelty indicator: the mean, over the distinct behaviours, of
        each behaviour's mean distance to its ``k' = min(k_nn, b - 1)`` nearest
        neighbours.
        """
        distinct = self._distinct_behaviours(plans)
        b = len(distinct)
        if b < 2: return 0.0
        k_prime  = min(k_nn, b - 1)
        # A behaviour is not its own neighbour; inf keeps the diagonal out of
        # every k-smallest without excluding it index by index.
        distances = self._behaviour_distance_matrix(distinct)
        np.fill_diagonal(distances, np.inf)
        nearest   = np.partition(distances, k_prime - 1, axis=1)[:, :k_prime]
        return float(nearest.mean(axis=1).mean())

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, plans, k, indicator='bcoverage', k_nn=DEFAULT_K_NN):
        """Select k plans from the given pool, maximising the chosen indicator.

        k plans come back whenever the pool holds that many, however the
        indicator moves across the steps -- B-MaxMin can only fall as the
        selection grows, a minimum over pairs never rising when a pair is
        added. The paper returns the k plans the greedy selects rather than
        truncating to the best-scoring prefix: the indicator is to certify a
        set of the requested size, not to choose that size.
        """
        plans = list(plans)
        extractors = {
            'bcoverage': self._extract_b_coverage,
            'bmaxsum':   partial(self._extract_greedy, aggregate=np.sum),
            'bmaxmin':   partial(self._extract_greedy, aggregate=np.min),
            'bnovelty':  partial(self._extract_b_novelty, k_nn=k_nn),
        }
        if indicator not in extractors:
            raise ValueError(f"unknown indicator '{indicator}'; valid indicators: {sorted(extractors)}")
        # Every plan of the pool gets its `behaviour` attribute, whichever rule
        # runs and whichever early return it takes.
        self._behaviours_of(plans)
        return extractors[indicator](plans, k)

    def _extract_b_coverage(self, plans, k):
        """One plan per behaviour, the cheapest exhibiting it, then pad.

        No distances are read, so this never builds the matrix the other three
        rules open on. Nor is the scan a heuristic: every plan covers exactly
        one behaviour, which is what makes greedy selection exact here and
        approximate everywhere else (Thm. bcov-greedy).

        Which plan represents a behaviour is left open by the theorem, and the
        paper takes the cheapest plan in the pool that exhibits it, as
        MAP-Elites keeps the fittest solution per cell; ties fall to the
        earliest plan. Behaviours are taken in first-occurrence order.
        """
        if k <= 0 or not plans:
            return []

        cheapest = {}   # behaviour -> index of its cheapest plan, in first-occurrence order
        costs = self._costs_of(plans)
        for idx, (behaviour, cost) in enumerate(zip(self._behaviours_of(plans), costs)):
            if behaviour not in cheapest or cost < costs[cheapest[behaviour]]:
                cheapest[behaviour] = idx
        covered = np.fromiter(cheapest.values(), dtype=np.intp, count=len(cheapest))[:k]

        # A repeat leaves the count untouched, so once the behaviours run out
        # the tail is padded in plan-index order, as the greedy rules pad theirs.
        rest = np.setdiff1d(np.arange(len(plans)), covered)
        return [plans[idx] for idx in np.concatenate([covered, rest])[:k]]

    def _extract_greedy(self, plans, k, *, aggregate):
        """This is reimplementation of IBM DiverseScore greedy selection.

        `aggregate` is the only thing separating B-MaxSum from B-MaxMin, and
        IBM has no such switch -- their inner loop always sums. np.sum is the
        faithful port; np.min is farthest-point, taking at each step the
        candidate whose *nearest* selected behaviour lies furthest away.

        The seed needs no split: over two behaviours there is one pair, so sum
        and min are the same number and the farthest pair opens both.
        """
        # Guards taken from compute_metrics_greedy: a subset of one, or a pool
        # of one, never reaches the seeding step -- there is no pair to open on.
        if k <= 0 or not plans:
            return []
        if k == 1 or len(plans) == 1:
            return plans[:k]

        # Step 1 -- seed_with_best_pair.
        matrix, codes = self._plan_distance_matrix(plans)
        selected, candidates = self._seed_with_best_pair(matrix)

        # Step 2 -- find_best_next_candidate, until k plans are held or the pool
        # runs out. The C++ loop's other two exit conditions both reduce to an
        # empty `candidates`: it is the seeds' complement, so it empties exactly
        # when the selection has taken the whole pool.
        while len(selected) < k and len(candidates):
            best, candidates = self._find_best_next_candidate(
                matrix, codes, selected, candidates, aggregate)
            selected.append(best)

        return [plans[idx] for idx in selected]

    def _plan_distance_matrix(self, plans):
        behaviours = self._behaviours_of(plans)
        distinct   = list(dict.fromkeys(behaviours))
        code_of    = {behaviour: code for code, behaviour in enumerate(distinct)}
        codes      = np.fromiter(map(code_of.get, behaviours), dtype=np.intp, count=len(behaviours))
        return self._behaviour_distance_matrix(distinct)[np.ix_(codes, codes)], codes

    def _behaviour_distance_matrix(self, distinct):
        # A fresh b x b array each call: callers write into it -- b_novelty
        # fills the diagonal with inf -- and the distances themselves are
        # cached, so a second call re-reads rather than recomputes.
        rows, cols = np.triu_indices(len(distinct), k=1)
        compact    = np.zeros((len(distinct), len(distinct)))
        compact[rows, cols] = np.fromiter(
            (self._pair_distance(distinct[i], distinct[j])
             for i, j in zip(rows.tolist(), cols.tolist())),
            dtype=float, count=rows.size)
        # Out-of-place: `compact += compact.T` reads cells the same statement
        # is writing.
        return compact + compact.T

    def _seed_with_best_pair(self, matrix):
        rows, cols = np.triu_indices(len(matrix), k=1)
        best = best_index(matrix[rows, cols])
        selected = [int(rows[best]), int(cols[best])]
        candidates = np.setdiff1d(np.arange(len(matrix)), selected)
        return selected, candidates

    def _find_best_next_candidate(self, matrix, codes, selected, candidates, aggregate):
        # The C++ inner loop is a row-slice reduction over the selected, so the
        # whole double scan is one aggregate(..., axis=1).
        scores = aggregate(matrix[np.ix_(candidates, selected)], axis=1)
        fresh = ~np.isin(codes[candidates], codes[selected])
        scores = np.where(fresh, scores, -np.inf) if fresh.any() else np.zeros(len(candidates))
        position = best_index(scores)
        return int(candidates[position]), np.delete(candidates, position)

    def _extract_b_novelty(self, plans, k, k_nn=DEFAULT_K_NN):
        """The plain greedy of the paper: at every step add the plan maximising
        B-Novelty over the *combined* set.

        Unlike B-MaxSum and B-MaxMin this is not a reduction over the candidate's
        distances to the selection -- adding a behaviour moves the neighbourhood
        of every behaviour already held -- so it cannot ride the `aggregate`
        slot and scores the whole resulting set instead.

        The farthest pair still opens it. Over a two-behaviour set each one's
        single neighbour is the other, so B-Novelty of a pair is just the
        distance between them, and the farthest pair maximises the indicator
        over every two-plan set exactly as it does for the other two rules.

        Candidates are ranked among the plans whose behaviour is *new* to the
        selection, the convention the paper states for B-MaxSum and that all
        three greedy rules share: a duplicate leaves the indicator exactly where
        it stands, so it is taken only once every remaining candidate repeats a
        held behaviour. B-Novelty is the one rule for which this needs saying,
        because it is not monotone: a fresh behaviour can lower the value below
        what a duplicate would have preserved, and the fresh one is still taken.
        The indicator reported for the returned set makes that fall visible.
        """
        if k <= 0 or not plans:
            return []
        if k == 1 or len(plans) == 1:
            return plans[:k]

        matrix, codes = self._plan_distance_matrix(plans)
        selected, candidates = self._seed_with_best_pair(matrix)
        # B-Novelty reads only the distinct behaviours, so it is scored on the
        # b x b block rather than the n x n one. `codes` is assigned in
        # first-occurrence order, so np.unique returns the codes already sorted
        # and `reps` the plan that first exhibited each.
        reps = np.unique(codes, return_index=True)[1]
        distances = matrix[np.ix_(reps, reps)]

        while len(selected) < k and len(candidates):
            held  = np.unique(codes[selected])
            fresh = ~np.isin(codes[candidates], held)
            if not fresh.any():
                # Nothing new left to add, and a repeat leaves the indicator
                # exactly where it stands, so the tail is padded in plan order.
                selected.append(int(candidates[0]))
                candidates = np.delete(candidates, 0)
                continue

            # Priced once per distinct candidate behaviour: a repeat of one
            # already priced scores the same.
            new = np.unique(codes[candidates][fresh])
            s = len(held)
            k_prime = min(k_nn, s)            # k' = min(k_nn, |T| - 1), |T| = s + 1

            # to_held[i, j]: candidate behaviour i to held behaviour j.
            to_held = distances[np.ix_(new, held)]
            # A behaviour is not its own neighbour; inf keeps the diagonal out
            # of every k-smallest without special-casing it.
            among_held = distances[np.ix_(held, held)].copy()
            np.fill_diagonal(among_held, np.inf)

            # What each held behaviour's neighbourhood becomes once the
            # candidate joins: its distances to the rest, plus the candidate's.
            augmented = np.concatenate(
                [np.broadcast_to(among_held, (len(new), s, s)), to_held[:, :, None]],
                axis=2)
            held_means = np.partition(
                augmented, k_prime - 1, axis=2)[:, :, :k_prime].mean(axis=2)
            # The candidate's own neighbours are the selection entire.
            own_mean = np.partition(
                to_held, k_prime - 1, axis=1)[:, :k_prime].mean(axis=1)
            values = (held_means.sum(axis=1) + own_mean) / (s + 1)

            # Scattered back over the pool in index order, so argmax breaks a
            # tie towards the lowest plan index. `new` is sorted, so
            # searchsorted finds the value priced for a candidate's behaviour.
            scores = np.full(len(candidates), -np.inf)
            scores[fresh] = values[np.searchsorted(new, codes[candidates][fresh])]
            position = best_index(scores)
            selected.append(int(candidates[position]))
            candidates = np.delete(candidates, position)

        return [plans[idx] for idx in selected]

    # ------------------------------------------------------------------
    # Behaviours, costs and distances
    # ------------------------------------------------------------------

    def _distinct_behaviours(self, plans):
        return list(dict.fromkeys(self._behaviours_of(plans)))

    def _behaviours_of(self, plans):
        result = []
        for plan in plans:
            if id(plan) not in self._behaviour_cache:
                states, cost = self._simulate(plan)
                setattr(plan, 'states', states)
                setattr(plan, 'cost', cost)
                behaviour = ' $$ '.join(dim.plan_behaviour(plan) for dim in self.dimensions.values())
                delattr(plan, 'states')  # the trace is only needed while the tokens are built
                setattr(plan, 'behaviour', behaviour)
                self._behaviour_cache[id(plan)] = behaviour
                self._cost_cache[id(plan)] = cost
            result.append(self._behaviour_cache[id(plan)])
        return result

    def _costs_of(self, plans):
        """Each plan's cost (Def. plan), simulated alongside its behaviour."""
        self._behaviours_of(plans)
        return [self._cost_cache[id(plan)] for plan in plans]

    def _simulate(self, plan):
        """``(states, cost)``; raises InapplicablePlanError when the plan
        cannot be replayed against the task."""
        if self._trace_cache is None:
            return simulate(self.task, plan, self._simulator)
        if id(plan) not in self._trace_cache:
            self._trace_cache[id(plan)] = simulate(self.task, plan, self._simulator)
        return self._trace_cache[id(plan)]

    def _pair_distance(self, b1, b2):
        if len(self.dimensions) == 0: return 0.0
        if (b1, b2) not in self._behaviour_distance_cache:
            self._behaviour_distance_cache[(b1, b2)] = sum(dimension.distance(b1, b2) for dimension in self.dimensions.values())
            self._behaviour_distance_cache[(b2, b1)] = self._behaviour_distance_cache[(b1, b2)]
        return self._behaviour_distance_cache[(b1, b2)]
