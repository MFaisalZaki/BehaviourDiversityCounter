import heapq

from collections import namedtuple

from unified_planning.shortcuts import SequentialSimulator

from behaviour_diversity_counter.dimensions.goal_predicate_ordering import GoalPredicatesOrderingDimension
from behaviour_diversity_counter.dimensions.cost_bound_makespan_optimal import MakespanOptimalCostDimension
from behaviour_diversity_counter.dimensions.resources import ResourceCountDimension, ResourceUsedDimension
from behaviour_diversity_counter.dimensions.utility_value import UtilityValueDimension
from behaviour_diversity_counter.dimensions.functions import NumericFunctionDimension

dimensions_map = {
    'go': GoalPredicatesOrderingDimension,
    'cb': MakespanOptimalCostDimension,
    'rc': ResourceCountDimension,
    'ru': ResourceUsedDimension,
    'uv': UtilityValueDimension,
    'fn': NumericFunctionDimension
}

DEFAULT_K_NN = 3

#: Two candidate scores this close are treated as *tied*, and the tie is broken
#: by plan index rather than by whichever float happens to be larger.
#:
#: Greedy selection compares scores that are sums of the same distances taken in
#: different orders, so two mathematically equal candidates routinely differ by
#: one unit in the last place. Letting that decide the pick is reproducible but
#: not stable: an unrelated change to how a score is accumulated would silently
#: return a different selection, and Experiments B and C read a changed
#: selection as a real disagreement between rules.
SCORE_TOLERANCE = 1e-12


def strictly_better(value, best):
    """Is ``value`` a real improvement on ``best``, rather than float noise?

    ``best`` of ``None`` means nothing has been picked yet, so anything wins.
    """
    if best is None:
        return True
    return value - best > SCORE_TOLERANCE * max(1.0, abs(value), abs(best))

#: What an extraction did, step by step.
#:
#: ``plans`` is what the caller should use: the highest-scoring *prefix* of the
#: greedy order. For B-Coverage and B-MaxSum that is always the whole order,
#: because both are monotone in the selection; for B-MaxMin and B-Novelty it is
#: not, and the prefix is genuinely shorter (see :meth:`extract`).
#: ``order`` is the full k-step pick order and ``scores[i]`` the indicator value
#: of ``order[:i + 1]``, so the trace is reportable in its own right.
Selection = namedtuple('Selection', 'plans order scores best_step')


class InapplicablePlanError(ValueError):
    """A plan could not be simulated against the task.

    The state trace is what every dimension reads, so a plan that cannot be
    replayed has no behaviour to report. Counting it anyway would silently
    inflate the diversity count with a behaviour no valid plan produced.
    """

class BehaviourDiversityCounter:
    def __init__(self, task, dimensions, weights=None):
        self.task = task
        self.dimensions = {}
        for name, addinfo in dimensions:
            if name not in dimensions_map:
                raise ValueError(f"unknown dimension '{name}'; valid keys: {sorted(dimensions_map)}")
            self.dimensions[name] = dimensions_map[name](task, addinfo)
        self._simulator = SequentialSimulator(problem=task)
        self._behaviour_cache = {}
        self._distance_cache = {}
        self._weights = {}
        self._counters_enabled = False
        self._simulator_apply = None
        self.reset_counters()
        self.set_weights(weights)

    # ------------------------------------------------------------------
    # Instrumentation
    # ------------------------------------------------------------------

    def enable_counters(self, enabled=True):
        """Turn the per-run counters on (or off again), resetting them.

        Off by default, and off costs nothing at all: enabling swaps the hot
        methods for counting variants through the instance dictionary rather
        than leaving a flag to test on every distance lookup, which is the one
        call Experiment A is trying to time.

        The cache hits are counted apart from the calls because the cache is
        precisely what makes a behaviour distance cheap -- there are only b
        distinct behaviours to compare, however many plans exhibit them -- and
        that is part of the result rather than an implementation detail.
        """
        enabled = bool(enabled)
        self.reset_counters()
        if enabled == self._counters_enabled:
            return
        self._counters_enabled = enabled
        if enabled:
            self._pair_distance = self._counted_pair_distance
            if self._simulator is not None:
                self._simulator_apply = self._simulator.apply
                self._simulator.apply = self._counted_apply
        else:
            self.__dict__.pop('_pair_distance', None)
            if self._simulator_apply is not None:
                self._simulator.apply = self._simulator_apply
                self._simulator_apply = None

    def reset_counters(self):
        """Zero the counters, so one counter object can time many runs."""
        self.n_distance_evals = 0
        self.n_distance_misses = 0
        self.n_simulator_calls = 0

    @property
    def counters(self):
        return {
            'n_distance_evals': self.n_distance_evals,
            'n_distance_misses': self.n_distance_misses,
            'n_simulator_calls': self.n_simulator_calls,
        }

    def _counted_apply(self, *args, **kwargs):
        self.n_simulator_calls += 1
        return self._simulator_apply(*args, **kwargs)

    def _counted_pair_distance(self, b1, b2):
        # Deliberately a parallel copy of _pair_distance rather than a wrapper
        # around it: a wrapper would build the cache key twice per call, which
        # would inflate exactly the measurement this exists to take. The one
        # thing both need to agree on -- the distance itself -- is shared.
        self.n_distance_evals += 1
        if len(self.dimensions) == 0:
            return 0.0
        key = (b1, b2) if b1 <= b2 else (b2, b1)
        if key not in self._distance_cache:
            self.n_distance_misses += 1
            self._distance_cache[key] = self._compute_pair_distance(b1, b2)
        return self._distance_cache[key]

    # ------------------------------------------------------------------
    # Weights
    # ------------------------------------------------------------------

    @property
    def weights(self):
        """The per-dimension weights of the separable distance, name -> float."""
        return dict(self._weights)

    def set_weights(self, weights=None):
        """Set the weights of ``d(b, b') = sum_i w_i * d_i(b_i, b'_i)``.

        ``None`` restores the uniform ``1/n``, under which the distance is the
        mean over the dimensions -- what the counter computed before weights
        existed. Setting the weights clears the pair-distance cache: that cache
        is keyed by the behaviour pair alone, so a stale entry would silently
        answer with the *previous* weight vector.
        """
        if weights is None:
            n = len(self.dimensions)
            weights = {name: 1.0 / n for name in self.dimensions} if n else {}
        else:
            unknown = set(weights) - set(self.dimensions)
            if unknown:
                raise ValueError(f'unknown dimension(s) in weights: {sorted(unknown)}; '
                                 f'valid keys: {sorted(self.dimensions)}')
            missing = set(self.dimensions) - set(weights)
            if missing:
                raise ValueError(f'no weight given for dimension(s): {sorted(missing)}')
            for name, weight in weights.items():
                if weight < 0:
                    raise ValueError(f"weight for '{name}' is negative: {weight}")
            weights = {name: float(weights[name]) for name in self.dimensions}
        self._weights = weights
        self._distance_cache.clear()

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

    def bdc(self, plans):
        """Deprecated alias of :meth:`b_coverage`, under the indicator's former
        name (Behaviour Diversity Count)."""
        return self.b_coverage(plans)

    def b_maxsum(self, plans):
        """The B-MaxSum indicator: the sum of pairwise distances between the
        distinct behaviours. Duplicates are discarded before aggregation."""
        distinct = self._distinct_behaviours(plans)
        return sum(self._pair_distance(distinct[i], distinct[j])
                   for i in range(len(distinct)) for j in range(i + 1, len(distinct)))

    def b_maxmin(self, plans):
        """The B-MaxMin indicator: the smallest pairwise distance between the
        distinct behaviours.

        Fewer than two distinct behaviours score ``0``, not ``+inf``: a set that
        offers the user no alternative at all should rank lowest, and the empty
        minimum's usual convention would rank it highest.
        """
        distinct = self._distinct_behaviours(plans)
        if len(distinct) < 2:
            return 0.0
        return min(self._pair_distance(distinct[i], distinct[j])
                   for i in range(len(distinct)) for j in range(i + 1, len(distinct)))

    def b_novelty(self, plans, k_nn=DEFAULT_K_NN):
        """The B-Novelty indicator: the mean, over the distinct behaviours, of
        each behaviour's mean distance to its ``k' = min(k_nn, b - 1)`` nearest
        neighbours.

        Fewer than two distinct behaviours score ``0``, for the same reason as
        :meth:`b_maxmin`. Note that when ``b <= k_nn`` the clamp leaves every
        behaviour averaging over *all* the others, so the indicator collapses to
        the mean pairwise distance -- B-MaxSum divided by C(b, 2).
        """
        distinct = self._distinct_behaviours(plans)
        b = len(distinct)
        if b < 2:
            return 0.0
        k_prime = min(k_nn, b - 1)
        total = 0.0
        for i, behaviour in enumerate(distinct):
            others = [self._pair_distance(behaviour, distinct[j])
                      for j in range(b) if j != i]
            total += sum(heapq.nsmallest(k_prime, others)) / k_prime
        return total / b

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(self, plans, k, indicator='bcoverage', k_nn=DEFAULT_K_NN, trace=False):
        """Select k plans from the given pool, maximising the chosen indicator.

        B-MaxMin and B-Novelty are *not* monotone: adding a plan can lower them,
        so what is returned is the highest-scoring prefix of the greedy order
        rather than its final k plans. ``trace=True`` returns the whole
        :class:`Selection` -- the k-step order and the score after every step --
        so that non-monotonicity is reportable rather than merely worked around.
        """
        extractors = {
            'bcoverage': self._extract_b_coverage,
            'bdc': self._extract_b_coverage,   # the indicator's former name
            'bmaxsum': self._extract_b_maxsum,
            'bmaxmin': self._extract_b_maxmin,
            'bnovelty': self._extract_b_novelty,
        }
        if indicator not in extractors:
            raise ValueError(f"unknown indicator '{indicator}'; valid indicators: {sorted(extractors)}")
        plans = list(plans)
        selection = extractors[indicator](plans, self._behaviours_of(plans), k, k_nn)
        return selection if trace else selection.plans

    def _extract_b_coverage(self, plans, behaviours, k, k_nn=None):
        # Scan the pool in order, taking a plan only when its behaviour is new; once
        # every behaviour is covered, fill the remaining slots with duplicates, which
        # leave the indicator unchanged.
        chosen, seen, scores = [], set(), []
        for idx, behaviour in enumerate(behaviours):
            if len(chosen) == k: break
            if behaviour not in seen:
                seen.add(behaviour)
                chosen.append(idx)
                scores.append(float(len(seen)))
        covered = len(seen)
        for idx in range(len(plans)):
            if len(chosen) == k: break
            if idx not in chosen:
                chosen.append(idx)
                scores.append(float(covered))
        return self._selection(plans, chosen, scores, monotone=True)

    def _extract_b_maxsum(self, plans, behaviours, k, k_nn=None):
        # Greedy: repeatedly add the plan whose behaviour has the greatest summed
        # distance to the behaviours already selected. The first pick is arbitrary
        # (singleton sets score zero), and duplicates gain nothing, so they are only
        # picked once every remaining candidate repeats a selected behaviour.
        # Each candidate's gain is accumulated as behaviours join the selection,
        # rather than being re-summed over the whole selection every round.
        remaining = list(range(len(plans)))
        chosen, selected, scores = [], set(), []
        gains = [0.0] * len(plans)
        total = 0.0
        while remaining and len(chosen) < k:
            # `remaining` is in ascending index order, so the first candidate to
            # clear the tie tolerance is the lowest-indexed of the tied best.
            best, best_gain = None, None
            for idx in remaining:
                gain = 0.0 if behaviours[idx] in selected else gains[idx]
                if strictly_better(gain, best_gain):
                    best, best_gain = idx, gain
            remaining.remove(best)
            chosen.append(best)
            if behaviours[best] not in selected:
                total += gains[best]
                selected.add(behaviours[best])
                for idx in remaining:
                    gains[idx] += self._pair_distance(behaviours[idx], behaviours[best])
            scores.append(total)
        return self._selection(plans, chosen, scores, monotone=True)

    def _extract_b_maxmin(self, plans, behaviours, k, k_nn=None):
        # Farthest-first: seed with the two behaviours at maximum distance, then
        # repeatedly add the plan whose behaviour is farthest from the nearest
        # behaviour already chosen. The set's value is the running minimum, so a
        # third pick can only lower it -- which is exactly why the caller is handed
        # the best prefix and the trace rather than the final k plans.
        if k <= 0 or not plans:
            return self._selection(plans, [], [], monotone=False)
        distinct = list(dict.fromkeys(behaviours))
        first_index = {}
        for idx, behaviour in enumerate(behaviours):
            first_index.setdefault(behaviour, idx)

        if len(distinct) < 2:
            # Nothing to separate: every subset scores zero, so the trace is flat
            # and the best prefix is the single first plan.
            chosen = list(range(min(k, len(plans))))
            return self._selection(plans, chosen, [0.0] * len(chosen), monotone=False)

        # The seed pair. `distinct` is in first-occurrence order, so scanning i < j
        # with a strict improvement test breaks ties towards the lowest plan indices.
        seed, best_distance = (distinct[0], distinct[1]), None
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                distance = self._pair_distance(distinct[i], distinct[j])
                if strictly_better(distance, best_distance):
                    best_distance, seed = distance, (distinct[i], distinct[j])

        chosen = [first_index[seed[0]], first_index[seed[1]]][:k]
        scores = [0.0, best_distance][:k]
        if k < 2:
            return self._selection(plans, chosen, scores, monotone=False)

        picked = set(chosen)
        set_min = best_distance
        nearest = [min(self._pair_distance(behaviour, seed[0]),
                       self._pair_distance(behaviour, seed[1]))
                   for behaviour in behaviours]
        while len(chosen) < k and len(picked) < len(plans):
            best, best_value = None, None
            for idx in range(len(plans)):
                if idx in picked:
                    continue
                if strictly_better(nearest[idx], best_value):
                    best, best_value = idx, nearest[idx]
            picked.add(best)
            chosen.append(best)
            set_min = min(set_min, best_value)
            scores.append(set_min)
            for idx in range(len(plans)):
                if idx not in picked:
                    nearest[idx] = min(nearest[idx],
                                       self._pair_distance(behaviours[idx], behaviours[best]))
        return self._selection(plans, chosen, scores, monotone=False)

    def _extract_b_novelty(self, plans, behaviours, k, k_nn=DEFAULT_K_NN):
        # Greedy on B-Novelty: at every step add the plan that maximises the
        # indicator of the resulting set, among the plans that contribute a
        # behaviour not already selected.
        #
        # The restriction to new behaviours is the convention B-MaxSum and
        # B-Coverage already follow here, and B-Novelty needs it stated because
        # it is the one indicator a duplicate leaves *exactly* unchanged: a
        # duplicate is invisible to the score, while a genuinely new behaviour
        # can lower it. Unrestricted greedy therefore stops covering behaviours
        # the moment the next one would cost anything and pads the rest of k
        # with copies -- on gripper's k=1000 pool it covers four behaviours and
        # fills the other sixteen slots with duplicates, at which point the
        # trace can no longer fall and the highest-scoring-prefix rule this
        # indicator is documented to need would never fire.
        #
        # B-Novelty reads only the *distinct* behaviours, so the value is computed
        # once per distinct candidate behaviour and then scanned back over the pool
        # in index order -- identical to pricing every plan, and the scan is what
        # breaks ties towards the lowest plan index.
        #
        # Only the k_nn smallest distances per chosen behaviour are ever needed
        # (k' = min(k_nn, b - 1) <= k_nn), so each chosen behaviour keeps a short
        # sorted list of them and a candidate costs O(|chosen|) to price rather
        # than O(|chosen|^2).
        if k <= 0 or not plans:
            return self._selection(plans, [], [], monotone=False)
        candidates = list(dict.fromkeys(behaviours))
        remaining_behaviours = set(candidates)

        chosen, picked, scores = [], set(), []
        selected = []          # the distinct behaviours chosen so far, in pick order
        selected_set = set()   # the same, for membership tests
        nearest = []           # nearest[i]: the k_nn smallest distances from selected[i]
        current = 0.0          # B-Novelty of the selection as it stands

        while len(chosen) < k and len(picked) < len(plans):
            m = len(selected)
            k_prime = min(k_nn, m)
            fresh = any(behaviour not in selected_set for behaviour in candidates
                        if behaviour in remaining_behaviours)
            values = {}
            for behaviour in candidates:
                if behaviour in selected_set:
                    # A duplicate leaves the value alone, so it is only ever
                    # taken once no unselected behaviour is left in the pool.
                    values[behaviour] = current if not fresh else -1.0
                elif k_prime == 0:
                    values[behaviour] = 0.0       # a singleton set has no neighbours
                else:
                    to_selected = [self._pair_distance(behaviour, other) for other in selected]
                    total = sum(heapq.nsmallest(k_prime, to_selected)) / k_prime
                    for i, distance in enumerate(to_selected):
                        total += sum(heapq.nsmallest(k_prime, nearest[i] + [distance])) / k_prime
                    values[behaviour] = total / (m + 1)

            best, best_value = None, None
            for idx in range(len(plans)):
                if idx in picked:
                    continue
                value = values[behaviours[idx]]
                if strictly_better(value, best_value):
                    best, best_value = idx, value

            picked.add(best)
            chosen.append(best)
            behaviour = behaviours[best]
            remaining_behaviours.discard(behaviour)
            if behaviour in selected_set:
                best_value = current      # the duplicate's real value, not the -1 sentinel
            scores.append(best_value)
            current = best_value
            if behaviour not in selected_set:
                to_selected = [self._pair_distance(behaviour, other) for other in selected]
                for i, distance in enumerate(to_selected):
                    nearest[i] = heapq.nsmallest(k_nn, nearest[i] + [distance])
                nearest.append(heapq.nsmallest(k_nn, to_selected))
                selected.append(behaviour)
                selected_set.add(behaviour)
        return self._selection(plans, chosen, scores, monotone=False)

    def _selection(self, plans, chosen, scores, monotone):
        """Wrap a greedy order as a :class:`Selection`.

        A monotone indicator never loses by taking one more plan, so its whole
        order is returned. Otherwise the returned set is the *longest* prefix
        that attained the best score.

        Longest, not shortest: only a strict fall is a reason to stop handing
        the user plans, and these indicators plateau constantly -- gripper's 24
        behaviours realise just three distinct pairwise distances between them,
        so a pick that leaves the score untouched is the common case rather than
        the exception. Stopping at the first best prefix would report B-Novelty
        as returning two plans out of a requested twenty on such a pool, which
        is a property of the tie rule and not of the indicator. It would also
        make thm:bmaxmin-degenerate true by construction: B-MaxMin would return
        a pair whether or not the third behaviour actually lowered the minimum,
        which is the one thing the experiment is supposed to measure.
        """
        if not chosen:
            return Selection([], [], [], 0)
        if monotone:
            best_step = len(chosen)
        else:
            best_step, best_score = 1, None
            for step, score in enumerate(scores, start=1):
                if strictly_better(score, best_score):
                    best_step, best_score = step, score
                elif not strictly_better(best_score, score):
                    best_step = step        # tied with the best: keep the plans
        order = [plans[idx] for idx in chosen]
        return Selection(order[:best_step], order, list(scores), best_step)

    # ------------------------------------------------------------------
    # Behaviours and distances
    # ------------------------------------------------------------------

    def _distinct_behaviours(self, plans):
        """The distinct behaviours, in order of first appearance.

        Ordered rather than a ``set``: every indicator sums over pairs of these,
        and a set's iteration order varies between processes, which would make
        the floating-point total non-reproducible.
        """
        return list(dict.fromkeys(self._behaviours_of(plans)))

    def _behaviours_of(self, plans):
        # Replay each plan once per counter: the result is cached by plan identity
        # and attached to the plan object as plan.behaviour.
        result = []
        for plan in plans:
            if id(plan) not in self._behaviour_cache:
                setattr(plan, 'states', self._simulate(plan))
                behaviour = ' $$ '.join(dim.plan_behaviour(plan) for dim in self.dimensions.values())
                delattr(plan, 'states')  # the trace is only needed while the tokens are built
                setattr(plan, 'behaviour', behaviour)
                self._behaviour_cache[id(plan)] = behaviour
            result.append(self._behaviour_cache[id(plan)])
        return result

    def _simulate(self, plan):
        states = []
        initial_state = self._simulator.get_initial_state()
        current_state = initial_state
        states += [current_state]
        for step, action_instance in enumerate(plan.actions):
            next_state = self._simulator.apply(current_state, action_instance)
            if next_state is None:
                raise InapplicablePlanError(
                    f'{action_instance} at step {step} is not applicable to the state '
                    f'it reaches; the plan cannot be simulated against this task.'
                )
            current_state = next_state
            states.append(current_state)
        return states

    def _pair_distance(self, b1, b2):
        # The paper's separable distance, d(b, b') = sum_i w_i * d_i(b_i, b'_i).
        # Distances are symmetric, so the pair is cached unordered; the cache is
        # cleared by set_weights, since it is keyed by the pair alone.
        if len(self.dimensions) == 0:
            return 0.0
        key = (b1, b2) if b1 <= b2 else (b2, b1)
        if key not in self._distance_cache:
            self._distance_cache[key] = self._compute_pair_distance(b1, b2)
        return self._distance_cache[key]

    def _compute_pair_distance(self, b1, b2):
        """The separable distance itself, uncached."""
        return sum(self._weights[name] * dimension.distance(b1, b2)
                   for name, dimension in self.dimensions.items())

