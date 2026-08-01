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

class InapplicablePlanError(ValueError):
    """A plan could not be simulated against the task.

    The state trace is what every dimension reads, so a plan that cannot be
    replayed has no behaviour to report. Counting it anyway would silently
    inflate the diversity count with a behaviour no valid plan produced.
    """

class BehaviourDiversityCounter:
    def __init__(self, task, dimensions):
        self.task = task
        self.dimensions = {}
        for name, addinfo in dimensions:
            if name not in dimensions_map:
                raise ValueError(f"unknown dimension '{name}'; valid keys: {sorted(dimensions_map)}")
            self.dimensions[name] = dimensions_map[name](task, addinfo)
        self._simulator = SequentialSimulator(problem=task)
        self._behaviour_cache = {}
        self._distance_cache = {}

    def behaviours(self, plans):
        """The distinct behaviours exhibited by the given plans."""
        return set(self._behaviours_of(plans))

    def bdc(self, plans):
        """The Behaviour Diversity Count indicator: the number of distinct
        behaviours the given plans cover."""
        return len(self.behaviours(plans))

    def b_maxsum(self, plans):
        """The B-MaxSum indicator: the sum of pairwise distances between the
        distinct behaviours. Duplicates are discarded before aggregation."""
        distinct = list(dict.fromkeys(self._behaviours_of(plans)))
        return sum(self._pair_distance(distinct[i], distinct[j])
                   for i in range(len(distinct)) for j in range(i + 1, len(distinct)))

    def extract(self, plans, k, indicator='bdc'):
        """Select k plans from the given pool, maximising the chosen indicator."""
        extractors = {'bdc': self._extract_bdc, 'bmaxsum': self._extract_b_maxsum}
        if indicator not in extractors:
            raise ValueError(f"unknown indicator '{indicator}'; valid indicators: {sorted(extractors)}")
        plans = list(plans)
        return extractors[indicator](plans, self._behaviours_of(plans), k)

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

    def _extract_bdc(self, plans, behaviours, k):
        # Scan the pool in order, taking a plan only when its behaviour is new; once
        # every behaviour is covered, fill the remaining slots with duplicates, which
        # leave the indicator unchanged.
        chosen, seen = [], set()
        for idx, behaviour in enumerate(behaviours):
            if len(chosen) == k: break
            if behaviour not in seen:
                seen.add(behaviour)
                chosen.append(idx)
        for idx in range(len(plans)):
            if len(chosen) == k: break
            if idx not in chosen:
                chosen.append(idx)
        return [plans[idx] for idx in chosen]

    def _extract_b_maxsum(self, plans, behaviours, k):
        # Greedy: repeatedly add the plan whose behaviour has the greatest summed
        # distance to the behaviours already selected. The first pick is arbitrary
        # (singleton sets score zero), and duplicates gain nothing, so they are only
        # picked once every remaining candidate repeats a selected behaviour.
        # Each candidate's gain is accumulated as behaviours join the selection,
        # rather than being re-summed over the whole selection every round.
        remaining = list(range(len(plans)))
        chosen, selected = [], set()
        gains = [0.0] * len(plans)
        while remaining and len(chosen) < k:
            best = max(remaining, key=lambda idx: 0.0 if behaviours[idx] in selected else gains[idx])
            remaining.remove(best)
            chosen.append(best)
            if behaviours[best] not in selected:
                selected.add(behaviours[best])
                for idx in remaining:
                    gains[idx] += self._pair_distance(behaviours[idx], behaviours[best])
        return [plans[idx] for idx in chosen]

    def _pair_distance(self, b1, b2):
        # Distances are symmetric, so the pair is cached unordered.
        if len(self.dimensions) == 0:
            return 0.0
        key = (b1, b2) if b1 <= b2 else (b2, b1)
        if key not in self._distance_cache:
            self._distance_cache[key] = sum(d.distance(b1, b2) for d in self.dimensions.values()) / len(self.dimensions)
        return self._distance_cache[key]
