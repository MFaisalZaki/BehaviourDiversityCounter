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
    def __init__(self, task, plans, dimensions):
        self.task  = task
        self.plans = list(plans)
        self.dimensions = {}
        for name, addinfo in dimensions:
            if name not in dimensions_map:
                raise ValueError(f"unknown dimension '{name}'; valid keys: {sorted(dimensions_map)}")
            self.dimensions[name] = dimensions_map[name](task, addinfo)
        self._simulator = SequentialSimulator(problem=task)
        self._plan_behaviours = None

    @property
    def behaviours(self):
        """The distinct behaviours exhibited by the plan set."""
        return set(self._infer_behaviours())

    def bdc(self):
        """The Behaviour Diversity Count indicator: the number of distinct
        behaviours the plan set covers."""
        return len(self.behaviours)

    def b_maxsum(self):
        """The B-MaxSum indicator: the sum of pairwise distances between the
        distinct behaviours. Duplicates are discarded before aggregation."""
        distinct = list(dict.fromkeys(self._infer_behaviours()))
        return sum(self._pair_distance(distinct[i], distinct[j])
                   for i in range(len(distinct)) for j in range(i + 1, len(distinct)))

    def extract(self, k, indicator='bdc'):
        """Select k plans from the pool, maximising the chosen indicator."""
        extractors = {'bdc': self._extract_bdc, 'bmaxsum': self._extract_b_maxsum}
        if indicator not in extractors:
            raise ValueError(f"unknown indicator '{indicator}'; valid indicators: {sorted(extractors)}")
        self._infer_behaviours()
        return extractors[indicator](k)

    def _infer_behaviours(self):
        # One simulation pass shared by every indicator: replay each plan, join its
        # per-dimension tokens, and annotate the plan object with the result.
        if self._plan_behaviours is None:
            self._plan_behaviours = []
            for plan in self.plans:
                setattr(plan, 'states', self._simulate(plan))
                behaviour = ' $$ '.join(dim.plan_behaviour(plan) for dim in self.dimensions.values())
                delattr(plan, 'states')  # the trace is only needed while the tokens are built
                setattr(plan, 'behaviour', behaviour)
                self._plan_behaviours.append(behaviour)
        return self._plan_behaviours

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

    def _extract_bdc(self, k):
        # Scan the pool in order, taking a plan only when its behaviour is new; once
        # every behaviour is covered, fill the remaining slots with duplicates, which
        # leave the indicator unchanged.
        chosen, seen = [], set()
        for idx, p in enumerate(self.plans):
            if len(chosen) == k: break
            if p.behaviour not in seen:
                seen.add(p.behaviour)
                chosen.append(idx)
        for idx in range(len(self.plans)):
            if len(chosen) == k: break
            if idx not in chosen:
                chosen.append(idx)
        return [self.plans[idx] for idx in chosen]

    def _extract_b_maxsum(self, k):
        # Greedy: repeatedly add the plan whose behaviour has the greatest summed
        # distance to the behaviours already selected. The first pick is arbitrary
        # (singleton sets score zero), and duplicates gain nothing, so they are only
        # picked once every remaining candidate repeats a selected behaviour.
        remaining = list(range(len(self.plans)))
        chosen, behaviours = [], []
        while remaining and len(chosen) < k:
            def gain(idx):
                b = self.plans[idx].behaviour
                return 0.0 if b in behaviours else sum(self._pair_distance(b, other) for other in behaviours)
            best = max(remaining, key=gain)
            remaining.remove(best)
            chosen.append(best)
            if self.plans[best].behaviour not in behaviours:
                behaviours.append(self.plans[best].behaviour)
        return [self.plans[idx] for idx in chosen]

    def _pair_distance(self, b1, b2):
        return sum(d.distance(b1, b2) for d in self.dimensions.values()) / len(self.dimensions) if len(self.dimensions) > 0 else 0.0
