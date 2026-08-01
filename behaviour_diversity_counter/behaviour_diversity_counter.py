from collections import defaultdict
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
    def __init__(self, task, planlist, dimensions):
        self.task       = task
        self.planslist  = list(planlist)
        self.dimensions = {name: dimensions_map[name](task, addinfo) for name, addinfo in dimensions}
        self.collected_behaviours = set()
        self.estimated_behaviours = set()
        self._estimated_behaviour_count = -1
        self._simulator = SequentialSimulator(problem=task)

    def _simulate_(self, plan):
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
    
    def _extract_behaviour_(self, plan, states):
        setattr(plan, 'states', states)
        return ' $$ '.join([dim.plan_behaviour(plan) for name, dim in self.dimensions.items()])

    def count(self):
        if len(self.collected_behaviours) == 0: self.optimise(k=len(self.planslist))
        return len(self.collected_behaviours)
    
    def _infer_plan_behaviour(self, plan):
        return self._extract_behaviour_(plan, self._simulate_(plan))
    
    def optimise(self, k):
        _behaviours = defaultdict(list)
        _ret_plans = []
        for idx, p in enumerate(self.planslist):
            states    = self._simulate_(p)
            behaviour = self._extract_behaviour_(p, states)
            self.collected_behaviours.add(behaviour)
            setattr(self.planslist[idx], 'behaviour', behaviour)
            _behaviours[behaviour].append(self.planslist[idx])
        
        while not all([len(v) == 0 for v in _behaviours.values()]) and len(_ret_plans) < k:
            for key in _behaviours.keys():
                if len(_ret_plans) >= k: break
                if len(_behaviours[key]) == 0: continue
                _ret_plans.append(_behaviours[key].pop())
        
        # estimate the maximum behaviour count.
        self._estimated_behaviour_count = 1
        for name, dimension in self.dimensions.items():
            dimension._estimate_domain()
            self._estimated_behaviour_count *= dimension.estimated_domain_size
            
        return _ret_plans
    
    def estimated_behaviour_count(self):
        return self._estimated_behaviour_count
    
    def compute_b_maxsum_metric(self):
        def pair_distance(b1, b2):
            return sum(d.distance(b1, b2) for d in self.dimensions.values()) / len(self.dimensions) if len(self.dimensions) > 0 else 0.0
        
        _behaviours = [self._infer_plan_behaviour(p) for p in self.planslist]
        n = len(_behaviours)
        # Cache the symmetric matrix once — pair_distance is the hot path.
        dmat = []
        for i in range(n):
            for j in range(i + 1, n):
                dmat.append(pair_distance(_behaviours[i], _behaviours[j]))
        return sum(dmat)/len(dmat) if len(dmat) > 0 else 0.0