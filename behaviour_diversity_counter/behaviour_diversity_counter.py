from collections import defaultdict
from unified_planning.shortcuts import SequentialSimulator

from behaviour_diversity_counter.features.goal_predicate_ordering import GoalPredicatesOrderingSimulator
from behaviour_diversity_counter.features.cost_bound_makespan_optimal import MakespanOptimalCostSimulator
from behaviour_diversity_counter.features.resources import ResourceCountSimulator
from behaviour_diversity_counter.features.utility_value import UtilityValueSimulator
from behaviour_diversity_counter.features.functions import FunctionsSimulator

features_map = {
    'go': GoalPredicatesOrderingSimulator,
    'cb': MakespanOptimalCostSimulator,
    'ru': ResourceCountSimulator,
    'uv': UtilityValueSimulator,
    'fn': FunctionsSimulator
}

class BehaviourDiversityCounter:
    def __init__(self, task, planlist, f):
        self.task      = task
        self.planslist = list(planlist)
        self.features  = {feat_name: features_map[feat_name](task, addinfo) for feat_name, addinfo in f}
        self.collected_behaviours = set()
        self.estimated_behaviours = set()
        self._estimated_behaviour_count = -1

    def _simulate_(self, plan):
        states = []
        with SequentialSimulator(problem=self.task) as simulator:
            initial_state = simulator.get_initial_state()
            current_state = initial_state
            states += [current_state]
            for action_instance in plan.actions:
                current_state = simulator.apply(current_state, action_instance)
                if current_state is None: return []
                states.append(current_state)
        return states
    
    def _extract_behaviour_(self, plan, states):
        setattr(plan, 'states', states)
        return ' $$ '.join([dim.plan_behaviour(plan) for name, dim in self.features.items()])

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
        for f, feature in self.features.items():
            feature._estimate_domain()
            self._estimated_behaviour_count *= feature.estimated_domain_size
            
        return _ret_plans
    
    def estimated_behaviour_count(self):
        return self._estimated_behaviour_count
    
    def compute_novelty_score(self):
        def pair_distance(b1, b2):
            return sum(f.distance(b1, b2) for f in self.behaviour_counter.features.values()) / len(self.behaviour_counter.features) if len(self.behaviour_counter.features) > 0 else 0.0
        
        _behaviours = [self._infer_plan_behaviour(p) for p in self.plans]
        n = len(_behaviours)
        # Cache the symmetric matrix once — pair_distance is the hot path.
        dmat = []
        for i in range(n):
            for j in range(i + 1, n):
                dmat.append(pair_distance(_behaviours[i], _behaviours[j]))
        return sum(dmat)/len(dmat) if len(dmat) > 0 else 0.0