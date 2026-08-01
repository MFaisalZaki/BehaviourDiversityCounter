from behaviour_diversity_counter.dimensions.base import BehaviourDimension

class MakespanOptimalCostDimension(BehaviourDimension):
    def __init__(self, task, addinfo):
        super().__init__(task, 'cb', addinfo)
    
    def _estimate_domain(self):
        _q_value = self.addinfo.get('q', None)
        assert _q_value is not None, 'The q value should be provided for the cost bound dimension.'
        if _q_value == 1.0:
            self.estimated_domain_size = 1
            return
        _optimal_cost = min(self.domain)
        _max_cost = int(_optimal_cost * _q_value)
        self.estimated_domain_size = len(set(range(_optimal_cost, _max_cost + 1)))

    def plan_behaviour(self, plan):
        self.domain.add(len(plan.actions))
        return f'{self.name}:' + str(len(plan.actions))

    def _cost(self, plan):
        # Match on the token prefix, not a substring: other dimensions' tokens may
        # contain this dimension's name inside predicate/object names.
        token = next(filter(lambda e: e.strip().startswith(self.name + ':'), plan.split('$$')), None)
        assert token is not None, 'The dimension value should be present in the plan behaviour.'
        return int(token.strip().replace(self.name + ':', '').strip())

    def distance(self, plan1, plan2):
        # Takes behaviour strings, like every other dimension: this is what
        # BehaviourDiversityCounter.compute_b_maxsum_metric passes in.
        cost1, cost2 = self._cost(plan1), self._cost(plan2)
        if max(cost1, cost2) == 0:
            return 0.0
        # Normalised into [0, 1] so the score can be averaged with the other dimensions.
        return abs(cost1 - cost2) / max(cost1, cost2)
