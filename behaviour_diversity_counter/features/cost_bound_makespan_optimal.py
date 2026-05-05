from behaviour_diversity_counter.features.base import DimensionConstructorSimulator

class MakespanOptimalCostSimulator(DimensionConstructorSimulator):
    def __init__(self, task, addinfo):
        super().__init__(task, 'cb', addinfo)
    
    def _estimate_domain(self):
        _q_value = self.addinfo.get('q', None)
        assert _q_value is not None, 'The q value should be provided for the cost bound feature.'
        if _q_value == 1.0:
            self.estimated_domain_size = 1
            return
        _optimal_cost = min(self.domain)
        _max_cost = int(_optimal_cost * _q_value)
        self.estimated_domain_size = len(set(range(_optimal_cost, _max_cost + 1)))

    def plan_behaviour(self, plan):
        self.domain.add(len(plan.actions))
        return f'{self.name}:' + str(len(plan.actions))
