from behaviour_diversity_counter.features.base import DimensionConstructorSimulator

class MakespanOptimalCostSimulator(DimensionConstructorSimulator):
    def __init__(self, task, addinfo):
        super().__init__(task, 'cb', addinfo)
    
    def plan_behaviour(self, plan):
        self.domain.add(len(plan.actions))
        return f'{self.name}:' + str(len(plan.actions))
