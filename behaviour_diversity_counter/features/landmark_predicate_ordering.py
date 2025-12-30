from collections import defaultdict
from behaviour_diversity_counter.features.base import DimensionConstructorSimulator

class LandmarkPredicatesOrderingSimulator(DimensionConstructorSimulator):
    def __init__(self, task, name, addinfo=None):
        super().__init__(task, name, addinfo)
        from unified_planning.model.walkers.free_vars import FreeVarsExtractor
        vars = list(map(lambda expr: FreeVarsExtractor().get(expr), self.task.goals))
        self.vars = [elem for s in vars for elem in s]
    
    def plan_behaviour(self, plan):
        _time_step_history = defaultdict(list)
        for t, state in enumerate(plan.states):
            for g in self.vars:
                _time_step_history[g].append(state.get_value(g).is_true())
        val = '->'.join(map(lambda e: str(e[0]), sorted([(g, next((i for i, x in enumerate(_time_step_history[g]) if x), -1)) for g in self.vars], key=lambda e:e[1])))
        self.domain.add(val)
        return f'{self.name}:' + val
