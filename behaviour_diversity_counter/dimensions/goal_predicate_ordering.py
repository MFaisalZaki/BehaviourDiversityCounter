from collections import defaultdict
from behaviour_diversity_counter.dimensions.base import BehaviourDimension

class GoalPredicatesOrderingDimension(BehaviourDimension):
    def __init__(self, task, addinfo=None):
        super().__init__(task, 'go', addinfo, addinfo.get('weight', 1.0))
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

    def distance(self, plan1, plan2):

        # Match on the token prefix, not a substring: predicate/object names in
        # other tokens may contain this dimension's name (e.g. 'truck' vs 'ru').
        plan1_dim_value = next(filter(lambda e: e.strip().startswith(self.name + ':'), plan1.split('$$')), None)
        plan2_dim_value = next(filter(lambda e: e.strip().startswith(self.name + ':'), plan2.split('$$')), None)

        assert plan1_dim_value is not None and plan2_dim_value is not None, 'The dimension value should be present in the plan behaviour.'

        plan1_dim_value = plan1_dim_value.strip().replace(self.name + ':', '').replace(' ', '').split('->')
        plan2_dim_value = plan2_dim_value.strip().replace(self.name + ':', '').replace(' ', '').split('->')

        hamming_distance = [x == y for x,y in zip(plan1_dim_value, plan2_dim_value)].count(False)
        distance = hamming_distance/len(plan1_dim_value) if len(plan1_dim_value) > 0 else 0.0
        return self.weight * distance
