from itertools import chain, combinations
from collections import defaultdict
from behaviour_diversity_counter.features.base import DimensionConstructorSimulator

class UtilityValueSimulator(DimensionConstructorSimulator):
    def __init__(self, task, addinfo):
        super().__init__(task, 'utility_value', addinfo)
        from unified_planning.model.walkers.free_vars import FreeVarsExtractor
        vars = list(map(lambda expr: FreeVarsExtractor().get(expr), self.task.goals))
        self.vars = [(str(elem), elem) for s in vars for elem in s]
    
    def _estimate_domain(self):
        estimated_domain = set()
        possible_acheived_goals = set(chain.from_iterable(combinations(self.addinfo['utility-goals'], r) for r in range(1, len(self.addinfo['utility-goals']) + 1)))
        for achieved_goals in possible_acheived_goals:
            val = str(sum([self.addinfo['utility-goals'][g] for g in achieved_goals])) + ' -- ' + ','.join(f'{k}={str(v)}' for k,v in self.addinfo['utility-goals'].items())
            estimated_domain.add(val)
        self.estimated_domain_size = len(estimated_domain)

    def plan_behaviour(self, plan):
        achieved_utilities = defaultdict(list)
        _acheived_utilities = defaultdict(list)
        for state in plan.states:
            for var, util in self.addinfo['utility-goals'].items():
                _acheived_utilities[var].append(state.get_value(var).is_true())

        for var, utils in _acheived_utilities.items():
            achieved_utilities[str(var)] = self.addinfo['utility-goals'][var] if any(utils) else 0
        val = str(sum(achieved_utilities.values())) + ' -- ' + ','.join(f'{k}={str(v)}' for k,v in achieved_utilities.items())
        self.domain.add(val)
        return f'{self.name}:' + val
