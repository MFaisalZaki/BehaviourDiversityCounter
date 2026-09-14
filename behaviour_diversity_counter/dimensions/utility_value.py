import re
from collections import defaultdict
from fractions import Fraction

from behaviour_diversity_counter.dimensions.base import BehaviourDimension, declared_weight

#: ``<goal>=<value>`` items of the token payload. Goal expressions may hold
#: commas ('at(tr1, l0)'), so the split anchors on the numeric value instead.
_ASSIGNMENT = re.compile(r'(.*?)=(-?[\d./]+)(?:,|$)')


class UtilityValueDimension(BehaviourDimension):
    """``uv``: which utility goals the plan achieves, and their total."""

    def __init__(self, task, addinfo):
        super().__init__(task, 'utility_value', addinfo, declared_weight(addinfo))

    def extract(self, plan):
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

    def _achieved(self, behaviour):
        _, _, assignments = self.payload(behaviour).partition(' -- ')
        return {goal: Fraction(value) for goal, value in _ASSIGNMENT.findall(assignments)}

    def dissimilarity(self, b1, b2):
        # Weighted Jaccard distance over the achieved utilities: the utility of
        # the goals both plans achieve against that of the goals either does,
        # 1 - sum_g min(u_g, u'_g) / sum_g max(u_g, u'_g). A metric in [0, 1],
        # zero exactly when the two plans achieve the same goals.
        utility1, utility2 = self._achieved(b1), self._achieved(b2)
        goals = utility1.keys() | utility2.keys()
        total = sum(max(utility1.get(g, 0), utility2.get(g, 0)) for g in goals)
        if total == 0:
            return 0.0
        shared = sum(min(utility1.get(g, 0), utility2.get(g, 0)) for g in goals)
        return self.weight * float(1 - shared / total)
