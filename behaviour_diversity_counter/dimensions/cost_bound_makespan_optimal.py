from fractions import Fraction

from behaviour_diversity_counter.dimensions.base import BehaviourDimension, declared_weight
from behaviour_diversity_counter.simulation import plan_cost


class MakespanOptimalCostDimension(BehaviourDimension):
    """``cb``: the plan's cost, in the paper's sense of Def. plan -- the sum of
    its action costs. Under a task without a cost metric every action costs
    one, and the value is the plan length.
    """

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'cb', addinfo, declared_weight(addinfo))

    def extract(self, plan):
        # The counter attaches the cost it accumulated while simulating; a
        # dimension used on its own replays the plan to get it.
        cost = getattr(plan, 'cost', None)
        if cost is None:
            cost = plan_cost(self.task, plan)
        self.domain.add(cost)
        return f'{self.name}:{cost}'

    def _cost(self, behaviour):
        return Fraction(self.payload(behaviour))

    def dissimilarity(self, b1, b2):
        cost1, cost2 = self._cost(b1), self._cost(b2)
        if max(cost1, cost2) == 0:
            return 0.0
        # 1 - min/max: a metric on the non-negative costs, in [0, 1], and zero
        # exactly on equal costs, as Def. feature and Def. similarity-space ask.
        return self.weight * float(abs(cost1 - cost2) / max(cost1, cost2))
