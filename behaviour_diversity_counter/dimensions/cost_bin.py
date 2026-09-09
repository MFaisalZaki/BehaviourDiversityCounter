from fractions import Fraction

from behaviour_diversity_counter.dimensions.base import BehaviourDimension, declared_weight, options
from behaviour_diversity_counter.simulation import plan_cost


class CostBinDimension(BehaviourDimension):
    """``cbin``: the bin of ``cost / c*`` in a grid of width ``w`` over ``[1, q]``.

    ``addinfo`` carries ``optimal-cost`` (``c*``), ``q`` (the quality bound,
    default 1.0) and ``width`` (default 0.1), so ``q = 2.0`` at the default
    width gives ten bins. Ratios below 1 land in the first bin and ratios at
    or above ``q`` in the last. With ``q = 1.0`` there is one bin and the
    feature is constant, as the brief allows. The distance is
    ``|bin - bin'| / (bins - 1)``, which respects the bin order.
    """

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'cbin', dict(options(addinfo)), declared_weight(addinfo))
        self.optimal_cost = Fraction(self.addinfo.get('optimal-cost', 1))
        self.q = Fraction(str(self.addinfo.get('q', 1.0)))
        self.width = Fraction(str(self.addinfo.get('width', 0.1)))
        if self.q < 1 or self.width <= 0:
            raise ValueError('cbin needs q >= 1 and a positive bin width')
        self.bins = max(1, int(round((self.q - 1) / self.width)))

    def bin_of(self, cost):
        if self.optimal_cost <= 0:
            return 0
        ratio = Fraction(cost) / self.optimal_cost
        index = int((ratio - 1) // self.width)
        return min(max(index, 0), self.bins - 1)

    def plan_behaviour(self, plan):
        cost = getattr(plan, 'cost', None)
        if cost is None:
            cost = plan_cost(self.task, plan)
        index = self.bin_of(cost)
        self.domain.add(index)
        return f'{self.name}:{index}'

    def distance(self, b1, b2):
        if self.bins < 2:
            return 0.0
        return self.weight * abs(int(self.payload(b1)) - int(self.payload(b2))) / (self.bins - 1)
