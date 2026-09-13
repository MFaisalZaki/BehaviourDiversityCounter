"""A stub behaviour space: the library's indicators without a PDDL task.

Built the way ``tests/test_golden.py`` builds its rover stub -- a counter
subclass that sets ``task``, ``dimensions``, ``_simulator`` and the three
caches by hand, over plans that are nothing but a behaviour string and a cost
-- but with the dimensions drawn at random, so the audit can sample behaviour
spaces rather than assert one worked example at a time.

A behaviour is encoded the way the library encodes it, the per-dimension
tokens ``'<name>:<value>'`` joined by ``' $$ '``; the reference-side behaviour
is the tuple of those per-dimension values.
"""

import math
import random

import pytest

from behaviour_diversity_counter import BehaviourDiversityCounter
from behaviour_diversity_counter.dimensions.base import BehaviourDimension


def encode(dimensions, values):
    """The library's behaviour string for a tuple of per-dimension values."""
    return ' $$ '.join(f'{dim.name}:{value}' for dim, value in zip(dimensions, values))


class StubDimension(BehaviourDimension):
    """One feature of Def. feature: a finite value set, a table of
    per-dimension dissimilarities in [0, 1] with a zero diagonal, and a weight.
    """

    def __init__(self, name, values, table, weight=None):
        super().__init__(task=None, name=name, addinfo=None, weight=weight)
        self.values = list(values)
        self.table = table

    def distance(self, b1, b2):
        return self.weight * self.table[(self.payload(b1), self.payload(b2))]


class StubPlan:
    """A plan that is nothing but its behaviour, its values and its cost."""

    def __init__(self, behaviour, values, cost):
        self.behaviour = behaviour
        self.values = values
        self.cost = cost

    def __repr__(self):
        return f'StubPlan({self.behaviour!r}, cost={self.cost})'


class StubCounter(BehaviourDiversityCounter):
    """A counter over stub dimensions, with no task and no simulator.

    Plans are handed their behaviours and costs through the caches, so the real
    ``_behaviours_of`` runs and never reaches ``_simulate``.
    """

    def __init__(self, dimensions):
        self.task = None
        self.dimensions = {dim.name: dim for dim in dimensions}
        self._apply_weight_convention()
        self._simulator = None
        self._behaviour_cache = {}
        self._cost_cache = {}
        self._behaviour_distance_cache = {}
        self._plans = []          # the caches are keyed by id(): keep them alive

    def make_plans(self, specs):
        """Plans for ``(values, cost)`` pairs, pre-registered as behaviours."""
        dims = list(self.dimensions.values())
        plans = []
        for values, cost in specs:
            plan = StubPlan(encode(dims, values), tuple(values), cost)
            self._behaviour_cache[id(plan)] = plan.behaviour
            self._cost_cache[id(plan)] = cost
            self._plans.append(plan)
            plans.append(plan)
        return plans


def metric_table(values, rng):
    """Family (a): a metric, definite dissimilarity -- either the discrete
    distance or ``|i - j| / (m - 1)`` over the ordered value set."""
    m = len(values)
    ordered = rng.random() < 0.5
    return {(x, y): (abs(i - j) / (m - 1) if ordered else float(i != j))
            for i, x in enumerate(values) for j, y in enumerate(values)}


def nonmetric_table(values, rng):
    """Family (b): a definite but generally non-metric dissimilarity -- a
    random symmetric matrix, zero on the diagonal, entries drawn in (0, 1].

    Half the tables are quantised to a tenth, which makes exact ties between
    candidate scores common rather than measure-zero.
    """
    coarse = rng.random() < 0.5
    table = {}
    for i, x in enumerate(values):
        table[(x, x)] = 0.0
        for y in values[i + 1:]:
            v = rng.randint(1, 10) / 10 if coarse else rng.uniform(0.01, 1.0)
            table[(x, y)] = table[(y, x)] = v
    return table


class Space:
    """One random behaviour space, seeded from a case number.

    2 to 4 dimensions of 2 to 6 values each, one of the two dissimilarity
    families throughout, and either the counter's uniform ``1/n`` weights or
    explicitly declared unequal ones summing to 1.
    """

    def __init__(self, seed):
        self.seed = seed
        rng = random.Random(seed)
        n = rng.randint(2, 4)
        self.metric = rng.random() < 0.5
        self.uniform_weights = rng.random() < 0.5
        weights = self._draw_weights(rng, n)
        table_of = metric_table if self.metric else nonmetric_table
        self.dimensions = []
        for i in range(n):
            values = [f'v{j}' for j in range(rng.randint(2, 6))]
            self.dimensions.append(
                StubDimension(f'd{i}', values, table_of(values, rng), weights[i]))
        StubCounter(self.dimensions)      # resolves the weight convention
        self.weights = [dim.weight for dim in self.dimensions]

    @staticmethod
    def _draw_weights(rng, n):
        if rng.random() < 0.5:
            return [None] * n             # undeclared: the counter's uniform 1/n
        raw = [rng.uniform(0.5, 2.0) for _ in range(n)]
        total = math.fsum(raw)
        weights = [x / total for x in raw]
        weights[-1] = 1.0 - math.fsum(weights[:-1])   # sums to 1 exactly
        return weights

    def __repr__(self):
        return (f'Space(seed={self.seed}, dims={[len(d.values) for d in self.dimensions]}, '
                f'{"metric" if self.metric else "non-metric"}, '
                f'weights={[round(w, 4) for w in self.weights]})')

    def counter(self):
        """A fresh counter over these dimensions, with empty caches."""
        return StubCounter(self.dimensions)

    def d(self, x, y):
        """psi_M on reference behaviour tuples: ``sum_i w_i psi_i(x_i, y_i)``."""
        return math.fsum(dim.weight * dim.table[(a, b)]
                         for dim, a, b in zip(self.dimensions, x, y))

    def draw(self, rng, size, distinct=None):
        """A pool of ``size`` ``(values, cost)`` pairs drawn from a menu of
        ``distinct`` behaviours, so duplicates arise when the menu is short."""
        menu = [tuple(rng.choice(dim.values) for dim in self.dimensions)
                for _ in range(distinct if distinct is not None else size)]
        return [(rng.choice(menu), rng.randint(1, 9)) for _ in range(size)]


@pytest.fixture
def space():
    """A factory for seeded random behaviour spaces."""
    return Space


@pytest.fixture
def stub():
    """The stub pieces themselves, for tests that build a space by hand."""
    return {'Space': Space, 'StubCounter': StubCounter,
            'StubDimension': StubDimension, 'StubPlan': StubPlan, 'encode': encode}
