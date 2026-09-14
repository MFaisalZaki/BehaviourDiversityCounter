"""The paper's worked examples, against a stub counter.

Every number here is stated in the paper, so a failure means the library and
the paper have parted company -- nothing computed downstream of these is
trustworthy until they agree again.

The counter is stubbed rather than built over a PDDL task: the indicators read
behaviour strings and per-dimension distances, never plans or states, so a task
and a simulator would only add moving parts between the paper's arithmetic and
the assertion. The two dimensions are the paper's own:

    dimension 1 ('nr')  the number of rovers used; distance 0 if the values are
                        equal and 1 otherwise; weight 1/2
    dimension 2 ('co')  the sample collection order, a three-letter string;
                        distance is the Hamming distance over the three
                        positions divided by three, so that it lies in [0, 1]
                        like every per-dimension distance of Def. feature;
                        weight 1/2

The weights are not written into the stub: with none declared the counter's
own convention gives the uniform 1/n, which is the paper's 1/2, 1/2.
"""

import itertools
import math

import pytest

from behaviour_diversity_counter import BehaviourDiversityCounter
from behaviour_diversity_counter.dimensions.base import BehaviourDimension


# ----------------------------------------------------------------------
# The stub behaviour space
# ----------------------------------------------------------------------

def token(behaviour, name):
    """One dimension's value out of a behaviour string, by token prefix."""
    for part in behaviour.split(' $$ '):
        if part.startswith(name + ':'):
            return part[len(name) + 1:]
    raise AssertionError(f"no '{name}' token in {behaviour!r}")


class RoversUsedDimension(BehaviourDimension):
    """Equality on the number of rovers: 0 when equal, 1 when not."""

    def __init__(self):
        super().__init__(task=None, name='nr', addinfo=None)

    def distance(self, b1, b2):
        same = token(b1, self.name) == token(b2, self.name)
        return self.weight * (0.0 if same else 1.0)


class CollectionOrderDimension(BehaviourDimension):
    """Hamming distance over the three collection-order positions, divided by
    three -- the number of samples -- so that it takes the values 0, 2/3 and 1.
    """

    def __init__(self):
        super().__init__(task=None, name='co', addinfo=None)

    def distance(self, b1, b2):
        s1, s2 = token(b1, self.name), token(b2, self.name)
        return self.weight * sum(x != y for x, y in zip(s1, s2)) / len(s1)


class StubPlan:
    """A plan that is nothing but its behaviour."""

    def __init__(self, rovers, order, cost=1):
        self.behaviour = f'nr:{rovers} $$ co:{order}'
        self.cost = cost

    def __repr__(self):
        return f'StubPlan({self.behaviour!r})'


class StubCounter(BehaviourDiversityCounter):
    """A counter over the two stub dimensions, with no task and no simulator.

    Plans are handed their behaviours directly through the behaviour cache, so
    the real ``_behaviours_of`` runs and never reaches ``_simulate``.
    """

    def __init__(self):
        self.task = None
        self.dimensions = {'nr': RoversUsedDimension(), 'co': CollectionOrderDimension()}
        self._apply_weight_convention()      # the real rule: uniform 1/2, 1/2
        self._simulator = None
        self._behaviour_cache = {}
        self._cost_cache = {}
        self._behaviour_distance_cache = {}
        self._plans = []          # keeps the plans alive: the cache is keyed by id()

    def make_plans(self, *specs):
        """Plans for (rovers, order[, cost]) tuples, pre-registered as their
        behaviours and costs."""
        plans = []
        for spec in specs:
            plan = StubPlan(*spec)
            self._behaviour_cache[id(plan)] = plan.behaviour
            self._cost_cache[id(plan)] = plan.cost
            self._plans.append(plan)
            plans.append(plan)
        return plans


@pytest.fixture
def counter():
    return StubCounter()


def naive_b_novelty(counter, plans, k_nn):
    """B-Novelty straight off the definition, for cross-checking the library."""
    distinct = list(dict.fromkeys(plan.behaviour for plan in plans))
    b = len(distinct)
    if b < 2:
        return 0.0
    k_prime = min(k_nn, b - 1)
    total = 0.0
    for i, behaviour in enumerate(distinct):
        others = sorted(counter._pair_distance(behaviour, distinct[j])
                        for j in range(b) if j != i)
        total += sum(others[:k_prime]) / k_prime
    return total / b


# ----------------------------------------------------------------------
# Test 1: the three-plan example, two distinct behaviours
# ----------------------------------------------------------------------

class TestWorkedExampleOne:
    """Behaviours (2, 'RIS'), (1, 'RIS'), (1, 'RIS').

    The two distinct behaviours differ on the rover count alone, so their
    distance is 1/2 * 1 + 1/2 * 0 = 0.5.
    """

    @pytest.fixture
    def plans(self, counter):
        return counter.make_plans((2, 'RIS'), (1, 'RIS'), (1, 'RIS'))

    def test_b_coverage(self, counter, plans):
        assert counter.b_coverage(plans) == 2

    def test_b_maxsum(self, counter, plans):
        assert counter.b_maxsum(plans) == pytest.approx(0.5)

    def test_b_maxmin(self, counter, plans):
        assert counter.b_maxmin(plans) == pytest.approx(0.5)

    def test_b_novelty(self, counter, plans):
        assert counter.b_novelty(plans, k_nn=1) == pytest.approx(0.5)

# ----------------------------------------------------------------------
# Test 2: the three-behaviour example
# ----------------------------------------------------------------------

class TestWorkedExampleTwo:
    """Behaviours (1, 'RSI'), (1, 'RIS'), (2, 'SIR'), pairwise 1/3, 1 and 5/6."""

    @pytest.fixture
    def plans(self, counter):
        return counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

    def test_the_three_pairwise_distances(self, counter, plans):
        rsi, ris, sir = (plan.behaviour for plan in plans)

        assert counter._pair_distance(rsi, ris) == pytest.approx(1 / 3)
        assert counter._pair_distance(rsi, sir) == pytest.approx(1.0)
        assert counter._pair_distance(ris, sir) == pytest.approx(5 / 6)

    def test_b_maxsum(self, counter, plans):
        assert counter.b_maxsum(plans) == pytest.approx(13 / 6)

    def test_b_maxmin(self, counter, plans):
        assert counter.b_maxmin(plans) == pytest.approx(1 / 3)

    def test_b_novelty(self, counter, plans):
        """Nearest-neighbour distances 1/3, 1/3 and 5/6, averaging to 1/2."""
        assert counter.b_novelty(plans, k_nn=1) == pytest.approx(1 / 2)


# ----------------------------------------------------------------------
# The brief's oracle table (EXPERIMENTS_BRIEF.md, section 3)
# ----------------------------------------------------------------------

class TestOracleTable:
    """Every row of the brief's table, with exact fractions."""

    def test_third_set(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'RSI'))
        rsi1, ris1, rsi2 = (plan.behaviour for plan in plans)

        assert counter._pair_distance(rsi1, rsi2) == pytest.approx(1 / 2)
        assert counter._pair_distance(ris1, rsi2) == pytest.approx(5 / 6)
        assert counter.b_coverage(plans) == 3
        assert counter.b_maxsum(plans) == pytest.approx(5 / 3)
        assert counter.b_maxmin(plans) == pytest.approx(1 / 3)
        assert counter.b_novelty(plans, k_nn=1) == pytest.approx(7 / 18)
        assert counter.b_novelty(plans, k_nn=2) == pytest.approx(5 / 9)

    def test_pair(self, counter):
        plans = counter.make_plans((1, 'RSI'), (2, 'SIR'))

        assert counter.b_coverage(plans) == 2
        assert counter.b_maxsum(plans) == pytest.approx(1.0)
        assert counter.b_maxmin(plans) == pytest.approx(1.0)
        assert counter.b_novelty(plans, k_nn=1) == pytest.approx(1.0)

    def test_single_behaviour(self, counter):
        plans = counter.make_plans((1, 'RSI'))

        assert (counter.b_coverage(plans), counter.b_maxsum(plans), counter.b_maxmin(plans),
                counter.b_novelty(plans, k_nn=1)) == (1, 0, 0, 0)

    def test_second_set_with_a_larger_neighbourhood(self, counter):
        """kappa' clamps to 2; per-behaviour means 2/3, 7/12 and 11/12."""
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

        assert counter.b_novelty(plans, k_nn=2) == pytest.approx(13 / 18)
        assert counter.b_novelty(plans, k_nn=3) == pytest.approx(13 / 18)

    def test_first_set_under_every_kappa(self, counter):
        plans = counter.make_plans((2, 'RIS'), (1, 'RIS'), (1, 'RIS'))

        for k_nn in (1, 2, 3, 15):
            assert counter.b_novelty(plans, k_nn=k_nn) == pytest.approx(1 / 2)

    def test_b_maxsum_is_not_submodular(self, counter):
        """The gain of <1,RIS> on {<1,RSI>, <2,SIR>} is 7/6; on the empty set 0."""
        rsi, sir, ris = counter.make_plans((1, 'RSI'), (2, 'SIR'), (1, 'RIS'))

        gain_large = counter.b_maxsum([rsi, sir, ris]) - counter.b_maxsum([rsi, sir])
        gain_small = counter.b_maxsum([ris]) - counter.b_maxsum([])

        assert gain_large == pytest.approx(7 / 6)
        assert gain_small == 0
        assert gain_large > gain_small

    def test_select_coverage_pads_to_k(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'), (1, 'RSI'), (2, 'SIR'), (1, 'RIS'))

        selected = counter.extract(plans, k=5, indicator='bcoverage')

        assert len(selected) == 5
        assert counter.b_coverage(selected) == 3


# ----------------------------------------------------------------------
# Test 3: the weights and the range of the distance
# ----------------------------------------------------------------------

class TestWeightConvention:
    """Def. feature asks for a per-dimension distance in [0, 1] and a positive
    weight; Prop. separable then puts the behaviour distance in [0, 1] when the
    weights sum to one. Uniform 1/n is the counter's default, and the paper's
    example weights."""

    def test_undeclared_weights_are_uniform(self, counter):
        assert [dim.weight for dim in counter.dimensions.values()] == [0.5, 0.5]

    def test_every_pair_distance_lies_in_the_unit_interval(self, counter):
        orders = [''.join(p) for p in itertools.permutations('RSI')]
        plans = counter.make_plans(*[(rovers, order) for rovers in (1, 2) for order in orders])
        behaviours = [plan.behaviour for plan in plans]

        distances = [counter._pair_distance(a, b) for a in behaviours for b in behaviours]

        assert min(distances) == 0.0
        assert max(distances) == pytest.approx(1.0)

    def test_the_distance_is_definite(self, counter):
        """Zero exactly on equal behaviours, as Def. separable-distance assumes."""
        a, b, c = counter.make_plans((1, 'RSI'), (1, 'RSI'), (2, 'RSI'))

        assert counter._pair_distance(a.behaviour, b.behaviour) == 0.0
        assert counter._pair_distance(a.behaviour, c.behaviour) > 0.0


# ----------------------------------------------------------------------
# Test 4: the k_nn clamp
# ----------------------------------------------------------------------

class TestNoveltyClamping:
    """With b <= k_nn every behaviour averages over *all* the others, so
    B-Novelty collapses to B-MaxSum / C(b, 2) -- the mean pairwise distance.

    Recorded as a golden test because it makes B-Novelty duplicate B-MaxSum's
    ranking on small behaviour sets, which reads as a copy-paste error in the
    result tables unless it is known to be the definition.
    """

    @pytest.mark.parametrize('specs', [
        [(1, 'RSI'), (1, 'RIS'), (2, 'SIR')],
        [(1, 'RSI'), (2, 'SIR')],
        [(1, 'RSI'), (1, 'RIS'), (2, 'SIR'), (3, 'IRS')],
    ])
    def test_novelty_is_the_mean_pairwise_distance(self, counter, specs):
        plans = counter.make_plans(*specs)
        b = counter.b_coverage(plans)
        k_nn = b  # b <= k_nn, so k' = min(k_nn, b - 1) = b - 1

        assert counter.b_novelty(plans, k_nn=k_nn) == pytest.approx(
            counter.b_maxsum(plans) / math.comb(b, 2))

    def test_the_clamp_binds_only_while_b_is_small(self, counter):
        """With b > k_nn the two indicators genuinely part company."""
        plans = counter.make_plans(
            (1, 'RSI'), (1, 'RIS'), (2, 'SIR'), (3, 'IRS'), (4, 'ISR'),
        )
        b = counter.b_coverage(plans)

        assert b == 5
        assert counter.b_novelty(plans, k_nn=2) != pytest.approx(
            counter.b_maxsum(plans) / math.comb(b, 2))


# ----------------------------------------------------------------------
# Test 5: duplicate invariance
# ----------------------------------------------------------------------

class TestDuplicateInvariance:
    """Adding a plan whose behaviour is already present changes nothing.

    Every indicator is defined over the *distinct* behaviours, so a duplicate
    plan is invisible to all four. This is what makes `redundancy = k - b_coverage`
    meaningful: the extra plans are not diversity the user can see.
    """

    @pytest.fixture
    def plans(self, counter):
        return counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

    def test_all_four_indicators_are_unchanged(self, counter, plans):
        before = (counter.b_coverage(plans), counter.b_maxsum(plans),
                  counter.b_maxmin(plans), counter.b_novelty(plans))
        duplicated = plans + counter.make_plans((1, 'RIS'))

        after = (counter.b_coverage(duplicated), counter.b_maxsum(duplicated),
                 counter.b_maxmin(duplicated), counter.b_novelty(duplicated))

        assert after == pytest.approx(before)

    def test_many_duplicates_change_nothing_either(self, counter, plans):
        before = counter.b_novelty(plans, k_nn=1)
        duplicated = plans + counter.make_plans(*[(2, 'SIR')] * 20)

        assert counter.b_novelty(duplicated, k_nn=1) == pytest.approx(before)


class TestSelectorsHonourTheirIndicators:
    def test_coverage_selection_covers_every_behaviour_it_can(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

        selected = counter.extract(plans, k=3, indicator='bcoverage')

        assert counter.b_coverage(selected) == 3

    def test_maxsum_selection_takes_the_farthest_pair_first(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

        selected = counter.extract(plans, k=2, indicator='bmaxsum')

        assert counter.b_maxsum(selected) == pytest.approx(1.0)

    def test_coverage_selection_keeps_the_cheapest_plan_per_behaviour(self, counter):
        """Which plan represents a behaviour is left open by the paper;
        the paper takes the cheapest one in the pool, as MAP-Elites keeps the
        fittest solution per cell. Pool order is not cost order here."""
        dear, cheap, other = counter.make_plans((1, 'RSI', 9), (1, 'RSI', 2), (2, 'SIR', 5))

        selected = counter.extract([dear, cheap, other], k=2, indicator='bcoverage')

        assert selected == [cheap, other]

    def test_coverage_selection_breaks_cost_ties_towards_the_earliest_plan(self, counter):
        first, second = counter.make_plans((1, 'RSI', 3), (1, 'RSI', 3))

        assert counter.extract([first, second], k=1, indicator='bcoverage') == [first]

    def test_novelty_selection_takes_a_fresh_behaviour_even_when_it_lowers_the_value(
        self, counter
    ):
        """B-Novelty is not monotone, so a new behaviour can pull the value
        below what a duplicate would have preserved. The greedy still ranks
        only the fresh candidates, as the paper's rules all do, and the fall
        shows in the indicator of the returned set rather than in a duplicate
        handed to the user in place of an option."""
        a, b, a_again, c = counter.make_plans((1, 'RSI'), (2, 'SIR'), (1, 'RSI'), (2, 'RSI'))
        assert counter.b_novelty([a, b], k_nn=1) == pytest.approx(1.0)
        assert counter.b_novelty([a, b, c], k_nn=1) < 1.0

        selected = counter.extract([a, b, a_again, c], k=3, indicator='bnovelty', k_nn=1)

        assert counter.b_coverage(selected) == 3
        assert counter.b_novelty(selected, k_nn=1) < 1.0

    def test_ties_break_towards_the_lowest_plan_index(self, counter):
        """Two plans exhibit the same behaviour; the earlier one must be taken."""
        plans = counter.make_plans((1, 'RSI'), (2, 'SIR'), (2, 'SIR'))

        for indicator in ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty'):
            selected = counter.extract(plans, k=2, indicator=indicator)
            assert selected[-1] is plans[1], indicator

    def test_an_unknown_indicator_is_still_rejected(self, counter):
        plans = counter.make_plans((1, 'RSI'))

        with pytest.raises(ValueError, match='valid indicators'):
            counter.extract(plans, k=1, indicator='nope')


class TestSingleBehaviourPools:
    """A pool exhibiting one behaviour scores 0 under both non-monotone
    indicators whatever is selected, so the selection keeps the k plans asked
    for rather than truncating to one."""

    @pytest.fixture
    def plans(self, counter):
        return counter.make_plans(*[(1, 'RIS')] * 5)

    @pytest.mark.parametrize('indicator', ['bmaxmin', 'bnovelty'])
    @pytest.mark.parametrize('k', [1, 3, 5, 7])
    def test_k_plans_come_back(self, counter, plans, indicator, k):
        selected = counter.extract(plans, k=k, indicator=indicator)

        assert len(selected) == min(k, len(plans))


# ----------------------------------------------------------------------
# Test 7: determinism
# ----------------------------------------------------------------------

DETERMINISM_SCRIPT = r'''
import json, random, sys
sys.path.insert(0, {tests!r})
sys.path.insert(0, {root!r})
from test_golden import StubCounter

rng = random.Random(int(sys.argv[1]))
specs = [(rng.randint(1, 4), ''.join(rng.sample('RIS', 3))) for _ in range(40)]
counter = StubCounter()
plans = counter.make_plans(*specs)
position = {{id(plan): index for index, plan in enumerate(plans)}}

out = {{}}
for indicator in ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty'):
    selected = counter.extract(plans, k=8, indicator=indicator)
    out[indicator] = [position[id(plan)] for plan in selected]
out['scores'] = [round(counter.b_maxsum(plans), 12),
                 round(counter.b_maxmin(plans), 12),
                 round(counter.b_novelty(plans), 12)]
print(json.dumps(out, sort_keys=True))
'''


class TestDeterminism:
    """The same seed and the same inputs must give byte-identical selections,
    for all four selectors.

    Run in subprocesses under different PYTHONHASHSEED values, because that is
    what a hidden dependence on set or dict iteration order actually looks
    like: string hashing is randomised per process, so a selector that ranked
    candidates by walking a set would agree with itself all day inside one
    interpreter and disagree across the runs of a sweep.
    """

    @staticmethod
    def _run(tmp_path, seed, hash_seed):
        import json
        import os
        import subprocess
        import sys

        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        script = tmp_path / f'determinism_{hash_seed}.py'
        script.write_text(DETERMINISM_SCRIPT.format(tests=here, root=root))
        environment = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
        result = subprocess.run([sys.executable, str(script), str(seed)],
                                capture_output=True, text=True, env=environment)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def test_two_runs_agree_across_hash_seeds(self, tmp_path):
        first = self._run(tmp_path, seed=2026, hash_seed=0)
        second = self._run(tmp_path, seed=2026, hash_seed=987654321)

        assert first == second
        for selector in ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty'):
            assert first[selector] == second[selector], selector

    def test_a_third_hash_seed_agrees_too(self, tmp_path):
        first = self._run(tmp_path, seed=2026, hash_seed=1)
        second = self._run(tmp_path, seed=2026, hash_seed=42)

        assert first == second

    def test_a_different_seed_gives_a_different_pool(self, tmp_path):
        """A guard on the guard: if the seed were ignored, the test above would
        pass while proving nothing."""
        first = self._run(tmp_path, seed=2026, hash_seed=0)
        other = self._run(tmp_path, seed=7, hash_seed=0)

        assert first != other
