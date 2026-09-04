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
                        distance is raw Hamming over the three positions;
                        weight 1/2
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
    """Raw Hamming distance over the three collection-order positions.

    Raw, not normalised into [0, 1]: the paper's worked examples add a Hamming
    count of 2 or 3 straight into the weighted sum, so dividing by the string
    length here would quietly move every golden number.
    """

    def __init__(self):
        super().__init__(task=None, name='co', addinfo=None)

    def distance(self, b1, b2):
        s1, s2 = token(b1, self.name), token(b2, self.name)
        return self.weight * float(sum(x != y for x, y in zip(s1, s2)))


class StubPlan:
    """A plan that is nothing but its behaviour."""

    def __init__(self, rovers, order, actions=None):
        self.behaviour = f'nr:{rovers} $$ co:{order}'
        # The plan-level baseline reads actions, not behaviours. Derived from
        # the behaviour so that distinct behaviours differ at the plan level
        # too, and repeated so the multiset reading has something to see.
        self.actions = actions if actions is not None else (
            [f'collect_{letter}' for letter in order] +
            [f'drive_rover{i}' for i in range(1, rovers + 1)] * 2)

    def __repr__(self):
        return f'StubPlan({self.behaviour!r})'


class StubCounter(BehaviourDiversityCounter):
    """A counter over the two stub dimensions, with no task and no simulator.

    Plans are handed their behaviours directly through the behaviour cache, so
    the real ``_behaviours_of`` runs and never reaches ``_simulate``.
    """

    def __init__(self, weights=None):
        self.task = None
        self.dimensions = {'nr': RoversUsedDimension(), 'co': CollectionOrderDimension()}
        self._simulator = None
        self._behaviour_cache = {}
        self._distance_cache = {}
        self._plans = []          # keeps the plans alive: the cache is keyed by id()
        self.set_weights(weights)

    def make_plans(self, *specs):
        """Plans for (rovers, order) pairs, pre-registered as their behaviours."""
        plans = []
        for rovers, order in specs:
            plan = StubPlan(rovers, order)
            self._behaviour_cache[id(plan)] = plan.behaviour
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
    """Behaviours (1, 'RSI'), (1, 'RIS'), (2, 'SIR'), pairwise 1, 2 and 1.5."""

    @pytest.fixture
    def plans(self, counter):
        return counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

    def test_the_three_pairwise_distances(self, counter, plans):
        rsi, ris, sir = (plan.behaviour for plan in plans)

        assert counter._pair_distance(rsi, ris) == pytest.approx(1.0)
        assert counter._pair_distance(rsi, sir) == pytest.approx(2.0)
        assert counter._pair_distance(ris, sir) == pytest.approx(1.5)

    def test_b_maxsum(self, counter, plans):
        assert counter.b_maxsum(plans) == pytest.approx(4.5)

    def test_b_maxmin(self, counter, plans):
        assert counter.b_maxmin(plans) == pytest.approx(1.0)

    def test_b_novelty(self, counter, plans):
        assert counter.b_novelty(plans, k_nn=1) == pytest.approx(3.5 / 3)


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

        assert counter.b_maxsum(selected) == pytest.approx(2.0)

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


# ----------------------------------------------------------------------
# Test 6: weights, and the cache they invalidate
# ----------------------------------------------------------------------

class TestWeights:
    """The paper's separable distance is d(b, b') = sum_i w_i * d_i(b_i, b'_i),
    and the weights are the one thing it asks a user to supply.

    The distance cache is keyed by the behaviour pair alone, so every one of
    these assertions is also an assertion that changing the weights cleared it.
    Experiment C changes the weights twenty-one times per task on one counter
    object: a cache that survived the change would return twenty-one identical
    rows and raise no error at all.
    """

    @pytest.fixture
    def behaviours(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))
        return [plan.behaviour for plan in plans]

    def test_weighting_out_the_order_dimension(self, counter, behaviours):
        """w_nr = 1, w_co = 0: only the rover count is left to separate them."""
        rsi, ris, sir = behaviours

        counter.set_weights({'nr': 1.0, 'co': 0.0})

        assert counter._pair_distance(rsi, ris) == pytest.approx(0.0)
        assert counter._pair_distance(rsi, sir) == pytest.approx(1.0)

    def test_the_same_counter_answers_for_both_weightings(self, counter, behaviours):
        """The heart of the test: twice on ONE counter, weights changed between.

        The first pass warms the cache with the uniform answers; if setting the
        weights did not clear it, the second pass would return them again.
        """
        rsi, ris, sir = behaviours

        # Pass one, the uniform 1/2 default.
        assert counter._pair_distance(rsi, ris) == pytest.approx(1.0)
        assert counter._pair_distance(rsi, sir) == pytest.approx(2.0)

        # Pass two, on the same object.
        counter.set_weights({'nr': 1.0, 'co': 0.0})
        assert counter._pair_distance(rsi, ris) == pytest.approx(0.0)
        assert counter._pair_distance(rsi, sir) == pytest.approx(1.0)

        # And back, so the failure cannot be a one-way cache that merely lags.
        counter.set_weights({'nr': 0.5, 'co': 0.5})
        assert counter._pair_distance(rsi, ris) == pytest.approx(1.0)
        assert counter._pair_distance(rsi, sir) == pytest.approx(2.0)

    def test_the_indicators_follow_the_weights_on_one_counter(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

        uniform = counter.b_maxsum(plans)
        counter.set_weights({'nr': 1.0, 'co': 0.0})
        rover_only = counter.b_maxsum(plans)
        counter.set_weights({'nr': 0.0, 'co': 1.0})
        order_only = counter.b_maxsum(plans)

        assert uniform == pytest.approx(4.5)
        assert rover_only == pytest.approx(2.0)   # only (2,SIR) differs in count
        assert order_only == pytest.approx(7.0)   # Hamming 2 + 3 + 2
        assert uniform == pytest.approx((rover_only + order_only) / 2)

    def test_a_weight_sweep_on_one_counter_does_not_repeat_itself(self, counter):
        """Experiment C's sweep in miniature: 21 weightings, one counter."""
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'), (3, 'IRS'))

        values = []
        for step in range(21):
            w = step / 20
            counter.set_weights({'nr': w, 'co': 1.0 - w})
            values.append(counter.b_maxsum(plans))

        assert len(set(values)) == 21
        assert values == sorted(values, reverse=True)  # order separates them more

    def test_uniform_weights_reproduce_the_unweighted_mean(self, counter):
        """The default has to leave the pre-weights behaviour exactly as it was."""
        plans = counter.make_plans((1, 'RSI'), (2, 'SIR'))
        rsi, sir = (plan.behaviour for plan in plans)
        unweighted = [RoversUsedDimension(), CollectionOrderDimension()]

        assert counter._pair_distance(rsi, sir) == pytest.approx(
            sum(d.distance(rsi, sir) for d in unweighted) / len(unweighted))

    def test_an_unknown_dimension_in_the_weights_is_rejected(self, counter):
        with pytest.raises(ValueError, match='unknown dimension'):
            counter.set_weights({'nr': 1.0, 'co': 0.0, 'nope': 1.0})

    def test_a_missing_weight_is_rejected(self, counter):
        """Silently defaulting the missing one would be a weighting the user
        never asked for."""
        with pytest.raises(ValueError, match='no weight given'):
            counter.set_weights({'nr': 1.0})

    def test_a_negative_weight_is_rejected(self, counter):
        with pytest.raises(ValueError, match='negative'):
            counter.set_weights({'nr': -1.0, 'co': 2.0})

    def test_weights_are_not_normalised(self, counter):
        """The paper's worked examples read raw per-dimension distances, so the
        weights are taken as given rather than rescaled to sum to one."""
        plans = counter.make_plans((1, 'RSI'), (2, 'SIR'))
        rsi, sir = (plan.behaviour for plan in plans)

        counter.set_weights({'nr': 2.0, 'co': 2.0})

        assert counter._pair_distance(rsi, sir) == pytest.approx(2.0 * 1 + 2.0 * 3)


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
sys.path.insert(0, {paperexps!r})
from test_golden import StubCounter
from paperexps.baseline import Stability, greedy_maxsum_stability

rng = random.Random(int(sys.argv[1]))
specs = [(rng.randint(1, 4), ''.join(rng.sample('RIS', 3))) for _ in range(40)]
counter = StubCounter()
plans = counter.make_plans(*specs)
position = {{id(plan): index for index, plan in enumerate(plans)}}

out = {{}}
for indicator in ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty'):
    selected = counter.extract(plans, k=8, indicator=indicator)
    out[indicator] = [position[id(plan)] for plan in selected]
out['maxsum_stability'] = [position[id(plan)]
                           for plan in greedy_maxsum_stability(plans, 8)]
out['scores'] = [round(counter.b_maxsum(plans), 12),
                 round(counter.b_maxmin(plans), 12),
                 round(counter.b_novelty(plans), 12),
                 round(Stability(plans).maxsum(plans), 12)]
print(json.dumps(out, sort_keys=True))
'''


class TestDeterminism:
    """The same seed and the same inputs must give byte-identical selections,
    for all five selectors.

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
        script.write_text(DETERMINISM_SCRIPT.format(
            tests=here, root=root,
            paperexps=os.path.join(root, 'paper-experiments')))
        environment = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
        result = subprocess.run([sys.executable, str(script), str(seed)],
                                capture_output=True, text=True, env=environment)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def test_two_runs_agree_across_hash_seeds(self, tmp_path):
        first = self._run(tmp_path, seed=2026, hash_seed=0)
        second = self._run(tmp_path, seed=2026, hash_seed=987654321)

        assert first == second
        for selector in ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty',
                         'maxsum_stability'):
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


# ----------------------------------------------------------------------
# Test 8: multiset stability
# ----------------------------------------------------------------------

class TestMultisetStability:
    """p = [a, a, b] against q = [a, b].

    Under the multiset reading the intersection is 2 and the union 3, so
    stability(p, q) = 1/3. Under the set reading both are {a, b} and it is 0.
    The two readings *must* differ here, or the multiset implementation is
    quietly the set one and the baseline has been handed a weaker notion of
    difference than Katz and Sohrabi defined.
    """

    @pytest.fixture
    def pair(self):
        return (StubPlan(1, 'RIS', actions=['a', 'a', 'b']),
                StubPlan(1, 'RIS', actions=['a', 'b']))

    def test_the_multiset_reading(self, pair):
        from paperexps.baseline import Stability

        assert Stability(multiset=True).distance(*pair) == pytest.approx(1 / 3)

    def test_the_set_reading(self, pair):
        from paperexps.baseline import Stability

        assert Stability(multiset=False).distance(*pair) == pytest.approx(0.0)

    def test_the_two_readings_actually_differ(self, pair):
        from paperexps.baseline import Stability

        assert Stability(multiset=True).distance(*pair) != \
               Stability(multiset=False).distance(*pair)

    def test_the_multiset_reading_is_the_default(self, pair):
        from paperexps.baseline import Stability

        assert Stability().distance(*pair) == pytest.approx(1 / 3)

    def test_order_never_matters_under_either_reading(self):
        from paperexps.baseline import Stability

        p = StubPlan(1, 'RIS', actions=['a', 'b', 'a'])
        q = StubPlan(1, 'RIS', actions=['a', 'a', 'b'])

        for multiset in (True, False):
            assert Stability(multiset=multiset).distance(p, q) == pytest.approx(0.0)

    def test_disjoint_plans_score_one(self):
        from paperexps.baseline import Stability

        p = StubPlan(1, 'RIS', actions=['a', 'a'])
        q = StubPlan(1, 'RIS', actions=['b'])

        assert Stability().distance(p, q) == pytest.approx(1.0)

    def test_two_empty_plans_are_identical(self):
        from paperexps.baseline import Stability

        p, q = StubPlan(1, 'RIS', actions=[]), StubPlan(1, 'RIS', actions=[])

        assert Stability().distance(p, q) == pytest.approx(0.0)

    def test_a_repeated_action_is_what_separates_the_readings(self):
        """Four drives against one: identical as sets, far apart as multisets."""
        from paperexps.baseline import Stability

        many = StubPlan(1, 'RIS', actions=['drive'] * 4 + ['sample'])
        once = StubPlan(1, 'RIS', actions=['drive', 'sample'])

        assert Stability(multiset=False).distance(many, once) == pytest.approx(0.0)
        assert Stability(multiset=True).distance(many, once) == pytest.approx(1 - 2 / 5)


class TestGreedyMaxSumStability:
    def test_it_opens_on_the_farthest_pair(self):
        from paperexps.baseline import greedy_maxsum_stability

        near_a = StubPlan(1, 'RIS', actions=['a', 'b'])
        near_b = StubPlan(1, 'RIS', actions=['a', 'b', 'c'])
        far = StubPlan(1, 'RIS', actions=['x', 'y', 'z'])

        selected = greedy_maxsum_stability([near_a, near_b, far], k=2)

        assert set(map(id, selected)) == {id(near_a), id(far)}

    def test_it_returns_exactly_k_plans(self):
        from paperexps.baseline import greedy_maxsum_stability

        plans = [StubPlan(1, 'RIS', actions=[f'a{i}', 'shared']) for i in range(10)]

        assert len(greedy_maxsum_stability(plans, k=4)) == 4

    def test_k_beyond_the_pool_returns_the_pool(self):
        from paperexps.baseline import greedy_maxsum_stability

        plans = [StubPlan(1, 'RIS', actions=[f'a{i}']) for i in range(3)]

        assert len(greedy_maxsum_stability(plans, k=99)) == 3

    def test_ties_fall_to_the_lowest_pool_index(self):
        from paperexps.baseline import greedy_maxsum_stability

        plans = [StubPlan(1, 'RIS', actions=[letter]) for letter in 'abcd']

        selected = greedy_maxsum_stability(plans, k=3)

        assert [plan.actions[0] for plan in selected] == ['a', 'b', 'c']
