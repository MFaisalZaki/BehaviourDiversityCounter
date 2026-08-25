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


# ----------------------------------------------------------------------
# The stub behaviour space
# ----------------------------------------------------------------------

def token(behaviour, name):
    """One dimension's value out of a behaviour string, by token prefix."""
    for part in behaviour.split(' $$ '):
        if part.startswith(name + ':'):
            return part[len(name) + 1:]
    raise AssertionError(f"no '{name}' token in {behaviour!r}")


class RoversUsedDimension:
    """Equality on the number of rovers: 0 when equal, 1 when not."""

    name = 'nr'

    def distance(self, b1, b2):
        return 0.0 if token(b1, self.name) == token(b2, self.name) else 1.0


class CollectionOrderDimension:
    """Raw Hamming distance over the three collection-order positions.

    Raw, not normalised into [0, 1]: the paper's worked examples add a Hamming
    count of 2 or 3 straight into the weighted sum, so dividing by the string
    length here would quietly move every golden number.
    """

    name = 'co'

    def distance(self, b1, b2):
        s1, s2 = token(b1, self.name), token(b2, self.name)
        return float(sum(x != y for x, y in zip(s1, s2)))


class StubPlan:
    """A plan that is nothing but its behaviour."""

    def __init__(self, rovers, order):
        self.behaviour = f'nr:{rovers} $$ co:{order}'

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
        self._weights = {}
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

    def test_bdc_is_an_alias_of_b_coverage(self, counter, plans):
        assert counter.bdc(plans) == counter.b_coverage(plans) == 2


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
# Test 3: thm:bmaxmin-degenerate
# ----------------------------------------------------------------------

class TestBMaxMinDegeneracy:
    """Maximising B-MaxMin returns exactly two plans, whatever k is asked for.

    A third behaviour can only lower the minimum pairwise distance, so the
    best-scoring prefix of the farthest-first order is always the seed pair.
    The experiment code has to survive this rather than crash on it, so the
    trace is checked to still run the full k steps.
    """

    @pytest.fixture
    def plans(self, counter):
        return counter.make_plans(
            (1, 'RSI'), (1, 'RIS'), (2, 'SIR'), (2, 'RIS'), (3, 'SRI'), (1, 'IRS'),
        )

    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6])
    def test_selection_is_two_plans_whatever_k(self, counter, plans, k):
        selected = counter.extract(plans, k=k, indicator='bmaxmin')

        assert len(selected) == 2

    @pytest.mark.parametrize('k', [2, 3, 4, 5, 6])
    def test_the_trace_still_runs_the_full_k_steps(self, counter, plans, k):
        selection = counter.extract(plans, k=k, indicator='bmaxmin', trace=True)

        assert len(selection.order) == k
        assert len(selection.scores) == k
        assert selection.best_step == 2

    def test_the_pair_returned_is_the_farthest_pair(self, counter, plans):
        selected = counter.extract(plans, k=4, indicator='bmaxmin')
        best = max(counter._pair_distance(a.behaviour, b.behaviour)
                   for a, b in itertools.combinations(plans, 2))

        assert counter.b_maxmin(selected) == pytest.approx(best)

    def test_the_score_never_rises_after_the_pair(self, counter, plans):
        selection = counter.extract(plans, k=6, indicator='bmaxmin', trace=True)
        after_pair = selection.scores[1:]

        assert after_pair == sorted(after_pair, reverse=True)


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


# ----------------------------------------------------------------------
# The greedy selectors against their definitions
# ----------------------------------------------------------------------

class TestNoveltyGreedyMatchesItsDefinition:
    """The B-Novelty extractor prices candidates incrementally, keeping only the
    k_nn smallest distances per selected behaviour. That is an optimisation of
    "recompute B-Novelty for every candidate", so it is checked against exactly
    that, step by step, on a pool with duplicates and ties."""

    @pytest.mark.parametrize('k_nn', [1, 2, 3])
    @pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
    def test_every_step_matches_the_naive_greedy(self, counter, k, k_nn):
        plans = counter.make_plans(
            (1, 'RSI'), (1, 'RIS'), (2, 'SIR'), (3, 'IRS'),
            (1, 'RIS'), (4, 'ISR'), (2, 'SIR'),
        )

        selection = counter.extract(plans, k=k, indicator='bnovelty',
                                    k_nn=k_nn, trace=True)

        chosen = []
        for step in range(k):
            # The stated rule: the best candidate, ties going to the lowest plan
            # index. The tolerance is what makes "tied" mean *mathematically*
            # tied -- these sums differ by an ulp depending on the order they
            # are accumulated in, which is not a difference in the indicator.
            best, best_value = None, None
            for plan in plans:
                if any(plan is picked for picked in chosen):
                    continue
                value = naive_b_novelty(counter, chosen + [plan], k_nn)
                if best_value is None or value > best_value + 1e-9:
                    best, best_value = plan, value
            chosen.append(best)
            assert selection.order[step] is best
            assert selection.scores[step] == pytest.approx(best_value)


class TestSelectorsHonourTheirIndicators:
    def test_coverage_selection_covers_every_behaviour_it_can(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

        selected = counter.extract(plans, k=3, indicator='bcoverage')

        assert counter.b_coverage(selected) == 3

    def test_bdc_still_names_the_coverage_selector(self, counter):
        plans = counter.make_plans((1, 'RSI'), (1, 'RIS'), (2, 'SIR'))

        assert [p.behaviour for p in counter.extract(plans, k=2, indicator='bdc')] == \
               [p.behaviour for p in counter.extract(plans, k=2, indicator='bcoverage')]

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


class TestTiesGoToTheLowestPlanIndex:
    """Greedy scores are sums of the same distances in different orders, so two
    mathematically equal candidates differ by an ulp about one time in a
    hundred. The tie policy is what stops that ulp from choosing the selection.
    """

    def test_the_tie_tolerance_treats_an_ulp_apart_as_tied(self):
        from behaviour_diversity_counter.behaviour_diversity_counter import strictly_better

        assert not strictly_better(1.5, 1.4999999999999998)
        assert not strictly_better(1.4999999999999998, 1.5)
        assert strictly_better(1.5001, 1.5)
        assert strictly_better(0.0, None)

    def test_equidistant_candidates_are_taken_in_pool_order(self, counter):
        """After 'RIS' is taken, 'RSI' and 'SIR' are both at distance exactly 1.0
        from it (Hamming 2, same rover count), so the second pick is a genuine
        tie and must fall to the earlier plan."""
        plans = counter.make_plans((1, 'RIS'), (1, 'RSI'), (1, 'SIR'))
        ris, rsi, sir = (plan.behaviour for plan in plans)
        assert counter._pair_distance(ris, rsi) == counter._pair_distance(ris, sir)

        for indicator in ('bmaxsum', 'bnovelty'):
            selection = counter.extract(plans, k=2, indicator=indicator, trace=True)
            assert selection.order[1] is plans[1], indicator

    def test_a_tied_seed_pair_is_taken_in_pool_order(self, counter):
        """Three behaviours differing only in rover count are mutually equidistant,
        so every pair is a candidate seed and farthest-first must take the first."""
        plans = counter.make_plans((1, 'RIS'), (2, 'RIS'), (3, 'RIS'))

        selection = counter.extract(plans, k=3, indicator='bmaxmin', trace=True)

        assert selection.order[:2] == [plans[0], plans[1]]

    def test_the_first_of_several_equally_good_prefixes_is_returned(self, counter):
        """A plateau in the trace must not pad the selection: B-MaxMin holds its
        value here for a step, and the shorter prefix is the one returned."""
        plans = counter.make_plans((1, 'RIS'), (2, 'SIR'), (3, 'SIR'), (4, 'SIR'))
        selection = counter.extract(plans, k=4, indicator='bmaxmin', trace=True)

        assert selection.best_step == 2
        assert selection.scores[1] == pytest.approx(max(selection.scores))
