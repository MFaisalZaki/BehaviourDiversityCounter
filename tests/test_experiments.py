"""The experiment helpers: the pieces that decide what the numbers mean.

These need no PDDL: everything below reads behaviour strings, which is all the
helpers themselves ever see.
"""

import pytest

from paperexps import exp_b, exp_c, harness


class FakePool:
    """Enough of harness.Pool for the helpers under test."""

    def __init__(self, behaviours, dimensions):
        self.behaviours = list(behaviours)
        self.distinct = list(dict.fromkeys(self.behaviours))
        self.dimensions = list(dimensions)
        self.task_id = 'fake-task'


def plans_for(behaviours):
    return [type('P', (), {'behaviour': behaviour})() for behaviour in behaviours]


class TestDimensionValue:
    def test_a_token_is_matched_by_prefix_not_substring(self):
        """'truck1' inside the go token contains 'ru', which is exactly the
        confusion the dimensions themselves guard against."""
        behaviour = 'go:at(truck1)->at(truck2) $$ ru:truck1'

        assert harness.dimension_value(behaviour, 'ru') == 'truck1'
        assert harness.dimension_value(behaviour, 'go') == 'at(truck1)->at(truck2)'

    def test_a_missing_dimension_is_none(self):
        assert harness.dimension_value('go:a', 'ru') is None


class TestJaccard:
    def test_disjoint_sets_score_zero(self):
        assert harness.jaccard({1, 2}, {3, 4}) == 0.0

    def test_identical_sets_score_one(self):
        assert harness.jaccard({1, 2}, {1, 2}) == 1.0

    def test_two_empty_sets_are_identical(self):
        assert harness.jaccard(set(), set()) == 1.0

    def test_partial_overlap(self):
        assert harness.jaccard({1, 2, 3}, {2, 3, 4}) == pytest.approx(2 / 4)


class TestDerivedSeeds:
    def test_different_purposes_get_different_seeds(self):
        """The hidden-preference draw has to be independent of the selection,
        or the thing no selector is supposed to see would be correlated with
        what they do see."""
        select = harness.derive_seed(2026, 'task-1', 'select')
        hidden = harness.derive_seed(2026, 'task-1', 'hidden-preference')

        assert select != hidden

    def test_different_tasks_get_different_seeds(self):
        assert harness.derive_seed(2026, 'task-1', 'select') != \
               harness.derive_seed(2026, 'task-2', 'select')

    def test_the_same_inputs_give_the_same_seed(self):
        assert harness.derive_seed(2026, 'task-1', 'select') == \
               harness.derive_seed(2026, 'task-1', 'select')

    def test_seeds_stay_within_the_positive_range(self):
        for task in ('a', 'bb', 'ccc', 'x' * 40):
            assert 0 <= harness.derive_seed(2026, task, 'select') < 2 ** 31


class TestHiddenPreferenceDraws:
    @pytest.fixture
    def pool(self):
        return FakePool(['go:A $$ ru:r1', 'go:B $$ ru:r1', 'go:C $$ ru:r2'],
                        ['go', 'ru'])

    def test_every_drawn_value_is_one_some_plan_exhibits(self, pool):
        """A preference no plan can satisfy would make the measure a coin toss
        rather than a question about the selection."""
        attainable = {('go', 'A'), ('go', 'B'), ('go', 'C'),
                      ('ru', 'r1'), ('ru', 'r2')}

        draws = exp_b.hidden_preferences(pool, 200, seed=7)

        assert set(draws) <= attainable

    def test_the_same_seed_draws_the_same_preferences(self, pool):
        assert exp_b.hidden_preferences(pool, 50, seed=7) == \
               exp_b.hidden_preferences(pool, 50, seed=7)

    def test_a_different_seed_draws_differently(self, pool):
        assert exp_b.hidden_preferences(pool, 50, seed=7) != \
               exp_b.hidden_preferences(pool, 50, seed=8)

    def test_a_pool_with_no_usable_dimension_draws_nothing(self):
        assert exp_b.hidden_preferences(FakePool(['x'], []), 50, seed=7) == []

    def test_hits_count_the_satisfied_draws(self, pool):
        selected = plans_for(['go:A $$ ru:r1'])
        draws = [('go', 'A'), ('go', 'B'), ('ru', 'r1'), ('ru', 'r2')]

        assert exp_b.hidden_preference_hits(selected, draws) == 2


class TestHiddenPreferenceExact:
    """The closed form beside the sampled rate.

    A draw is a dimension uniformly, then one of its attainable values
    uniformly, so the probability of satisfying it is the mean over dimensions
    of the fraction of that dimension's values the selection covers.
    """

    @pytest.fixture
    def pool(self):
        return FakePool(['go:A $$ ru:r1', 'go:B $$ ru:r1', 'go:C $$ ru:r2'],
                        ['go', 'ru'])

    def test_covering_everything_is_certain(self, pool):
        values = exp_b.pool_dimension_values(pool)
        selected = plans_for(pool.distinct)

        exact, coverage = exp_b.hidden_preference_exact(selected, values)

        assert exact == pytest.approx(1.0)
        assert coverage == 'go:3/3,ru:2/2'

    def test_one_behaviour_covers_one_value_of_each_dimension(self, pool):
        values = exp_b.pool_dimension_values(pool)

        exact, coverage = exp_b.hidden_preference_exact(
            plans_for(['go:A $$ ru:r1']), values)

        assert exact == pytest.approx((1 / 3 + 1 / 2) / 2)
        assert coverage == 'go:1/3,ru:1/2'

    def test_a_constant_dimension_is_a_free_hit_for_every_selection(self):
        """This is what the sampled rate is really measuring on these
        benchmarks: where every plan in the pool uses the same resources, half
        of every draw is satisfied by any selection alive, and the ru column
        contributes 1.0 to all five rules alike."""
        pool = FakePool(['go:A $$ ru:r1', 'go:B $$ ru:r1', 'go:C $$ ru:r1'],
                        ['go', 'ru'])
        values = exp_b.pool_dimension_values(pool)

        exact, coverage = exp_b.hidden_preference_exact(
            plans_for(['go:A $$ ru:r1']), values)

        assert coverage == 'go:1/3,ru:1/1'
        assert exact == pytest.approx((1 / 3 + 1.0) / 2)

    def test_in_one_dimension_it_is_just_the_coverage_fraction(self):
        """With a single dimension a behaviour *is* its value, so the measure
        cannot separate two rules that cover the same number of behaviours --
        only how many, never which."""
        pool = FakePool(['go:A', 'go:B', 'go:C', 'go:D'], ['go'])
        values = exp_b.pool_dimension_values(pool)

        first, _ = exp_b.hidden_preference_exact(plans_for(['go:A', 'go:B']), values)
        other, _ = exp_b.hidden_preference_exact(plans_for(['go:C', 'go:D']), values)

        assert first == other == pytest.approx(0.5)

    def test_a_pool_with_no_usable_dimension_reports_nothing(self):
        assert exp_b.hidden_preference_exact(plans_for(['x']), {}) == ('', '')


class TestWeightSweep:
    def test_the_sweep_hits_zero_and_one_exactly(self):
        steps = exp_c.weight_steps(0.05)

        assert len(steps) == 21
        assert steps[0] == 0.0
        assert steps[-1] == 1.0

    def test_no_float_drift_across_the_sweep(self):
        """0.05 accumulated twenty times is not 1.0; these have to be exact so
        that w_go = 0.5 is the same number on every task."""
        assert exp_c.weight_steps(0.05)[10] == 0.5

    def test_a_coarser_step_still_ends_at_one(self):
        assert exp_c.weight_steps(0.25) == [0.0, 0.25, 0.5, 0.75, 1.0]


class TestStableHash:
    def test_order_does_not_matter(self):
        assert exp_c.stable_hash([3, 1, 2]) == exp_c.stable_hash([1, 2, 3])

    def test_different_sets_hash_differently(self):
        assert exp_c.stable_hash([1, 2]) != exp_c.stable_hash([1, 3])

    def test_it_is_the_same_in_every_process(self):
        """sha1, not hash(): Python randomises string hashing per process, so
        hash() would compare an array job's jobs as noise."""
        assert exp_c.stable_hash(['a', 'b']) == '4a3dec2d1f824528'
