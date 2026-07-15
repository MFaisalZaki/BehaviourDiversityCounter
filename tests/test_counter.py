"""Tests for BehaviourDiversityCounter -- the object that ties the dimensions
together: simulate each plan, join the per-dimension tokens into one behaviour
string, then count / select / score over those strings."""

import pytest

from behaviour_diversity_counter.behaviour_diversity_counter import (
    BehaviourDiversityCounter,
    InapplicablePlanError,
    features_map,
)


class TestConstruction:
    def test_features_are_built_from_name_addinfo_pairs(self, task, resource_file):
        counter = BehaviourDiversityCounter(task, [], [('go', None), ('ru', resource_file)])

        assert set(counter.features) == {'go', 'ru'}
        assert isinstance(counter.features['go'], features_map['go'])

    def test_plan_list_is_materialised_from_any_iterable(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, iter([plan_l1_then_l2]), [('go', None)])

        assert counter.planslist == [plan_l1_then_l2]

    def test_unknown_feature_name_is_rejected(self, task):
        with pytest.raises(KeyError):
            BehaviourDiversityCounter(task, [], [('nope', None)])


class TestCount:
    def test_plans_differing_only_in_goal_order_are_distinct_behaviours(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None)]
        )

        assert counter.count() == 2

    def test_plans_agreeing_on_the_dimension_collapse_to_one_behaviour(
        self, task, plan_l1_then_l2, plan_two_trucks
    ):
        """Both deliver l1 then l2, so under 'go' alone they are one behaviour."""
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_two_trucks], [('go', None)]
        )

        assert counter.count() == 1

    def test_adding_a_dimension_separates_previously_equal_plans(
        self, task, resource_file, plan_l1_then_l2, plan_two_trucks
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_two_trucks], [('go', None), ('ru', resource_file)]
        )

        assert counter.count() == 2

    def test_duplicate_plans_count_once(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l1_then_l2], [('go', None)]
        )

        assert counter.count() == 1

    def test_empty_plan_list_counts_zero(self, task):
        counter = BehaviourDiversityCounter(task, [], [('go', None)])

        assert counter.count() == 0

    def test_count_is_idempotent(self, task, plan_l1_then_l2, plan_l2_then_l1):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None)]
        )

        assert counter.count() == counter.count() == 2

    def test_behaviour_string_joins_one_token_per_dimension(
        self, task, plan_l1_then_l2
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2], [('go', None), ('cb', {'q': 1.0})]
        )
        counter.count()

        assert counter.collected_behaviours == {
            'go:delivered(l1)->delivered(l2) $$ cb:4'
        }

    def test_each_plan_is_annotated_with_its_behaviour(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [plan_l1_then_l2], [('go', None)])

        counter.count()

        assert plan_l1_then_l2.behaviour == 'go:delivered(l1)->delivered(l2)'


class TestOptimise:
    def test_returns_at_most_k_plans(self, task, plan_l1_then_l2, plan_l2_then_l1):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None)]
        )

        assert len(counter.optimise(k=1)) == 1

    def test_returns_every_plan_when_k_exceeds_the_pool(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None)]
        )

        assert len(counter.optimise(k=99)) == 2

    def test_selection_prefers_distinct_behaviours_over_duplicates(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """Two plans share a behaviour and one is unique; k=2 must not return the
        two look-alikes -- it must cover both behaviours first."""
        counter = BehaviourDiversityCounter(
            task,
            [plan_l1_then_l2, plan_l1_then_l2, plan_l2_then_l1],
            [('go', None)],
        )

        selected = counter.optimise(k=2)

        assert len({p.behaviour for p in selected}) == 2

    def test_k_of_zero_selects_nothing(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [plan_l1_then_l2], [('go', None)])

        assert counter.optimise(k=0) == []

    def test_optimise_populates_collected_behaviours(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None)]
        )

        counter.optimise(k=1)

        assert len(counter.collected_behaviours) == 2


class TestEstimatedBehaviourCount:
    def test_is_unset_before_optimise_runs(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [plan_l1_then_l2], [('go', None)])

        assert counter.estimated_behaviour_count() == -1

    def test_single_dimension_estimate_is_that_dimension_domain(
        self, task, plan_l1_then_l2
    ):
        counter = BehaviourDiversityCounter(task, [plan_l1_then_l2], [('go', None)])

        counter.count()

        # 2 goals -> 2! orderings
        assert counter.estimated_behaviour_count() == 2

    def test_estimate_is_the_product_over_dimensions(
        self, task, resource_file, plan_l1_then_l2
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2], [('go', None), ('ru', resource_file)]
        )

        counter.count()

        # go: 2! = 2, ru: 2**2 - 1 = 3
        assert counter.estimated_behaviour_count() == 6

    def test_estimate_upper_bounds_the_observed_count(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None)]
        )

        assert counter.count() <= counter.estimated_behaviour_count()


class TestNoveltyScore:
    def test_identical_plans_score_zero(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l1_then_l2], [('go', None)]
        )

        assert counter.compute_novelty_score() == 0.0

    def test_fully_reordered_plans_score_one(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None)]
        )

        assert counter.compute_novelty_score() == 1.0

    def test_score_averages_over_dimensions(
        self, task, resource_file, plan_l1_then_l2, plan_two_trucks
    ):
        """Same goal ordering (go distance 0), different resource sets
        ({tr1} vs {tr1,tr2} -> ru distance 0.5). Mean = 0.25."""
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_two_trucks], [('go', None), ('ru', resource_file)]
        )

        assert counter.compute_novelty_score() == 0.25

    def test_score_averages_over_all_plan_pairs(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """Three plans -> pairs (A,B)=1.0, (A,A)=0.0, (B,A)=1.0 -> mean 2/3."""
        counter = BehaviourDiversityCounter(
            task,
            [plan_l1_then_l2, plan_l2_then_l1, plan_l1_then_l2],
            [('go', None)],
        )

        assert counter.compute_novelty_score() == pytest.approx(2 / 3)

    def test_single_plan_has_no_pairs_and_scores_zero(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [plan_l1_then_l2], [('go', None)])

        assert counter.compute_novelty_score() == 0.0

    def test_empty_plan_list_scores_zero(self, task):
        counter = BehaviourDiversityCounter(task, [], [('go', None)])

        assert counter.compute_novelty_score() == 0.0

    def test_no_dimensions_scores_zero(self, task, plan_l1_then_l2, plan_l2_then_l1):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], []
        )

        assert counter.compute_novelty_score() == 0.0

    def test_novelty_works_with_the_cost_bound_dimension(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """Regression: 'cb' raised AttributeError because its distance() expected plan
        objects while compute_novelty_score passes behaviour strings.

        Both plans are 4 actions long, so cb contributes 0.0 and go contributes 1.0.
        """
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('go', None), ('cb', {'q': 1.0})]
        )

        assert counter.compute_novelty_score() == 0.5

    def test_novelty_reflects_a_cost_difference(self, task, domain, make_plan):
        """A 2-action plan against a 4-action plan: |2-4|/4 = 0.5 on cb alone."""
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        short = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        long = make_plan(
            (move, (tr1, l0, l1)), (drop, (tr1, l1)),
            (move, (tr1, l1, l2)), (drop, (tr1, l2)),
        )
        counter = BehaviourDiversityCounter(task, [short, long], [('cb', {'q': 2.0})])

        assert counter.compute_novelty_score() == 0.5

    def test_dimensions_without_a_distance_cannot_be_scored(
        self, task, resource_file, plan_l1_then_l2, plan_l2_then_l1
    ):
        """'rc', 'uv' and 'fn' have no distance(); novelty is only defined over
        the dimensions that implement one."""
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, plan_l2_then_l1], [('rc', resource_file)]
        )

        with pytest.raises(AssertionError):
            counter.compute_novelty_score()


class TestInapplicablePlans:
    """Regression: _simulate_ used to return [] for a plan whose preconditions fail.
    The dimensions read that empty trace as "no goal was ever achieved", so under go
    every goal got first-achieved index -1, the sort stayed stable, and the plan
    reported its goals in declaration order -- byte-identical to what a *valid* plan
    achieving them in that order produces. The invalid plan was counted silently."""

    def test_simulating_an_inapplicable_plan_raises(self, task, inapplicable_plan):
        counter = BehaviourDiversityCounter(task, [inapplicable_plan], [('go', None)])

        with pytest.raises(InapplicablePlanError):
            counter._simulate_(inapplicable_plan)

    def test_the_error_names_the_offending_action_and_step(
        self, task, inapplicable_plan
    ):
        counter = BehaviourDiversityCounter(task, [inapplicable_plan], [('go', None)])

        with pytest.raises(InapplicablePlanError, match=r'drop\(tr1, l1\) at step 0'):
            counter._simulate_(inapplicable_plan)

    def test_counting_an_inapplicable_plan_raises_rather_than_miscounting(
        self, task, plan_l1_then_l2, inapplicable_plan
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, inapplicable_plan], [('go', None)]
        )

        with pytest.raises(InapplicablePlanError):
            counter.count()

    def test_novelty_on_an_inapplicable_plan_raises(
        self, task, plan_l1_then_l2, inapplicable_plan
    ):
        counter = BehaviourDiversityCounter(
            task, [plan_l1_then_l2, inapplicable_plan], [('go', None)]
        )

        with pytest.raises(InapplicablePlanError):
            counter.compute_novelty_score()

    def test_the_error_is_a_value_error(self, task, inapplicable_plan):
        """Callers already catching ValueError keep working."""
        counter = BehaviourDiversityCounter(task, [inapplicable_plan], [('go', None)])

        with pytest.raises(ValueError):
            counter._simulate_(inapplicable_plan)

    def test_a_plan_failing_midway_is_caught(self, task, domain, make_plan):
        """The first action applies; the second does not."""
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        # After move(tr1, l0, l1) the truck is at l1, so drop(tr1, l2) fails.
        plan = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l2)))
        counter = BehaviourDiversityCounter(task, [plan], [('go', None)])

        with pytest.raises(InapplicablePlanError, match='at step 1'):
            counter.count()
