"""Tests for BehaviourDiversityCounter -- the object that ties the dimensions
together. It is bound to a task and its dimensions; every indicator
(b_coverage, b_maxsum) and the extraction (extract) receive the plan set to
work on."""

import pytest

from behaviour_diversity_counter import (
    BehaviourDiversityCounter,
    InapplicablePlanError,
    dimensions_map,
)


class TestConstruction:
    def test_dimensions_are_built_from_name_addinfo_pairs(self, task, resource_file):
        counter = BehaviourDiversityCounter(task, [('go', None), ('ru', resource_file)])

        assert set(counter.dimensions) == {'go', 'ru'}
        assert isinstance(counter.dimensions['go'], dimensions_map['go'])

    def test_indicators_accept_any_plan_iterable(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_coverage(iter([plan_l1_then_l2])) == 1

    def test_unknown_dimension_name_is_rejected(self, task):
        with pytest.raises(ValueError, match='valid keys'):
            BehaviourDiversityCounter(task, [('nope', None)])


class TestBDC:
    def test_plans_differing_only_in_goal_order_are_distinct_behaviours(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_coverage([plan_l1_then_l2, plan_l2_then_l1]) == 2

    def test_plans_agreeing_on_the_dimension_collapse_to_one_behaviour(
        self, task, plan_l1_then_l2, plan_two_trucks
    ):
        """Both deliver l1 then l2, so under 'go' alone they are one behaviour."""
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_coverage([plan_l1_then_l2, plan_two_trucks]) == 1

    def test_adding_a_dimension_separates_previously_equal_plans(
        self, task, resource_file, plan_l1_then_l2, plan_two_trucks
    ):
        counter = BehaviourDiversityCounter(task, [('go', None), ('ru', resource_file)])

        assert counter.b_coverage([plan_l1_then_l2, plan_two_trucks]) == 2

    def test_duplicate_plans_count_once(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_coverage([plan_l1_then_l2, plan_l1_then_l2]) == 1

    def test_empty_plan_list_counts_zero(self, task):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_coverage([]) == 0

    def test_b_coverage_is_idempotent(self, task, plan_l1_then_l2, plan_l2_then_l1):
        counter = BehaviourDiversityCounter(task, [('go', None)])
        plans = [plan_l1_then_l2, plan_l2_then_l1]

        assert counter.b_coverage(plans) == counter.b_coverage(plans) == 2

    def test_one_counter_scores_many_plan_sets(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_coverage([plan_l1_then_l2]) == 1
        assert counter.b_coverage([plan_l1_then_l2, plan_l2_then_l1]) == 2

    def test_behaviour_string_joins_one_token_per_dimension(
        self, task, plan_l1_then_l2
    ):
        counter = BehaviourDiversityCounter(task, [('go', None), ('cb', {'q': 1.0})])

        assert counter.behaviours([plan_l1_then_l2]) == {
            'go:delivered(l1)->delivered(l2) $$ cb:4'
        }

    def test_each_plan_is_annotated_with_its_behaviour(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        counter.b_coverage([plan_l1_then_l2])

        assert plan_l1_then_l2.behaviour == 'go:delivered(l1)->delivered(l2)'


class TestExtract:
    def test_returns_at_most_k_plans(self, task, plan_l1_then_l2, plan_l2_then_l1):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert len(counter.extract([plan_l1_then_l2, plan_l2_then_l1], k=1)) == 1

    def test_returns_every_plan_when_k_exceeds_the_pool(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert len(counter.extract([plan_l1_then_l2, plan_l2_then_l1], k=99)) == 2

    def test_selection_prefers_distinct_behaviours_over_duplicates(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """Two plans share a behaviour and one is unique; k=2 must not return the
        two look-alikes -- it must cover both behaviours first."""
        counter = BehaviourDiversityCounter(task, [('go', None)])

        selected = counter.extract(
            [plan_l1_then_l2, plan_l1_then_l2, plan_l2_then_l1], k=2
        )

        assert len({p.behaviour for p in selected}) == 2

    def test_k_of_zero_selects_nothing(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.extract([plan_l1_then_l2], k=0) == []

    def test_extracted_subsets_can_be_scored_by_the_indicators(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        subset = counter.extract(
            [plan_l1_then_l2, plan_l1_then_l2, plan_l2_then_l1], k=2
        )

        assert counter.b_coverage(subset) == 2
        assert counter.b_maxsum(subset) == 1.0

    def test_remaining_slots_are_filled_with_duplicate_behaviours(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """Two distinct behaviours cannot fill k=3; the third slot takes a
        duplicate, which leaves the indicator value unchanged."""
        counter = BehaviourDiversityCounter(task, [('go', None)])

        selected = counter.extract(
            [plan_l1_then_l2, plan_l1_then_l2, plan_l2_then_l1], k=3
        )

        assert len(selected) == 3
        assert len({p.behaviour for p in selected}) == 2

    def test_unknown_indicator_is_rejected(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        with pytest.raises(ValueError, match='valid indicators'):
            counter.extract([plan_l1_then_l2], k=1, indicator='nope')


class TestDeclaredWeights:
    """A dimension carries its own weight and applies it; the counter decides
    what the values are, because every rule about them is a rule about the set.
    """

    def test_a_dimension_declares_its_weight_in_addinfo(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """The two plans reverse the goal order at equal cost, so `go` scores
        them 1.0 apart and `cb` 0.0. Under the declared 1/4 the pair is 0.25;
        under the uniform default it would be 0.5."""
        counter = BehaviourDiversityCounter(
            task, [('go', {'weight': 0.25}), ('cb', {'weight': 0.75})])

        assert counter.b_maxsum([plan_l1_then_l2, plan_l2_then_l1]) == pytest.approx(0.25)

    def test_declaring_is_all_or_nothing(self, task):
        with pytest.raises(ValueError, match=r"no weight given for dimension\(s\): \['cb'\]"):
            BehaviourDiversityCounter(task, [('go', {'weight': 1.0}), ('cb', None)])

    def test_no_declaration_leaves_the_uniform_default(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None), ('cb', None)])

        assert counter.b_maxsum([plan_l1_then_l2, plan_l2_then_l1]) == pytest.approx(0.5)


class TestExtractBMaxSum:
    """extract(plans, k, indicator='bmaxsum') -- the greedy extraction: repeatedly
    add the plan whose behaviour has the greatest summed distance to the behaviours
    already selected."""

    def test_greedy_covers_distinct_behaviours_before_duplicates(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        selected = counter.extract(
            [plan_l1_then_l2, plan_l1_then_l2, plan_l2_then_l1],
            k=2, indicator='bmaxsum',
        )

        assert len({p.behaviour for p in selected}) == 2

    def test_greedy_adds_the_farthest_behaviour_first(self, task, domain, make_plan):
        """From costs {2, 3, 4} with the cost-2 plan picked first (singletons all
        score zero), cost 4 gains 1/2 while cost 3 gains only 1/3."""
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        cost2 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        cost3 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)),
                          (move, (tr1, l1, l2)))
        cost4 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)),
                          (move, (tr1, l1, l2)), (drop, (tr1, l2)))
        counter = BehaviourDiversityCounter(task, [('cb', {'q': 2.0})])

        selected = counter.extract([cost2, cost3, cost4], k=2, indicator='bmaxsum')

        assert {p.behaviour for p in selected} == {'cb:2', 'cb:4'}

    def test_the_greedy_opens_on_the_farthest_pair_not_the_first_plan(
        self, task, domain, make_plan
    ):
        """The opening pick is the better half of the farthest pair, not plan 0.

        Costs {5, 2, 10}, in that pool order. Every singleton sums to zero, so
        the opening pick gets no signal from the objective at all -- and the
        rule B-MaxSum is trying to maximise says to take the pair that sums
        highest. That is {2, 10}, at |2 - 10| / 10 = 0.8. Opening on plan 0
        instead and then taking its farthest neighbour gives {5, 2}, at
        |5 - 2| / 5 = 0.6, which is simply a worse answer to the question asked.
        """
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        cycle = [(move, (tr1, l0, l1)), (move, (tr1, l1, l2)), (move, (tr1, l2, l0))]
        cost2 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        cost5 = make_plan(*(cycle * 2)[:5])
        cost10 = make_plan(*(cycle * 4)[:10])
        counter = BehaviourDiversityCounter(task, [('cb', None)])

        selected = counter.extract([cost5, cost2, cost10], k=2, indicator='bmaxsum')

        assert {p.behaviour for p in selected} == {'cb:2', 'cb:10'}
        assert counter.b_maxsum(selected) == pytest.approx(0.8)

    def test_duplicates_are_picked_only_when_the_pool_is_exhausted(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        selected = counter.extract(
            [plan_l1_then_l2, plan_l1_then_l2, plan_l2_then_l1],
            k=3, indicator='bmaxsum',
        )

        assert len(selected) == 3
        assert selected[2].behaviour == selected[0].behaviour


class TestBMaxSum:
    def test_identical_plans_score_zero(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_maxsum([plan_l1_then_l2, plan_l1_then_l2]) == 0.0

    def test_fully_reordered_plans_score_one(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_maxsum([plan_l1_then_l2, plan_l2_then_l1]) == 1.0

    def test_a_pair_distance_averages_over_dimensions(
        self, task, resource_file, plan_l1_then_l2, plan_two_trucks
    ):
        """Same goal ordering (go distance 0), different resource sets
        ({tr1} vs {tr1,tr2} -> ru distance 0.5). The single pair scores 0.25."""
        counter = BehaviourDiversityCounter(task, [('go', None), ('ru', resource_file)])

        assert counter.b_maxsum([plan_l1_then_l2, plan_two_trucks]) == 0.25

    def test_duplicate_behaviours_are_discarded_before_aggregation(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """Three plans but two distinct behaviours -> the duplicate contributes
        nothing; only the pair (A,B)=1.0 is summed."""
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_maxsum(
            [plan_l1_then_l2, plan_l2_then_l1, plan_l1_then_l2]
        ) == 1.0

    def test_the_metric_sums_over_distinct_behaviour_pairs(
        self, task, domain, make_plan
    ):
        """Plans of costs 2, 3 and 4 under cb alone: |2-3|/3 + |2-4|/4 + |3-4|/4
        = 13/12 -- a sum over pairs, not an average, so it can exceed 1."""
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        cost2 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        cost3 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)),
                          (move, (tr1, l1, l2)))
        cost4 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)),
                          (move, (tr1, l1, l2)), (drop, (tr1, l2)))
        counter = BehaviourDiversityCounter(task, [('cb', {'q': 2.0})])

        assert counter.b_maxsum([cost2, cost3, cost4]) == pytest.approx(13 / 12)

    def test_single_plan_has_no_pairs_and_scores_zero(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_maxsum([plan_l1_then_l2]) == 0.0

    def test_empty_plan_list_scores_zero(self, task):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        assert counter.b_maxsum([]) == 0.0

    def test_no_dimensions_scores_zero(self, task, plan_l1_then_l2, plan_l2_then_l1):
        counter = BehaviourDiversityCounter(task, [])

        assert counter.b_maxsum([plan_l1_then_l2, plan_l2_then_l1]) == 0.0

    def test_b_maxsum_works_with_the_cost_bound_dimension(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        """Regression: 'cb' raised AttributeError because its distance() expected plan
        objects while b_maxsum passes behaviour strings.

        Both plans are 4 actions long, so cb contributes 0.0 and go contributes 1.0.
        """
        counter = BehaviourDiversityCounter(task, [('go', None), ('cb', {'q': 1.0})])

        assert counter.b_maxsum([plan_l1_then_l2, plan_l2_then_l1]) == 0.5

    def test_b_maxsum_reflects_a_cost_difference(self, task, domain, make_plan):
        """A 2-action plan against a 4-action plan: |2-4|/4 = 0.5 on cb alone."""
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        short = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        long = make_plan(
            (move, (tr1, l0, l1)), (drop, (tr1, l1)),
            (move, (tr1, l1, l2)), (drop, (tr1, l2)),
        )
        counter = BehaviourDiversityCounter(task, [('cb', {'q': 2.0})])

        assert counter.b_maxsum([short, long]) == 0.5

    def test_dimensions_without_a_distance_cannot_be_scored(
        self, task, resource_file, plan_l1_then_l2, plan_two_trucks
    ):
        """'rc', 'uv' and 'fn' have no distance(); the B-MaxSum metric is only
        defined over the dimensions that implement one. The two plans must differ
        under 'rc', or deduplication would leave no pair to measure."""
        counter = BehaviourDiversityCounter(task, [('rc', resource_file)])

        with pytest.raises(AssertionError):
            counter.b_maxsum([plan_l1_then_l2, plan_two_trucks])


class TestCaching:
    def test_a_plan_is_simulated_once_per_counter(self, task, plan_l1_then_l2):
        counter = BehaviourDiversityCounter(task, [('go', None)])
        simulations = []
        original = counter._simulate
        counter._simulate = lambda plan: (simulations.append(plan), original(plan))[1]

        counter.b_coverage([plan_l1_then_l2])
        counter.b_maxsum([plan_l1_then_l2])
        counter.extract([plan_l1_then_l2], k=1)

        assert len(simulations) == 1

    def test_pair_distances_are_computed_once_per_behaviour_pair(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])
        calls = []
        dimension = counter.dimensions['go']
        original = dimension.distance
        dimension.distance = lambda b1, b2: (calls.append((b1, b2)), original(b1, b2))[1]
        plans = [plan_l1_then_l2, plan_l2_then_l1]

        counter.b_maxsum(plans)
        counter.b_maxsum(plans)
        counter.extract(plans, k=2, indicator='bmaxsum')

        assert len(calls) == 1


class TestInapplicablePlans:
    """Regression: _simulate used to return [] for a plan whose preconditions fail.
    The dimensions read that empty trace as "no goal was ever achieved", so under go
    every goal got first-achieved index -1, the sort stayed stable, and the plan
    reported its goals in declaration order -- byte-identical to what a *valid* plan
    achieving them in that order produces. The invalid plan was counted silently."""

    def test_simulating_an_inapplicable_plan_raises(self, task, inapplicable_plan):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        with pytest.raises(InapplicablePlanError):
            counter._simulate(inapplicable_plan)

    def test_the_error_names_the_offending_action_and_step(
        self, task, inapplicable_plan
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        with pytest.raises(InapplicablePlanError, match=r'drop\(tr1, l1\) at step 0'):
            counter._simulate(inapplicable_plan)

    def test_counting_an_inapplicable_plan_raises_rather_than_miscounting(
        self, task, plan_l1_then_l2, inapplicable_plan
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        with pytest.raises(InapplicablePlanError):
            counter.b_coverage([plan_l1_then_l2, inapplicable_plan])

    def test_b_maxsum_on_an_inapplicable_plan_raises(
        self, task, plan_l1_then_l2, inapplicable_plan
    ):
        counter = BehaviourDiversityCounter(task, [('go', None)])

        with pytest.raises(InapplicablePlanError):
            counter.b_maxsum([plan_l1_then_l2, inapplicable_plan])

    def test_the_error_is_a_value_error(self, task, inapplicable_plan):
        """Callers already catching ValueError keep working."""
        counter = BehaviourDiversityCounter(task, [('go', None)])

        with pytest.raises(ValueError):
            counter._simulate(inapplicable_plan)

    def test_a_plan_failing_midway_is_caught(self, task, domain, make_plan):
        """The first action applies; the second does not."""
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        # After move(tr1, l0, l1) the truck is at l1, so drop(tr1, l2) fails.
        plan = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l2)))
        counter = BehaviourDiversityCounter(task, [('go', None)])

        with pytest.raises(InapplicablePlanError, match='at step 1'):
            counter.b_coverage([plan])
