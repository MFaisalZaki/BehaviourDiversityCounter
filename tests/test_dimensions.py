"""Tests for the individual dimensions.

Each dimension turns a simulated plan into a behaviour token. Tokens from all
enabled dimensions are joined with ' $$ ' by BehaviourDiversityCounter, and the
distance() implementations parse their own token back out of that joined string.
"""

import math

import pytest
from unified_planning.shortcuts import SequentialSimulator

from behaviour_diversity_counter import (
    BehaviourDiversityCounter,
)
from behaviour_diversity_counter.dimensions.base import BehaviourDimension
from behaviour_diversity_counter.dimensions.cost_bound_makespan_optimal import (
    MakespanOptimalCostDimension,
)
from behaviour_diversity_counter.dimensions.functions import NumericFunctionDimension
from behaviour_diversity_counter.dimensions.goal_predicate_ordering import (
    GoalPredicatesOrderingDimension,
)
from behaviour_diversity_counter.dimensions.resources import (
    ResourceCountDimension,
    ResourceUsedDimension,
)
from behaviour_diversity_counter.dimensions.utility_value import UtilityValueDimension


def simulate(task, plan):
    """Attach the state trace a dimension's plan_behaviour() reads."""
    simulator = SequentialSimulator(problem=task)
    state = simulator.get_initial_state()
    states = [state]
    for action_instance in plan.actions:
        state = simulator.apply(state, action_instance)
        assert state is not None, f'fixture plan is inapplicable at {action_instance}'
        states.append(state)
    setattr(plan, 'states', states)
    return plan


class TestBase:
    def test_subclasses_must_implement_estimate_domain(self, task):
        with pytest.raises(AssertionError, match='implemented by the child class'):
            BehaviourDimension(task, 'x', None)._estimate_domain()

    def test_subclasses_must_implement_distance(self, task):
        with pytest.raises(AssertionError, match='implemented by the child class'):
            BehaviourDimension(task, 'x', None).distance('a', 'b')


class TestGoalPredicatesOrdering:
    def test_behaviour_records_the_order_goals_were_first_achieved(
        self, task, plan_l1_then_l2
    ):
        dimension = GoalPredicatesOrderingDimension(task)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert behaviour == 'go:delivered(l1)->delivered(l2)'

    def test_reversed_delivery_yields_the_reversed_token(self, task, plan_l2_then_l1):
        dimension = GoalPredicatesOrderingDimension(task)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l2_then_l1))

        assert behaviour == 'go:delivered(l2)->delivered(l1)'

    def test_which_truck_delivers_does_not_change_the_ordering(
        self, task, plan_l1_then_l2, plan_two_trucks
    ):
        dimension = GoalPredicatesOrderingDimension(task)

        assert dimension.plan_behaviour(simulate(task, plan_l1_then_l2)) == (
            dimension.plan_behaviour(simulate(task, plan_two_trucks))
        )

    def test_observed_behaviours_accumulate_in_domain(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        dimension = GoalPredicatesOrderingDimension(task)

        dimension.plan_behaviour(simulate(task, plan_l1_then_l2))
        dimension.plan_behaviour(simulate(task, plan_l2_then_l1))

        assert dimension.domain == {
            'delivered(l1)->delivered(l2)',
            'delivered(l2)->delivered(l1)',
        }

    def test_estimated_domain_is_the_permutations_of_the_goals(self, task):
        dimension = GoalPredicatesOrderingDimension(task)

        dimension._estimate_domain()

        assert dimension.estimated_domain_size == math.factorial(2) == 2

    def test_distance_is_zero_for_identical_orderings(self, task, plan_l1_then_l2):
        dimension = GoalPredicatesOrderingDimension(task)
        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert dimension.distance(behaviour, behaviour) == 0.0

    def test_distance_is_one_when_every_position_differs(
        self, task, plan_l1_then_l2, plan_l2_then_l1
    ):
        dimension = GoalPredicatesOrderingDimension(task)
        first = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))
        second = dimension.plan_behaviour(simulate(task, plan_l2_then_l1))

        assert dimension.distance(first, second) == 1.0

    def test_distance_is_symmetric(self, task, plan_l1_then_l2, plan_l2_then_l1):
        dimension = GoalPredicatesOrderingDimension(task)
        first = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))
        second = dimension.plan_behaviour(simulate(task, plan_l2_then_l1))

        assert dimension.distance(first, second) == dimension.distance(second, first)

    def test_distance_requires_its_own_token_to_be_present(self, task, plan_l1_then_l2):
        dimension = GoalPredicatesOrderingDimension(task)
        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        with pytest.raises(AssertionError, match='should be present'):
            dimension.distance(behaviour, 'cb:4')

    def test_distance_finds_its_token_among_others(self, task, plan_l1_then_l2):
        """The token is located by prefix, so 'ru:...' must not be mistaken for it."""
        dimension = GoalPredicatesOrderingDimension(task)
        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        combined = f'cb:4 $$ {behaviour} $$ ru:tr1'

        assert dimension.distance(combined, behaviour) == 0.0


class TestMakespanOptimalCost:
    def test_behaviour_is_the_plan_length(self, task, plan_l1_then_l2):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        assert dimension.plan_behaviour(plan_l1_then_l2) == 'cb:4'

    def test_plan_lengths_accumulate_in_domain(
        self, task, plan_l1_then_l2, inapplicable_plan
    ):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        dimension.plan_behaviour(plan_l1_then_l2)
        dimension.plan_behaviour(inapplicable_plan)

        assert dimension.domain == {4, 1}

    def test_estimate_requires_a_q_value(self, task):
        dimension = MakespanOptimalCostDimension(task, {})

        with pytest.raises(AssertionError, match='q value should be provided'):
            dimension._estimate_domain()

    def test_optimal_only_bound_admits_a_single_cost(self, task, plan_l1_then_l2):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})
        dimension.plan_behaviour(plan_l1_then_l2)

        dimension._estimate_domain()

        assert dimension.estimated_domain_size == 1

    def test_relaxed_bound_admits_costs_up_to_q_times_optimal(
        self, task, plan_l1_then_l2
    ):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.5})
        dimension.plan_behaviour(plan_l1_then_l2)  # optimal cost observed = 4

        dimension._estimate_domain()

        # costs 4, 5, 6 -- inclusive of int(4 * 1.5)
        assert dimension.estimated_domain_size == 3

    def test_distance_accepts_behaviour_strings_like_the_other_dimensions(self, task):
        """Regression: distance() read .actions off its arguments, so it only worked
        on plan objects -- but b_maxsum passes behaviour strings."""
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        # |4 - 6| / max(4, 6)
        assert dimension.distance('cb:4', 'cb:6') == pytest.approx(1 / 3)

    def test_distance_is_zero_for_equal_costs(self, task):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        assert dimension.distance('cb:4', 'cb:4') == 0.0

    def test_distance_is_normalised_into_the_unit_interval(self, task):
        """The score averages across dimensions, so cost cannot be unbounded."""
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        assert dimension.distance('cb:1', 'cb:1000') <= 1.0
        assert dimension.distance('cb:0', 'cb:9') == 1.0

    def test_two_empty_plans_are_identical(self, task):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        assert dimension.distance('cb:0', 'cb:0') == 0.0

    def test_distance_is_symmetric(self, task):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        assert dimension.distance('cb:4', 'cb:6') == dimension.distance('cb:6', 'cb:4')

    def test_distance_finds_its_token_among_others(self, task):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        combined = 'go:delivered(l1)->delivered(l2) $$ cb:4 $$ ru:tr1'

        assert dimension.distance(combined, 'cb:4') == 0.0

    def test_distance_requires_its_own_token_to_be_present(self, task):
        dimension = MakespanOptimalCostDimension(task, {'q': 1.0})

        with pytest.raises(AssertionError, match='should be present'):
            dimension.distance('go:delivered(l1)', 'cb:4')


class TestResourceUsed:
    def test_behaviour_lists_only_the_resources_actually_used(
        self, task, resource_file, plan_l1_then_l2
    ):
        dimension = ResourceUsedDimension(task, resource_file)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert behaviour == 'ru:tr1'

    def test_behaviour_is_a_sorted_set_of_used_resources(
        self, task, resource_file, plan_two_trucks
    ):
        dimension = ResourceUsedDimension(task, resource_file)

        behaviour = dimension.plan_behaviour(simulate(task, plan_two_trucks))

        assert behaviour == 'ru:tr1,tr2'

    def test_usage_counts_do_not_affect_the_used_set(
        self, task, resource_file, plan_l1_then_l2, plan_l2_then_l1
    ):
        dimension = ResourceUsedDimension(task, resource_file)

        assert dimension.plan_behaviour(simulate(task, plan_l1_then_l2)) == (
            dimension.plan_behaviour(simulate(task, plan_l2_then_l1))
        )

    def test_estimated_domain_is_the_non_empty_subsets_of_resources(
        self, task, resource_file
    ):
        dimension = ResourceUsedDimension(task, resource_file)

        dimension._estimate_domain()

        # 2 resources -> 2**2 - 1 non-empty subsets
        assert dimension.estimated_domain_size == 3

    def test_distance_is_zero_for_the_same_resource_set(
        self, task, resource_file, plan_l1_then_l2
    ):
        dimension = ResourceUsedDimension(task, resource_file)
        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert dimension.distance(behaviour, behaviour) == 0.0

    def test_distance_is_the_jaccard_complement(self, task, resource_file):
        dimension = ResourceUsedDimension(task, resource_file)

        # {tr1} vs {tr1, tr2}: intersection 1, union 2 -> 1 - 1/2
        assert dimension.distance('ru:tr1', 'ru:tr1,tr2') == 0.5

    def test_disjoint_resource_sets_are_maximally_distant(self, task, resource_file):
        dimension = ResourceUsedDimension(task, resource_file)

        assert dimension.distance('ru:tr1', 'ru:tr2') == 1.0

    def test_two_empty_resource_sets_are_identical(self, task, resource_file):
        dimension = ResourceUsedDimension(task, resource_file)

        assert dimension.distance('ru:', 'ru:') == 0.0

    def test_token_is_matched_by_prefix_not_substring(self, task, resource_file):
        """'ru' occurs inside 'truck1'; the goal token must not shadow ours."""
        dimension = ResourceUsedDimension(task, resource_file)

        combined = 'go:at(truck1,l0)->delivered(l1) $$ ru:tr1'

        assert dimension.distance(combined, 'ru:tr1') == 0.0

    def test_distance_requires_its_own_token_to_be_present(self, task, resource_file):
        dimension = ResourceUsedDimension(task, resource_file)

        with pytest.raises(AssertionError, match='should be present'):
            dimension.distance('cb:4', 'ru:tr1')


class TestResourceCount:
    def test_behaviour_counts_each_resource_use(
        self, task, resource_file, plan_l1_then_l2
    ):
        dimension = ResourceCountDimension(task, resource_file)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        # tr1 appears in all four actions; tr2 in none.
        assert behaviour == 'rc:tr1=4,tr2=0'

    def test_counts_are_ordered_deterministically(
        self, task, resource_file, plan_l1_then_l2
    ):
        """Regression: counts were emitted in set-iteration order. addinfo['objects']
        is a set, and set order for strings varies with hash randomisation between
        processes, so the same plan yielded different behaviour strings from run to
        run -- behaviours could not be compared or stored across runs.

        Six objects are used rather than the fixture's two: unsorted order would
        land on sorted order by chance 1 time in 720, rather than 1 in 2.
        """
        dimension = ResourceCountDimension(task, resource_file)
        dimension.addinfo['objects'] = {'tr5', 'tr3', 'tr1', 'tr6', 'tr2', 'tr4'}

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert behaviour == 'rc:tr1=4,tr2=0,tr3=0,tr4=0,tr5=0,tr6=0'

    def test_counts_distinguish_plans_that_ru_conflates(
        self, task, resource_file, plan_l1_then_l2, plan_two_trucks
    ):
        dimension = ResourceCountDimension(task, resource_file)

        first = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))
        second = dimension.plan_behaviour(simulate(task, plan_two_trucks))

        assert first != second

    def test_estimated_domain_is_the_non_empty_subsets_of_resources(
        self, task, resource_file
    ):
        dimension = ResourceCountDimension(task, resource_file)

        dimension._estimate_domain()

        assert dimension.estimated_domain_size == 3

    def test_distance_is_not_implemented(self, task, resource_file):
        dimension = ResourceCountDimension(task, resource_file)

        with pytest.raises(AssertionError, match='not implemented'):
            dimension.distance('rc:tr1=4', 'rc:tr1=0')

    def test_behaviour_is_a_single_separator_free_token(
        self, task, resource_file, plan_l1_then_l2
    ):
        """Regression: counts were joined with ' $$ ', the same separator used between
        dimensions, and carried no 'rc:' prefix -- so a combined string could not be
        split back into one token per dimension."""
        dimension = ResourceCountDimension(task, resource_file)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert ' $$ ' not in behaviour
        assert behaviour.startswith('rc:')

    def test_token_survives_being_combined_with_other_dimensions(
        self, task, resource_file, plan_l1_then_l2
    ):
        counter = BehaviourDiversityCounter(
            task, [('go', None), ('rc', resource_file), ('ru', resource_file)]
        )

        behaviour = next(iter(counter.behaviours([plan_l1_then_l2])))

        # One token per dimension, recoverable by prefix.
        assert len(behaviour.split(' $$ ')) == 3
        assert sorted(t.split(':')[0] for t in behaviour.split(' $$ ')) == ['go', 'rc', 'ru']


class TestUtilityValue:
    def test_behaviour_sums_the_utilities_of_achieved_goals(
        self, task, utility_goals, plan_l1_then_l2
    ):
        dimension = UtilityValueDimension(task, utility_goals)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert behaviour == 'utility_value:8 -- delivered(l1)=5,delivered(l2)=3'

    def test_unachieved_goals_contribute_zero(
        self, task, domain, utility_goals, make_plan
    ):
        move, drop = domain['move'], domain['drop']
        tr1, l0, l1 = domain['tr1'], domain['l0'], domain['l1']
        only_l1 = make_plan((move, (tr1, l0, l1)), (drop, (tr1, l1)))
        dimension = UtilityValueDimension(task, utility_goals)

        behaviour = dimension.plan_behaviour(simulate(task, only_l1))

        assert behaviour == 'utility_value:5 -- delivered(l1)=5,delivered(l2)=0'

    def test_utility_is_credited_even_if_later_undone(
        self, task, domain, make_plan
    ):
        """The dimension asks whether a goal was *ever* true, not whether it holds
        at the end -- any(utils) over the whole state trace. at(tr1, l0) starts
        true and is retracted by the first move, yet still earns its utility."""
        at, move = domain['at'], domain['move']
        tr1, l0, l1 = domain['tr1'], domain['l0'], domain['l1']
        dimension = UtilityValueDimension(task, {'utility-goals': {at(tr1, l0): 7}})
        plan = make_plan((move, (tr1, l0, l1)))

        behaviour = dimension.plan_behaviour(simulate(task, plan))

        assert behaviour == 'utility_value:7 -- at(tr1, l0)=7'

    def test_estimated_domain_counts_distinct_achievable_sums(
        self, task, utility_goals
    ):
        dimension = UtilityValueDimension(task, utility_goals)

        dimension._estimate_domain()

        # non-empty subsets of {5, 3} produce sums 5, 3, 8
        assert dimension.estimated_domain_size == 3

    def test_distance_is_not_implemented(self, task, utility_goals):
        dimension = UtilityValueDimension(task, utility_goals)

        with pytest.raises(AssertionError, match='implemented by the child class'):
            dimension.distance('utility_value:8', 'utility_value:5')


class TestFunctions:
    def test_estimate_uses_the_declared_range(self, task, function_file):
        dimension = NumericFunctionDimension(task, function_file)

        dimension._estimate_domain()

        # range(0, 90, 10) -> 9 bins
        assert dimension.estimated_domain_size == 9

    def test_behaviour_is_the_index_of_the_final_value_bin(
        self, task, function_file, plan_l1_then_l2
    ):
        """Regression: min/max were swapped at parse time, which made the bin list
        empty and raised IndexError before any value could be binned."""
        dimension = NumericFunctionDimension(task, function_file)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        # 100 fuel - 2 moves * 10 = 80 -> bin index 8
        assert behaviour == 'fuel:8'

    def test_behaviour_matches_the_value_recorded_in_domain(
        self, task, function_file, plan_l1_then_l2
    ):
        """Regression: plan_behaviour returned ','.join(val) over an already-joined
        string, so 'fuel:8' came back as 'f,u,e,l,:,8' while domain kept 'fuel:8'."""
        dimension = NumericFunctionDimension(task, function_file)

        behaviour = dimension.plan_behaviour(simulate(task, plan_l1_then_l2))

        assert dimension.domain == {'fuel:8'}
        assert behaviour in dimension.domain

    def test_fuel_spent_changes_the_bin(self, task, domain, function_file, make_plan):
        move = domain['move']
        tr1, l0, l1, l2 = domain['tr1'], domain['l0'], domain['l1'], domain['l2']
        dimension = NumericFunctionDimension(task, function_file)
        one_move = make_plan((move, (tr1, l0, l1)))
        three_moves = make_plan(
            (move, (tr1, l0, l1)), (move, (tr1, l1, l2)), (move, (tr1, l2, l0))
        )

        # 90 fuel -> bin 8 (the top bin absorbs values above its range);
        # 70 fuel -> bin 7.
        assert dimension.plan_behaviour(simulate(task, one_move)) == 'fuel:8'
        assert dimension.plan_behaviour(simulate(task, three_moves)) == 'fuel:7'

    def test_dimension_is_usable_through_the_counter(
        self, task, function_file, plan_l1_then_l2
    ):
        """End to end: the fn dimension was unusable as shipped."""
        counter = BehaviourDiversityCounter(task, [('fn', function_file)])

        assert counter.bdc([plan_l1_then_l2]) == 1
        assert counter.behaviours([plan_l1_then_l2]) == {'fuel:8'}
