"""The evaluation harness: the pieces that decide what the numbers mean.

The statistics are checked against scipy where it is installed; the plan-level
baselines, the two new dimensions and the model builder against hand-worked
values on the fixture task; and the pool matcher against file names.
"""

import math
import random
from types import SimpleNamespace

import pytest

import baseline
import stats
from harness import behaviour_set, jaccard
from model import default_dimensions, rovers_dimensions, with_weights
from utils import POOL_RE, filter_tasks, generator_name, plan_key

from behaviour_diversity_counter import BehaviourDiversityCounter
from behaviour_diversity_counter.dimensions.cost_bin import CostBinDimension
from behaviour_diversity_counter.dimensions.goal_predicate_ordering import GoalPredicatesOrderingDimension
from behaviour_diversity_counter.dimensions.resources import ResourceNumberDimension
from behaviour_diversity_counter.simulation import simulate


scipy_stats = None
try:
    import scipy.stats as scipy_stats
except Exception:                                                       # noqa: BLE001
    pass


# ---------------------------------------------------------------- statistics

class TestWilcoxon:
    @pytest.mark.skipif(scipy_stats is None, reason='scipy is not installed')
    @pytest.mark.parametrize('seed', range(8))
    def test_exact_route_agrees_with_scipy(self, seed):
        rng = random.Random(seed)
        n = rng.randint(6, 20)
        x = [rng.gauss(0, 1) for _ in range(n)]
        y = [rng.gauss(0.4, 1) for _ in range(n)]

        ours = stats.wilcoxon(x, y)
        theirs = scipy_stats.wilcoxon(x, y, method='exact')

        assert ours['method'] == 'exact'
        assert ours['p'] == pytest.approx(theirs.pvalue, abs=1e-9)

    @pytest.mark.skipif(scipy_stats is None, reason='scipy is not installed')
    @pytest.mark.parametrize('seed', range(4))
    def test_approximate_route_agrees_with_scipy(self, seed):
        rng = random.Random(seed)
        n = 40
        x = [rng.randint(0, 6) for _ in range(n)]
        y = [rng.randint(0, 6) for _ in range(n)]

        ours = stats.wilcoxon(x, y)
        theirs = scipy_stats.wilcoxon(x, y, method='approx', correction=True, zero_method='wilcox')

        assert ours['method'] == 'approx'
        assert ours['p'] == pytest.approx(theirs.pvalue, abs=1e-6)

    def test_identical_samples_have_no_test(self):
        assert stats.wilcoxon([1, 2, 3], [1, 2, 3])['p'] is None

    def test_median_and_iqr_of_the_differences(self):
        result = stats.wilcoxon([5, 6, 7, 8], [1, 1, 1, 1])

        assert result['median_diff'] == 5.5
        assert result['iqr_diff'] == pytest.approx(1.5)


class TestHolm:
    def test_worked_example(self):
        assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_none_entries_are_ignored(self):
        assert stats.holm([0.02, None]) == [0.02, None]


class TestKendall:
    @pytest.mark.skipif(scipy_stats is None, reason='scipy is not installed')
    @pytest.mark.parametrize('seed', range(5))
    def test_tau_b_agrees_with_scipy(self, seed):
        rng = random.Random(seed)
        x = [rng.randint(0, 5) for _ in range(30)]
        y = [rng.randint(0, 5) for _ in range(30)]

        assert stats.kendall_tau(x, y) == pytest.approx(scipy_stats.kendalltau(x, y).statistic, abs=1e-9)

    def test_perfect_agreement_and_reversal(self):
        assert stats.kendall_tau([1, 2, 3], [4, 5, 6]) == 1.0
        assert stats.kendall_tau([1, 2, 3], [6, 5, 4]) == -1.0
        assert stats.kendall_tau([1, 1, 1], [1, 2, 3]) is None


class TestQuantiles:
    def test_median_and_quantile(self):
        assert stats.median([3, 1, 2]) == 2
        assert stats.median([4, 1, 2, 3]) == 2.5
        assert stats.quantile([1, 2, 3, 4], 0.25) == 1.75


# ----------------------------------------------------------- plan baselines

class TestPlanMetrics:
    def test_stability_is_the_jaccard_distance_of_the_action_sets(
        self, task, plan_l1_then_l2, plan_two_trucks
    ):
        metric = baseline.PlanMetric('stability', [plan_l1_then_l2, plan_two_trucks])

        # Shared: move(tr1,l0,l1), drop(tr1,l1). Union: 6 actions.
        assert metric.distance(0, 1) == pytest.approx(1 - 2 / 6)
        assert metric.distance(0, 0) == 0.0

    def test_uniqueness_is_all_or_nothing(self, task, plan_l1_then_l2, plan_l2_then_l1):
        metric = baseline.PlanMetric('uniqueness', [plan_l1_then_l2, plan_l2_then_l1, plan_l1_then_l2])

        assert metric.distance(0, 1) == 1.0
        assert metric.distance(0, 2) == 0.0

    def test_state_distance_reads_the_visited_states(self, task, plan_l1_then_l2, plan_l2_then_l1):
        trace = {id(plan): simulate(task, plan) for plan in (plan_l1_then_l2, plan_l2_then_l1)}
        metric = baseline.PlanMetric('state', [plan_l1_then_l2, plan_l2_then_l1], trace=trace)

        # Five states each; only the initial state and the final state (both
        # delivered, truck positions differ: l2 vs l1) -- so only the initial
        # state is shared: 1 of 9.
        assert len(metric.features[0]) == 5
        assert metric.distance(0, 1) == pytest.approx(1 - 1 / 9)
        assert metric.distance(0, 0) == 0.0

    def test_state_metric_needs_a_trace(self, task, plan_l1_then_l2):
        with pytest.raises(ValueError):
            baseline.PlanMetric('state', [plan_l1_then_l2])

    def test_plan_set_score_is_the_mean_pairwise_distance(self):
        import numpy as np
        matrix = np.array([[0, 1, 0.5], [1, 0, 0.25], [0.5, 0.25, 0]])

        assert baseline.plan_set_score(matrix, [0, 1, 2]) == pytest.approx((1 + 0.5 + 0.25) / 3)
        assert baseline.plan_set_score(matrix, [0]) == 0.0

    def test_greedy_opens_on_the_farthest_pair_then_adds_the_best_gain(self):
        import numpy as np
        matrix = np.array([[0, 0.2, 0.9, 0.5], [0.2, 0, 0.3, 0.1], [0.9, 0.3, 0, 0.4], [0.5, 0.1, 0.4, 0]])

        assert baseline.greedy_select(matrix, 3) == [0, 2, 3]
        assert baseline.greedy_select(matrix, 1) == [0]
        assert baseline.greedy_select(matrix, 9) == [0, 2, 3, 1]


# ------------------------------------------------------------ new dimensions

class TestCostBin:
    def test_bins_over_the_quality_range(self, task):
        dimension = CostBinDimension(task, {'optimal-cost': 10, 'q': 2.0, 'width': 0.1})

        assert dimension.bins == 10
        assert [dimension.bin_of(c) for c in (10, 11, 15, 19, 20, 25, 3)] == [0, 1, 5, 9, 9, 9, 0]

    def test_distance_respects_the_bin_order(self, task):
        dimension = CostBinDimension(task, {'optimal-cost': 10, 'q': 2.0, 'width': 0.1})

        assert dimension.distance('cbin:0', 'cbin:9') == 1.0
        assert dimension.distance('cbin:3', 'cbin:4') == pytest.approx(1 / 9)
        assert dimension.distance('cbin:3', 'cbin:3') == 0.0

    def test_q_one_is_a_constant_feature(self, task, plan_l1_then_l2):
        dimension = CostBinDimension(task, {'optimal-cost': 4, 'q': 1.0})

        assert dimension.bins == 1
        assert dimension.plan_behaviour(plan_l1_then_l2) == 'cbin:0'
        assert dimension.distance('cbin:0', 'cbin:0') == 0.0


class TestResourceNumber:
    def test_counts_the_resources_used_with_a_discrete_distance(
        self, task, resource_file, plan_l1_then_l2, plan_two_trucks
    ):
        from tests.test_dimensions import simulate as attach
        dimension = ResourceNumberDimension(task, resource_file)

        one = dimension.plan_behaviour(attach(task, plan_l1_then_l2))
        two = dimension.plan_behaviour(attach(task, plan_two_trucks))

        assert (one, two) == ('rn:1', 'rn:2')
        assert dimension.distance(one, two) == 1.0
        assert dimension.distance(one, one) == 0.0


class TestGoalCap:
    def test_the_cap_keeps_the_first_atoms_of_the_canonical_order(self, task):
        capped = GoalPredicatesOrderingDimension(task, {'max-goals': 1})
        full = GoalPredicatesOrderingDimension(task)

        assert full.goal_atoms == 2 and full.max_goals is None and len(full.vars) == 2
        assert capped.max_goals == 1 and len(capped.vars) == 1
        assert capped.vars[0] == full.vars[0]

    def test_the_canonical_order_is_by_expression_identity(self, task):
        dimension = GoalPredicatesOrderingDimension(task)

        assert [v.node_id for v in dimension.vars] == sorted(v.node_id for v in dimension.vars)


# ------------------------------------------------------------------- model

def fake_pool(resources=None, optimal_cost=4, q=2.0):
    return SimpleNamespace(optimal_cost=optimal_cost, info={'q': q, 'resources-file': resources})


class TestModel:
    def test_default_model_is_ordering_plus_cost_bin_plus_agents_where_declared(self):
        params = {'max-goals': 8, 'cost-bin-width': 0.1}

        assert [k for k, _ in default_dimensions(fake_pool(), params)] == ['go', 'cbin']
        assert [k for k, _ in default_dimensions(fake_pool('r.txt'), params)] == ['go', 'cbin', 'ru']
        assert default_dimensions(fake_pool(), params)[0][1] == {'max-goals': 8}
        assert default_dimensions(fake_pool(), params, bin_width=0.25)[1][1]['width'] == 0.25

    def test_rovers_model_is_the_running_example(self):
        assert [k for k, _ in rovers_dimensions(fake_pool('r.txt'), {'max-goals': None})] == ['rn', 'go']
        with pytest.raises(ValueError):
            rovers_dimensions(fake_pool(), {})

    def test_weights_are_declared_on_every_dimension(self):
        dims = with_weights([('go', {}), ('cbin', {'q': 2.0})], [0.25, 0.75])

        assert dims == [('go', {'weight': 0.25}), ('cbin', {'q': 2.0, 'weight': 0.75})]
        with pytest.raises(ValueError):
            with_weights([('go', {})], [0.5, 0.5])


# ------------------------------------------------------------------- utils

class TestPoolNames:
    def test_the_generator_tag_is_read_off_the_file_name(self):
        match = POOL_RE.match('1.0-100-classical-2006-rovers-15-fi-bc-results.json').groupdict()

        assert match['domain'] == 'rovers' and match['inst'] == '15' and match['generator'] == 'fi-bc'
        assert generator_name('fi-bc') == 'forbid-iterative'
        assert POOL_RE.match('2.0-10-classical-2008-parcprinter-1-topk-results.json').groupdict()['generator'] == 'topk'
        assert POOL_RE.match('1.0-10-classical-2004-pipesworld-tankage-17-fi-bc-results.json').groupdict()['domain'] == 'pipesworld-tankage'

    def test_plan_key_ignores_comments_and_spacing(self):
        assert plan_key('; cost 3\n(move a  b)\n\n(drop b)') == plan_key('(move a b)\n(drop b)\n; behaviour x')

    def test_filter_tasks(self):
        tasks = [{'domain': 'rovers', 'generator': 'fi-bc', 'q': 1.0, 'k': 10},
                 {'domain': 'gripper', 'generator': 'topk', 'q': 2.0, 'k': 10}]

        assert filter_tasks(tasks, {}) == tasks
        assert filter_tasks(tasks, {'domains': ['rovers']}) == tasks[:1]
        assert filter_tasks(tasks, {'generators': ['topk']}) == tasks[1:]
        assert filter_tasks(tasks, {'pool-q-values': [2.0]}) == tasks[1:]
        assert filter_tasks(tasks, {'generators': []}) == tasks


class TestSmallHelpers:
    def test_jaccard(self):
        assert jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
        assert jaccard(set(), set()) == 1.0

    def test_behaviour_set(self):
        plans = [SimpleNamespace(behaviour='b'), SimpleNamespace(behaviour='a'), SimpleNamespace(behaviour='b')]
        assert behaviour_set(plans) == ['a', 'b']
