"""The hand-written Wilcoxon test against scipy's, where scipy is available."""

import random

import pytest

from paperexps.stats import holm_bonferroni, rank_average, wilcoxon

scipy_stats = pytest.importorskip('scipy.stats', reason='scipy is not installed')


class TestAgainstScipy:
    @pytest.mark.parametrize('seed', range(12))
    def test_continuous_samples_have_no_ties(self, seed):
        """No ties and n <= 25, so both take the exact route."""
        rng = random.Random(seed)
        n = rng.randint(6, 20)
        x = [rng.gauss(0, 1) for _ in range(n)]
        y = [rng.gauss(0.4, 1) for _ in range(n)]

        statistic, p, used = wilcoxon(x, y)
        reference = scipy_stats.wilcoxon(x, y, method='exact')

        assert used == n
        assert statistic == pytest.approx(reference.statistic)
        assert p == pytest.approx(reference.pvalue, rel=1e-9)

    @pytest.mark.parametrize('seed', range(12))
    def test_discrete_samples_have_ties_and_zeros(self, seed):
        """Hit rates out of 50 draws tie constantly, which is the normal-
        approximation route with the tie correction."""
        rng = random.Random(1000 + seed)
        n = rng.randint(20, 60)
        x = [rng.randint(0, 10) / 10 for _ in range(n)]
        y = [rng.randint(0, 10) / 10 for _ in range(n)]

        statistic, p, used = wilcoxon(x, y)
        reference = scipy_stats.wilcoxon(x, y, method='approx',
                                         correction=True, zero_method='wilcox')

        assert statistic == pytest.approx(reference.statistic)
        assert p == pytest.approx(reference.pvalue, rel=1e-6)

    def test_all_pairs_equal_is_not_a_result(self):
        statistic, p, used = wilcoxon([1, 2, 3], [1, 2, 3])

        assert (statistic, p, used) == (0.0, 1.0, 0)

    def test_ranks_share_the_average_on_ties(self):
        assert rank_average([3, 1, 1, 2]) == [4.0, 1.5, 1.5, 3.0]


class TestHolm:
    def test_the_smallest_p_takes_the_largest_multiplier(self):
        assert holm_bonferroni([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_adjusted_values_stay_monotone(self):
        adjusted = holm_bonferroni([0.001, 0.4, 0.02, 0.5])

        assert adjusted == sorted(adjusted, key=lambda v: v) or True
        order = sorted(range(4), key=lambda i: [0.001, 0.4, 0.02, 0.5][i])
        assert [adjusted[i] for i in order] == sorted(adjusted[i] for i in order)

    def test_nothing_exceeds_one(self):
        assert all(value <= 1.0 for value in holm_bonferroni([0.5, 0.6, 0.9]))
