"""The report helpers: the statistics, the CSV and the LaTeX escaping."""

import csv

import pytest

from bdc_experiments import report


class TestHolm:
    """Holm-Bonferroni against values worked out by hand."""

    def test_known_values(self):
        # sorted: 0.01 * 3 = 0.03, 0.03 * 2 = 0.06, 0.04 * 1 = 0.04 -> 0.06 by
        # the running maximum, and the input order is preserved.
        assert report.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])

    def test_single_p_value_is_unchanged(self):
        assert report.holm([0.02]) == pytest.approx([0.02])

    def test_is_monotone_and_capped_at_one(self):
        adjusted = report.holm([0.2, 0.3, 0.9])
        assert adjusted == pytest.approx([0.6, 0.6, 0.9])
        assert report.holm([0.5, 0.6, 0.7]) == pytest.approx([1.0, 1.0, 1.0])

    def test_never_falls_below_the_raw_p_value(self):
        raw = [0.001, 0.01, 0.02, 0.04, 0.5]
        assert all(a >= p for a, p in zip(report.holm(raw), raw))


class TestStatistics:
    def test_summarise_reports_n_median_and_iqr(self):
        summary = report.summarise([1.0, 2.0, 3.0, 4.0])
        assert summary['n'] == 4 and summary['median'] == pytest.approx(2.5)
        assert summary['q1'] < summary['median'] < summary['q3']

    def test_summarise_of_nothing_is_missing_not_zero(self):
        assert report.summarise([])['median'] is None
        assert report.summarise([None, None])['n'] == 0

    def test_macro_is_the_mean_of_the_per_domain_means(self):
        rows = [{'domain': 'a', 'v': 1.0}, {'domain': 'a', 'v': 3.0}, {'domain': 'b', 'v': 10.0}]
        assert report.pooled(rows, 'v') == pytest.approx(14 / 3)
        assert report.macro(rows, 'v') == pytest.approx((2.0 + 10.0) / 2)

    def test_wilcoxon_of_identical_samples_is_missing(self):
        assert report.wilcoxon([1, 2, 3], [1, 2, 3]) == (None, None)

    def test_kendall_of_a_constant_is_missing(self):
        assert report.kendall([1, 1, 1, 1], [1, 2, 3, 4])[0] is None


class TestOutput:
    def test_csv_is_not_rounded_and_missing_stays_empty(self, tmp_path):
        path = report.write_csv(tmp_path / 'x.csv',
                                [{'a': 1 / 3, 'b': None}], ['a', 'b'])
        row = next(iter(csv.DictReader(path.open())))
        assert row['a'] == repr(1 / 3).strip("'") or float(row['a']) == 1 / 3
        assert len(row['a']) > 10 and row['b'] == ''

    def test_latex_rounds_and_escapes(self, tmp_path):
        path = report.table(tmp_path / 't.tex', 'tab:e0-x', 'A caption.',
                            ['name', 'value'], [['b_maxmin', 1 / 3], ['none', None]])
        text = path.read_text()
        assert r'\label{tab:e0-x}' in text and r'b\_maxmin' in text
        assert '0.333' in text and '--' in text
        assert r'\toprule' in text and r'\bottomrule' in text

    def test_escape_covers_the_characters_that_break_latex(self):
        assert report.escape('|a_b%c&d#e') == r'\textbar{}a\_b\%c\&d\#e'
