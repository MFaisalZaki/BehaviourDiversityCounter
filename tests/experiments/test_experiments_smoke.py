"""Every experiment, end to end on the committed smoke pools.

No planner, no benchmark checkout: `configs/smoke.toml` sits beside four pools
and the PDDL they were generated from, and the run seeds itself from them.
"""

import json
from pathlib import Path

import pytest

from bdc_experiments import cli, report, runner
from bdc_experiments.config import load

#: What each experiment's report must write. The paper's subsections consume
#: these names, so a rename here is a broken \input there.
OUTPUTS = {
    'e1': ['e1_behaviours.csv', 'e1_selections.csv', 'e1_pairwise_diffs.csv', 'e1_note.md',
           'tables/e1_behaviours.tex', 'tables/e1_selections.tex', 'manifest.json'],
    'e2': ['e2_random_subsets.csv', 'e2_kendall.csv', 'e2_cross.csv', 'e2_cross_macro.csv',
           'e2_cross_all_pools.csv', 'tables/e2_kendall.tex', 'tables/e2_cross.tex',
           'figures/e2_tau.pdf', 'manifest.json'],
    'e3': ['e3_ratios.csv', 'e3_worst_cases.csv', 'e3_summary.csv', 'e3_checks.json',
           'tables/e3_ratios.tex', 'figures/e3_ratios.pdf', 'manifest.json'],
    'e4': ['e4_prefix_values.csv', 'e4_summary.csv', 'tables/e4_summary.tex',
           'figures/e4_prefixes.pdf', 'manifest.json'],
    'e5': ['e5_resolution.csv', 'e5_summary.csv', 'tables/e5_resolution.tex',
           'figures/e5_cap_vs_b.pdf', 'figures/e5_time_vs_b.pdf', 'manifest.json'],
    'e6': ['e6_timing.csv', 'e6_medians.csv', 'tables/e6_medians.tex',
           'figures/e6_mapping_vs_generation.pdf', 'figures/e6_selection_vs_b.pdf',
           'manifest.json'],
}


@pytest.fixture(scope='module')
def swept(tmp_path_factory):
    """The whole smoke sweep, run once for every test in this module."""
    root = tmp_path_factory.mktemp('smoke')
    for experiment in OUTPUTS:
        assert cli.main(['run', 'smoke', experiment, '--results-dir', str(root)]) == 0, experiment
        assert cli.main(['report', 'smoke', experiment, '--results-dir', str(root)]) == 0, experiment
    assert cli.main(['report', 'smoke', 'setup', '--results-dir', str(root)]) == 0
    return load('smoke', results_dir=root)


@pytest.mark.parametrize('experiment', sorted(OUTPUTS))
def test_every_declared_output_is_written(swept, experiment):
    directory = report.report_dir(swept, experiment)
    for name in OUTPUTS[experiment]:
        path = directory / name
        assert path.is_file(), f'{experiment} did not write {name}'
        assert path.stat().st_size > 0, f'{experiment} wrote an empty {name}'
        if name.endswith('.csv'):                     # a header at the very least
            assert path.read_text().splitlines(), name
        if name.endswith('.tex'):
            text = path.read_text()
            assert r'\toprule' in text and f'{{tab:{experiment}-' in text, name


@pytest.mark.parametrize('experiment', sorted(OUTPUTS))
def test_no_task_failed_and_the_manifest_says_so(swept, experiment):
    results = runner.load_results(swept, experiment)
    assert results, f'{experiment} produced no results at all'
    failed = [r['task_id'] + ': ' + r['error']['message'] for r in results if r.get('error')]
    assert not failed, '\n'.join(failed)
    manifest = json.loads((report.report_dir(swept, experiment) / 'manifest.json').read_text())
    assert manifest['tasks']['total'] == len(results)
    assert manifest['tasks']['failed'] == 0
    assert manifest['config']['hash'] == swept['meta']['config_hash']
    assert manifest['git']['revision'] and manifest['tie_breaking']
    assert manifest['packages']['unified-planning']


@pytest.mark.parametrize('experiment', sorted(OUTPUTS))
def test_a_skipped_task_says_why_and_reports_nothing(swept, experiment):
    for result in runner.load_results(swept, experiment):
        skipped = (result.get('extra') or {}).get('skipped')
        if skipped:
            assert isinstance(skipped, str) and len(skipped) > 10, result['task_id']
            assert not result['rows'], f'{result["task_id"]} was skipped but reported rows'


def test_the_e3_checks_pass_on_the_smoke_pools(swept):
    """Every B-Coverage greedy ratio is 1: a violation is a blocking issue."""
    checks = json.loads((report.report_dir(swept, 'e3') / 'e3_checks.json').read_text())
    assert checks['passed'], checks.get('violations')
    assert checks['cases'] > 0, 'the check passed vacuously: it examined no case'


def test_the_reports_rebuild_from_the_dumps_alone(swept):
    """`bdcexp report` is a pure function of results/ and behaviours/.

    Everything but the manifest, which records when it was written and from
    which revision, comes back byte for byte.
    """
    before = {}
    for experiment in OUTPUTS:
        directory = report.report_dir(swept, experiment)
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.name != 'manifest.json':
                before[str(path)] = path.read_bytes()
    for experiment in OUTPUTS:
        assert cli.main(['report', 'smoke', experiment,
                         '--results-dir', swept['run']['results_dir']]) == 0
    changed = [name for name, blob in before.items() if Path(name).read_bytes() != blob]
    assert not changed, 'a rebuild changed:\n' + '\n'.join(changed)


def test_the_setup_report_covers_every_pool(swept):
    rows = (report.report_dir(swept, 'setup') / 'benchmark.csv').read_text().splitlines()
    assert len(rows) == 4, rows          # a header and the three smoke domains
    models = (report.report_dir(swept, 'setup') / 'models.csv').read_text()
    for name in ('generic', 'rovers_astronaut', 'driverlog_dispatcher', 'satellite_operator'):
        assert name in models, name
