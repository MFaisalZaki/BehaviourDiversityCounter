"""The whole evaluation, end to end on the committed smoke pools.

No planner, no benchmark checkout: `configs/smoke.toml` sits beside four pools
and the PDDL they were generated from, and the run seeds itself from them.
"""

import json
from pathlib import Path

import pytest

from bdc_experiments import cli, report, runner
from bdc_experiments.config import load

#: What each report must write. The paper's subsections consume these names,
#: so a rename here is a broken \input there.
OUTPUTS = {
    'setup': ['benchmark.csv', 'models.csv', 'tables/setup_benchmark.tex',
              'tables/setup_models.tex', 'manifest.json'],
    'e1': ['e1_behaviours.csv', 'e1_selections.csv', 'e1_pairwise_diffs.csv', 'e1_stability.csv',
           'e1_note.md', 'manifest.json']
          + [f'tables/e1_{kind}_{domain}.tex' for domain in ('rovers', 'driverlog', 'satellite')
             for kind in ('behaviours', 'selections', 'pairwise')],
    'e2': ['e2_random_subsets.csv', 'e2_subsets.csv', 'e2_kendall.csv', 'e2_cross.csv',
           'e2_cross_macro.csv', 'e2_weights.csv', 'tables/e2_kendall.tex', 'tables/e2_cross.tex',
           'tables/e2_weights.tex', 'figures/e2_tau.pdf', 'manifest.json'],
    'e3': ['e3_timing.csv', 'e3_medians.csv', 'tables/e3_medians.tex',
           'figures/e3_mapping_vs_generation.pdf', 'figures/e3_selection_vs_b.pdf',
           'manifest.json'],
}


@pytest.fixture(scope='module')
def swept(tmp_path_factory):
    """The whole smoke sweep, run once for every test in this module."""
    root = tmp_path_factory.mktemp('smoke')
    for kind in runner.KINDS:
        assert cli.main(['run', 'smoke', kind, '--results-dir', str(root)]) == 0, kind
    assert cli.main(['report', 'smoke', 'all', '--results-dir', str(root)]) == 0
    return load('smoke', results_dir=root)


@pytest.mark.parametrize('name', sorted(OUTPUTS))
def test_every_declared_output_is_written(swept, name):
    directory = report.report_dir(swept, name)
    for output in OUTPUTS[name]:
        path = directory / output
        assert path.is_file(), f'{name} did not write {output}'
        assert path.stat().st_size > 0, f'{name} wrote an empty {output}'
        if output.endswith('.csv'):                   # a header at the very least
            assert path.read_text().splitlines(), output
        if output.endswith('.tex'):
            text = path.read_text()
            assert r'\toprule' in text and f'{{tab:{name}-' in text, output


@pytest.mark.parametrize('kind', sorted(runner.KINDS))
def test_no_task_failed(swept, kind):
    results = runner.load_results(swept, kind)
    assert results, f'{kind} produced no results at all'
    failed = [r['task_id'] + ': ' + r['error']['message'] for r in results if r.get('error')]
    assert not failed, '\n'.join(failed)
    for result in results:
        skipped = (result.get('extra') or {}).get('skipped')
        if skipped:
            assert isinstance(skipped, str) and len(skipped) > 10, result['task_id']
            assert not result['rows'], f'{result["task_id"]} was skipped but reported rows'


def test_every_model_ran_on_every_pool_it_claims(swept):
    names = {r['task_id'].rsplit('/', 1)[1] for r in runner.load_results(swept, 'select')}
    for name in ('generic', 'stability', 'rovers_astronaut', 'driverlog_dispatcher',
                 'satellite_operator', 'rovers_astronaut-w0.25-0.75'):
        assert name in names, name


@pytest.mark.parametrize('name', sorted(OUTPUTS))
def test_the_manifest_names_the_code_and_the_data(swept, name):
    manifest = json.loads((report.report_dir(swept, name) / 'manifest.json').read_text())
    assert manifest['tasks']['failed'] == 0
    assert manifest['config']['hash'] == swept['meta']['config_hash']
    assert manifest['git']['revision'] and manifest['tie_breaking']
    assert manifest['packages']['unified-planning']


def test_the_checks_pass_on_the_smoke_pools(swept):
    """B-Coverage is constant on every random equal-count subset (E2), and the
    weights never move a B-Coverage selection (E2)."""
    e2 = json.loads((report.report_dir(swept, 'e2') / 'manifest.json').read_text())['extra']
    assert e2['bcoverage_constant_check'] == 'PASS' and e2['cells_checked'] > 0
    assert e2['bcoverage_selects_same_set_under_every_weight'] == 1.0


def test_the_case_study_reads_one_instance_per_domain(swept):
    e1 = json.loads((report.report_dir(swept, 'e1') / 'manifest.json').read_text())['extra']
    assert set(e1['chosen']) == {'rovers', 'driverlog', 'satellite'}
    assert all(c['b'] >= swept['e1']['min_behaviours'] for c in e1['chosen'].values())


def test_the_stability_model_counts_action_sets(swept):
    """Under the stability model a behaviour is a distinct action set, so b
    never exceeds the pool size and its dump carries no matrix."""
    for result in runner.load_results(swept, 'select'):
        if result['model']['name'] != 'stability':
            continue
        dump = runner.load_dump(swept, result)
        assert dump['matrix'] is None
        assert 1 <= len(dump['distinct']) <= result['pool']['size']
        assert all(len(t) == 1 for t in dump['distinct'])


def test_the_reports_rebuild_from_the_dumps_alone(swept):
    """`bdcexp report` is a pure function of results/ and behaviours/.

    Everything but the manifest, which records when it was written and from
    which revision, comes back byte for byte.
    """
    before = {}
    for name in OUTPUTS:
        for path in sorted(report.report_dir(swept, name).rglob('*')):
            if path.is_file() and path.name != 'manifest.json':
                before[str(path)] = path.read_bytes()
    assert cli.main(['report', 'smoke', 'all', '--results-dir', swept['run']['results_dir']]) == 0
    changed = [name for name, blob in before.items() if Path(name).read_bytes() != blob]
    assert not changed, 'a rebuild changed:\n' + '\n'.join(changed)


def test_the_setup_report_covers_every_pool(swept):
    rows = (report.report_dir(swept, 'setup') / 'benchmark.csv').read_text().splitlines()
    assert len(rows) == 4, rows          # a header and the three smoke domains
    models = (report.report_dir(swept, 'setup') / 'models.csv').read_text()
    for name in ('generic', 'stability', 'rovers_astronaut', 'driverlog_dispatcher', 'satellite_operator'):
        assert name in models, name
