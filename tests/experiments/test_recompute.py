"""Ground rule 8, enforced: no number a task writes can fail to be recomputed
from its own result file plus the behaviour dump.

Every selection the sweep records carries the plans' indices, costs,
behaviours and action strings and the four indicator values the library
reported for the returned set. Here each is rebuilt from the dump alone --
no counter, no pool, no planner -- and compared. Under the stability model
the dump has no matrix and the stability distance is recomputed from the
action sets instead.
"""

import pytest

from bdc_experiments import cli, reference, runner
from bdc_experiments import report as rp
from bdc_experiments.config import load


@pytest.fixture(scope='module')
def swept(tmp_path_factory):
    root = tmp_path_factory.mktemp('recompute')
    assert cli.main(['run', 'smoke', 'select', '--results-dir', str(root)]) == 0
    return load('smoke', results_dir=root)


def test_every_recorded_selection_recomputes(swept):
    checked = 0
    for result in rp.selections(swept):
        dump = result['dump']
        by_index = {entry['index']: entry for entry in dump['plans']}
        assert len(dump['distinct']) == result['rows'][0]['b']
        for entry in result['extra']['selections']:
            for position, index in enumerate(entry['indices']):
                plan = by_index[index]
                assert entry['behaviours'][position] == plan['behaviour']
                assert entry['costs'][position] == plan['cost']
                assert entry['distinct'][position] == plan['distinct']
            rebuilt = rp.score(dump, entry['distinct'], entry['kappa'])
            for name, reported in entry['values'].items():
                assert rebuilt[name] == pytest.approx(reported, abs=1e-9), (
                    f"{result['task_id']} {entry['indicator']} kappa={entry['kappa']}: {name} "
                    f'recomputed as {rebuilt[name]} against the reported {reported}')
            row = next(r for r in result['rows']
                       if r['indicator'] == entry['indicator'] and r['kappa'] == entry['kappa'])
            assert all(row[name] == entry['values'][name] for name in runner.INDICATORS)
            checked += 1
    assert checked > 60, f'only {checked} selections: the smoke pools are not exercising this'


def test_the_stability_values_are_the_stability_distance_over_action_sets(swept):
    """The stability model's B-MaxSum is the sum of stability distances over the
    distinct action sets of the returned plans, read straight off the actions."""
    checked = 0
    for result in rp.selections(swept, keep=lambda name: name == 'stability'):
        for entry in result['extra']['selections']:
            sets = list(dict.fromkeys(frozenset(actions) for actions in entry['actions']))
            expected = sum(reference.ref_stability(a, b)
                           for i, a in enumerate(sets) for b in sets[i + 1:])
            assert entry['values']['bmaxsum'] == pytest.approx(expected, abs=1e-9), result['task_id']
            assert entry['values']['bcoverage'] == len(sets)
            checked += 1
    assert checked
