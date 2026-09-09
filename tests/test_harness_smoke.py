"""End to end on one real Rovers pool, when the sandbox is on disk.

The sandbox is what ../paper-experiments/setup_benchmark.sh builds; point
``BDC_SANDBOX`` at another copy to run these elsewhere.  Without one the tests
skip, since the benchmark checkout is not part of the repository.
"""

import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.environ.get('BDC_SANDBOX') or os.path.join(HERE, '..', 'paper-experiments', 'paperexps', 'sandbox')

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(SANDBOX, 'fi-generated-plans-dir')), reason='no sandbox on disk')


@pytest.fixture(scope='module')
def rovers_task():
    from utils import match_plans_with_problems
    tasks = match_plans_with_problems(os.path.join(SANDBOX, 'fi-generated-plans-dir'),
                                      os.path.join(SANDBOX, 'classical-domains'),
                                      os.path.join(SANDBOX, 'ru-info-dir'))
    from utils import pool_plan_count
    candidates = [t for t in tasks if t['domain'] == 'rovers' and os.path.getsize(t['pool_file']) > 48
                  and pool_plan_count(t['pool_file']) > 0]
    assert candidates, 'no non-empty Rovers pool in the sandbox'
    # The smallest pool that holds plans keeps the smoke test quick.
    return min(candidates, key=lambda t: os.path.getsize(t['pool_file']))


@pytest.fixture(scope='module')
def params():
    return {'name': 'smoke', 'k-values': [5, 10], 'k-nn': 1, 'max-goals': 8, 'cost-bin-width': 0.1,
            'seed': 1, 'random-subsets': 20, 'b-range': [2, 20], 'pool-cap': 1000, 'pool-cap-factor': 10,
            'pool-sizes': [100], 'repeats': 1, 'k': 3, 'min-rovers': 1, 'min-goals': 1, 'min-distinct': 1,
            'default-weight': 0.5, 'weights': [0.25], 'k-nn-values': [2], 'bin-widths': [0.25]}


def test_pool_loads_sorted_by_cost(rovers_task):
    from harness import load_pool
    pool = load_pool(rovers_task)

    assert pool.plans
    assert [p.cost for p in pool.plans] == sorted(p.cost for p in pool.plans)
    assert pool.optimal_cost == pool.plans[0].cost
    assert all(id(p) in pool.trace for p in pool.plans)


@pytest.mark.parametrize('module_name', ['exp_e1_metric_vs_feature', 'exp_e2_cross_eval',
                                         'exp_e3_greedy_vs_optimal', 'exp_e4_generators',
                                         'exp_e5_runtime'])
def test_every_experiment_runs_and_reports(module_name, rovers_task, params, tmp_path):
    module = __import__(module_name)
    from harness import run_tasks
    p = {**params, 'name': module.NAME}
    if module_name == 'exp_e5_runtime':
        p['k-values'] = [5]
    dump = tmp_path / 'dump'
    dump.mkdir()
    run_tasks([rovers_task], str(dump), module.run_task, p)
    (result,) = [json.load(open(dump / f)) for f in os.listdir(dump)]
    assert result['error'] is None, result['error']
    assert result['pool']['size'] > 0

    from harness import load_results
    paths = {k: str(tmp_path / k) for k in ('results', 'tables', 'figures')}
    outputs, notes = module.report(load_results(str(dump)), p, paths, 'config.json', 'now')
    assert all(os.path.exists(path) for path in outputs)


def test_case_study_and_sensitivity_run(rovers_task, params, tmp_path):
    import exp_e0_case_study
    import exp_e6_sensitivity
    from harness import load_results, run_tasks
    for module in (exp_e0_case_study, exp_e6_sensitivity):
        p = {**params, 'name': module.NAME, 'k': 3 if module is exp_e0_case_study else 5}
        dump = tmp_path / module.NAME
        dump.mkdir()
        run_tasks([rovers_task], str(dump), module.run_task, p)
        (result,) = load_results(str(dump))
        assert result['error'] is None, result['error']
        paths = {k: str(tmp_path / module.NAME / k) for k in ('results', 'tables', 'figures')}
        outputs, _ = module.report([result], p, paths, 'config.json', 'now')
        assert all(os.path.exists(path) for path in outputs)
