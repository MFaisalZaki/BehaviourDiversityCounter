"""Pool generation with a stub planner standing in for forbid-iterative.

The stub answers the same command line as ``python -m forbiditerative.plan``
and leaves Fast Downward plan files in ``found_plans/done/``, which is what
the generator reads; the real planner is exercised only on the cluster.
"""

import json
import os
import sys
import textwrap

import pytest

import gen_pools

STUB = textwrap.dedent('''
    import argparse, os, sys
    p = argparse.ArgumentParser()
    for flag in ('--planner', '--domain', '--problem', '--number-of-plans', '--quality-bound',
                 '--overall-time-limit', '--overall-memory-limit', '--results-file'):
        p.add_argument(flag)
    for flag in ('--symmetries', '--suppress-planners-output', '--plans-as-json'):
        p.add_argument(flag, action='store_true')
    a = p.parse_args()
    os.makedirs('found_plans/done')
    n = 1 if a.planner == 'topk' and a.number_of_plans == '1' else 4
    costs = [4, 4, 9, 5][:n]                      # one optimal, one duplicate cost, one over 2*c*
    for i, cost in enumerate(costs, start=1):
        with open(f'found_plans/done/sas_plan.{i}', 'w') as f:
            f.write('(move tr1 l0 l1)\\n' * (cost - 1) + '(drop tr1 l1)\\n; cost = %d (unit cost)\\n' % cost)
    print('stub planner:', a.planner)
''')


@pytest.fixture
def benchmark(tmp_path):
    root = tmp_path / 'classical-domains' / 'classical' / 'toy'
    root.mkdir(parents=True)
    (root / 'api.py').write_text("domains = [{'name': 'toy', 'ipc': '2099', 'problems': "
                                 "[('toy/domain.pddl', 'toy/p02.pddl'), ('toy/domain.pddl', 'toy/p01.pddl')]}]\n")
    (root / 'domain.pddl').write_text('(define (domain toy))')
    (root / 'p01.pddl').write_text('(define (problem p01))')
    (root / 'p02.pddl').write_text('(define (problem p02))')
    return str(tmp_path / 'classical-domains')


@pytest.fixture
def params(tmp_path, benchmark):
    stub = tmp_path / 'stub_planner.py'
    stub.write_text(STUB)
    return {'name': 'GEN-test', 'kind': 'generate', 'generator': 'fi', 'tag': 'fi-bc', 'q': 2.0,
            'pool-k-values': [2], 'pool-factor': 10, 'pool-cap': 1000, 'domains': ['toy'],
            'fi-command': [sys.executable, str(stub)], 'plansdir': str(tmp_path / 'pools'),
            'dump-dir': str(tmp_path / 'pools'), 'benchmark': benchmark, 'ru-info': str(tmp_path / 'ru'),
            'time-limit': '30m', 'memory-limit': '8G'}


class TestListTasks:
    def test_one_task_per_instance_and_k_in_pool_file_layout(self, params):
        tasks = gen_pools.list_tasks(params)

        assert [t['task_id'] for t in tasks] == ['2.0-2-classical-2099-toy-1-fi-bc', '2.0-2-classical-2099-toy-2-fi-bc']
        # Instances are numbered through the problems sorted by path, as the matcher counts them.
        assert tasks[0]['problem_file'].endswith('p01.pddl') and tasks[1]['problem_file'].endswith('p02.pddl')
        assert tasks[0]['dumpfile'] == '2.0-2-classical-2099-toy-1-fi-bc-results.json'

    def test_the_optimal_track_directory_is_preferred(self):
        assert gen_pools._pick_directory(['x-sat11-strips', 'x-opt11-strips']) == 'x-opt11-strips'
        assert gen_pools._pick_directory(['b', 'a']) == 'a'


class TestPlannerCommand:
    def test_fi_and_topq_flags(self, params):
        fi = gen_pools.planner_command(params, 'diverse', 'd.pddl', 'p.pddl', number_of_plans=20)
        topq = gen_pools.planner_command(params, 'topq_via_topk', 'd.pddl', 'p.pddl', quality_bound=2.0)

        assert fi[2:4] == ['--planner', 'diverse'] and '--number-of-plans' in fi and '20' in fi
        assert '--quality-bound' not in fi and '--symmetries' not in fi
        assert '--quality-bound' in topq and '2.0' in topq and '--symmetries' in topq
        assert fi[fi.index('--overall-time-limit') + 1] == '30m'


class TestRunTask:
    def test_pool_is_written_in_the_evaluation_layout_and_cost_bounded(self, params, tmp_path):
        tasks = gen_pools.list_tasks(params)
        gen_pools.run(tasks[:1], params['dump-dir'], params)

        path = tmp_path / 'pools' / tasks[0]['dumpfile']
        pool = json.loads(path.read_text())
        info = pool['info']
        assert info['optimal-cost'] == 4 and info['optimal-cost-source'] == 'topk --number-of-plans 1'
        assert info['found-plans'] == 4 and info['filtered-out'] == 1 and info['cost-bound'] == 8.0
        assert len(pool['plans']) == 3 and all('; cost = ' in p for p in pool['plans'])
        assert info['domain'] == 'toy/domain.pddl' and info['problem'] == 'p01.pddl'
        assert info['requested-plans'] == 20 and info['planner-status'] == 0
        assert pool['total-time-seconds'] > 0

    def test_existing_pool_is_not_regenerated_without_force(self, params, capsys):
        tasks = gen_pools.list_tasks(params)
        gen_pools.run(tasks[:1], params['dump-dir'], params)
        gen_pools.run(tasks[:1], params['dump-dir'], params)

        assert 'skipping' in capsys.readouterr().out

    def test_a_missing_planner_yields_an_empty_pool_that_records_the_failure(self, params, tmp_path):
        params['fi-command'] = ['/no/such/planner']
        tasks = gen_pools.list_tasks(params)
        gen_pools.run(tasks[:1], params['dump-dir'], params)

        pool = json.loads((tmp_path / 'pools' / tasks[0]['dumpfile']).read_text())
        assert pool['plans'] == [] and pool['info']['planner-status'] == 'not-found'

    def test_generated_pools_are_matched_by_the_evaluation(self, params, tmp_path):
        from utils import match_plans_with_problems
        tasks = gen_pools.list_tasks(params)
        gen_pools.run(tasks, params['dump-dir'], params)

        matched = match_plans_with_problems(params['plansdir'], params['benchmark'], params['ru-info'])

        assert [t['task_id'] for t in matched] == [t['task_id'] for t in tasks]
        assert all(t['q'] == 2.0 and t['generator'] == 'fi-bc' for t in matched)


class TestCollectPlans:
    def test_results_json_is_the_fallback(self, tmp_path):
        (tmp_path / 'results.json').write_text(json.dumps({'plans': [
            {'cost': 3, 'actions': ['(a)', '(b)', '(c)']}, '(x)\n(y)\n; cost = 2 (unit cost)']}))

        plans, where = gen_pools.collect_plans(str(tmp_path))

        assert where == 'results.json'
        assert plans[0][1] == 3 and plans[0][0].startswith('(a)\n(b)\n(c)')
        assert plans[1][1] is None
