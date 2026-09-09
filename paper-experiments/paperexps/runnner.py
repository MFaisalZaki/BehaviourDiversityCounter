"""Run one experiment of the paper's evaluation, one task at a time, or build
its report from the tasks already run.

    python runnner.py --config-file exp-cfg-files/default.json --experiment-name E1 --list-tasks
    python runnner.py --config-file ... --experiment-name E1 --task-id 1.0-10-classical-2002-rovers-1-fi-bc
    python runnner.py --config-file ... --experiment-name E1              # every task, resumable
    python runnner.py --config-file ... --experiment-name E1 --report     # CSVs, tables, figures, manifest
    python runnner.py --config-file ... --experiment-name GEN-fi-q2       # generate pools (see gen_pools.py)

A task is one pool file and writes one result file into the experiment's
``dump-dir``; ``--list-tasks`` prints ``task_id<TAB>dumpfile``, the manifest a
slurm array indexes into (see ../setup_benchmark.sh).
"""

import argparse
import json
import os
import time

import exp_e0_case_study
import exp_e1_metric_vs_feature
import exp_e2_cross_eval
import exp_e3_greedy_vs_optimal
import exp_e4_generators
import exp_e5_runtime
import exp_e6_sensitivity
import gen_pools
from harness import load_results, run_tasks
from utils import create_dump_dir, dumpfile_name, filter_tasks, match_plans_with_problems, resolve

EXPERIMENTS = {module.NAME: module for module in (
    exp_e0_case_study, exp_e1_metric_vs_feature, exp_e2_cross_eval, exp_e3_greedy_vs_optimal,
    exp_e4_generators, exp_e5_runtime, exp_e6_sensitivity)}


def load_config(path, name):
    with open(path) as handle:
        config = json.load(handle)
    entry = next((e for e in config['experiments'] if e['name'] == name), None)
    if entry is None:
        raise SystemExit(f"Experiment '{name}' not found in {path}; "
                         f"declared: {[e['name'] for e in config['experiments']]}")
    params = {**config.get('grid', {}), **entry, 'seed': config.get('seed'),
              'output': config.get('output', {})}
    return config, params


def is_generation(params):
    return params.get('kind') == 'generate'


def experiment_tasks(params):
    if is_generation(params):
        return gen_pools, gen_pools.list_tasks(params)
    module = EXPERIMENTS[params['name']]
    tasks = match_plans_with_problems(params['plansdir'], params['benchmark'], params['ru-info'])
    tasks = filter_tasks(tasks, params)
    if hasattr(module, 'select_tasks'):
        tasks = module.select_tasks(tasks, params)
    return module, tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one experiment of the paper's evaluation.")
    parser.add_argument('--config-file', required=True, help='Path to the configuration file.')
    parser.add_argument('--experiment-name', required=True,
                        help=f'One of {sorted(EXPERIMENTS)}, or a generation entry of the configuration.')
    parser.add_argument('--list-tasks', action='store_true',
                        help='Print "task_id<TAB>dumpfile" for every task and exit. '
                             'This is the manifest a slurm array indexes into.')
    parser.add_argument('--task-id', type=str, default=None,
                        help='Run only this task. One array element, one task.')
    parser.add_argument('--force', action='store_true', help='Rerun tasks whose result file exists.')
    parser.add_argument('--report', action='store_true',
                        help='Build the CSVs, tables, figures and manifest from the results on disk.')
    args = parser.parse_args(argv)

    config, params = load_config(args.config_file, args.experiment_name)
    if not is_generation(params) and params['name'] not in EXPERIMENTS:
        raise SystemExit(f"unknown experiment '{args.experiment_name}'; one of {sorted(EXPERIMENTS)} "
                         f"or an entry with \"kind\": \"generate\"")
    basedir = create_dump_dir(params['dump-dir'])

    if is_generation(params):
        module, tasks = experiment_tasks(params)
        if args.report:
            print(f"| {params['name']} is a generation sweep: its output is the pool files in {basedir}")
            return
        if args.list_tasks:
            for t in tasks:
                print(f"{t['task_id']}\t{t['dumpfile']}")
            return
        if args.task_id is not None:
            tasks = [t for t in tasks if t['task_id'] == args.task_id]
            assert tasks, f"no task with id '{args.task_id}' in this configuration"
        gen_pools.run(tasks, basedir, params, force=args.force)
        return

    module = EXPERIMENTS[params['name']]
    if args.report:
        started = time.strftime('%Y-%m-%dT%H:%M:%S')
        paths = {key: resolve(params['output'][key]) for key in ('results', 'tables', 'figures')}
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
        results = load_results(basedir)
        outputs, notes = module.report(results, params, paths, args.config_file, started)
        print(f"| {params['name']}: {len(results)} result files read")
        for path in outputs:
            print(f'|   wrote {path}')
        for note in notes:
            print(f'|   note: {note}')
        return

    module, tasks = experiment_tasks(params)

    if args.list_tasks:
        for t in tasks:
            print(f"{t['task_id']}\t{dumpfile_name(t)}")
        return

    if args.task_id is not None:
        tasks = [t for t in tasks if t['task_id'] == args.task_id]
        assert tasks, f"no task with id '{args.task_id}' in this configuration"

    run_tasks(tasks, basedir, module.run_task, params, force=args.force)


if __name__ == '__main__':
    main()
