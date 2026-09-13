"""``bdcexp generate | run | report``: the three things the sweep needs."""

import argparse

from bdc_experiments import benchmark, generate, pools, report, runner
from bdc_experiments.config import load, snapshot


def _common(parser):
    parser.add_argument('config', help='a config file, or the name of a packaged one (default, smoke)')
    parser.add_argument('--results-dir', default=None, help='override [run].results_dir')
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(prog='bdcexp', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    one = _common(sub.add_parser('generate', help='phase one: pools of plans'))
    one.add_argument('--instance', default=None, help='one instance id')
    one.add_argument('--list', action='store_true', help='list the instances and exit')
    one.add_argument('--force', action='store_true', help='regenerate existing pools')

    two = _common(sub.add_parser('run', help='phase two: one experiment'))
    two.add_argument('experiment', choices=sorted(runner.MODULES))
    two.add_argument('--task', default=None, help='one task id')
    two.add_argument('--list', action='store_true', help='list the task ids and exit')
    two.add_argument('--force', action='store_true', help='rerun tasks that already have a result')
    two.add_argument('--jobs', type=int, default=1, help='local process pool size')

    three = _common(sub.add_parser('report', help='CSVs, tables, figures, manifest'))
    three.add_argument('experiment', choices=sorted(runner.MODULES) + ['setup'])

    args = parser.parse_args(argv)
    cfg = load(args.config, results_dir=args.results_dir)
    if not getattr(args, 'list', False):
        snapshot(cfg)

    if args.command == 'generate':
        instances = benchmark.instances(cfg) if benchmark.benchmark_dir(cfg).is_dir() \
            else benchmark.instances(cfg, root=benchmark.ensure(cfg))
        if args.list:
            print('\n'.join(instance['id'] for instance in instances))
            return 0
        benchmark.ensure(cfg)
        generate.run(cfg, instances, force=args.force, only=args.instance)
        return 0

    if args.command == 'run':
        pools.ensure_pools(cfg)
        if args.list:
            print('\n'.join(runner.tasks(cfg, args.experiment)))
            return 0
        counts = runner.run(cfg, args.experiment, only=args.task, force=args.force, jobs=args.jobs)
        print(f"{args.experiment}: {counts['ok']} ok, {counts['failed']} failed, "
              f"{counts['skipped']} skipped")
        return 1 if counts['failed'] else 0

    pools.ensure_pools(cfg)
    if args.experiment == 'setup':
        for path in report.setup_report(cfg):
            print(path)
        return 0
    results = runner.load_results(cfg, args.experiment)
    if not results:
        raise SystemExit(f'no results for {args.experiment} under '
                         f"{cfg['run']['results_dir']}; run it first")
    for path in runner.module(args.experiment).report(cfg, results):
        print(path)
    return 0
