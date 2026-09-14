"""``bdcexp generate | run | report | jobs``: the sweep, and its job arrays."""

import argparse

from bdc_experiments import benchmark, generate, jobs, pools, report, runner
from bdc_experiments.config import load, snapshot


def _common(parser):
    parser.add_argument('config', help='a config file, or the name of a packaged one (default, smoke)')
    parser.add_argument('--results-dir', default=None, help='override [run].results_dir')
    parser.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                        help='override one int or str setting, e.g. slurm.partition=long')
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(prog='bdcexp', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    one = _common(sub.add_parser('generate', help='phase one: the pools, out of the archive'))
    one.add_argument('--instance', default=None, help='one instance id')
    one.add_argument('--domain', default=None, help='one domain')
    one.add_argument('--list', action='store_true', help='list the instances and exit')
    one.add_argument('--force', action='store_true', help='rewrite existing pools')

    two = _common(sub.add_parser('run', help='phase two: the selection sweep, or the timing'))
    two.add_argument('kind', choices=sorted(runner.KINDS))
    two.add_argument('--task', default=None, help='one task id')
    two.add_argument('--list', action='store_true', help='list the task ids and exit')
    two.add_argument('--force', action='store_true', help='rerun tasks that already have a result')
    two.add_argument('--jobs', type=int, default=1, help='local process pool size')

    three = _common(sub.add_parser('report', help='CSVs, tables, figures, manifest'))
    three.add_argument('report', choices=['setup', *sorted(runner.REPORTS), 'all'])

    four = _common(sub.add_parser('jobs', help='the sweep as slurm job arrays, and a local launcher'))
    four.add_argument('--skip-existing', action='store_true',
                      help='leave out the pools and tasks that already have a file')

    args = parser.parse_args(argv)
    try:
        cfg = load(args.config, results_dir=args.results_dir, overrides=args.set)
    except (ValueError, FileNotFoundError) as bad:
        raise SystemExit(f'bdcexp: {bad}')
    if not getattr(args, 'list', False):
        snapshot(cfg)

    if args.command == 'generate':
        if not args.list and generate.archive(cfg) is None:
            raise SystemExit('[generation].archive is empty: this config runs on committed pools')
        root = benchmark.benchmark_dir(cfg)
        root = root if root.is_dir() else benchmark.ensure(cfg)
        if args.list:
            print('\n'.join(instance['id'] for instance in benchmark.instances(cfg, root=root)))
            return 0
        benchmark.ensure(cfg)
        generate.run(cfg, force=args.force, only=args.instance, domain=args.domain)
        return 0

    pools.ensure_pools(cfg)
    if args.command == 'jobs':
        for path in jobs.write(cfg, skip_existing=args.skip_existing):
            print(path)
        return 0

    if args.command == 'run':
        if args.list:
            print('\n'.join(runner.tasks(cfg, args.kind)))
            return 0
        counts = runner.run(cfg, args.kind, only=args.task, force=args.force, jobs=args.jobs)
        print(f"{args.kind}: {counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped")
        return 1 if counts['failed'] else 0

    names = ['setup', *sorted(runner.REPORTS)] if args.report == 'all' else [args.report]
    for name in names:
        build = report.setup_report if name == 'setup' else runner.module(name).report
        for path in build(cfg):
            print(path)
    return 0
