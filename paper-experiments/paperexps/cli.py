"""Command line entry point: ``bdcevalcli <command>``.

    bdcevalcli init      -> an experiment directory (limits + experiment knobs)
    bdcevalcli discover  -> which FI pools a configuration selects
    bdcevalcli generate  -> one run command per (experiment, pool), plus slurm arrays
    bdcevalcli run       -> run ONE experiment over ONE pool           (slurm calls this)
    bdcevalcli analyze   -> the summary tables and the headline numbers
    bdcevalcli report    -> the paper figures

Everything except ``run`` is stdlib-only -- none of it imports
unified_planning -- so a sweep can be generated and analysed on a laptop while
only the compute nodes carry the planning stack. (``report``'s figures want
matplotlib; ``analyze``'s tables do not.)
"""

import argparse
import os
import sys

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common
from paperexps.config import EXPERIMENTS, Experiment

__version__ = '1.0.0'


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'func', None):
        parser.print_help()
        return 1
    return args.func(args) or 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog='bdcevalcli',
        description='Empirical evaluation for "Reshaping Diversity Planning: Take 2" -- '
                    'three experiments over the ForbidIterative plan pools.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'typical flow:\n'
            '  bdcevalcli init      --exp-dir experiment\n'
            '  bdcevalcli discover  --exp-dir experiment\n'
            '  bdcevalcli generate  --exp-dir experiment --sandbox-dir sandbox\n'
            '  bash sandbox/slurm/submit_all.sh        (or: bash sandbox/run_local.sh 8)\n'
            '  bdcevalcli analyze   --sandbox-dir sandbox\n'
            '  bdcevalcli report    --sandbox-dir sandbox\n'
            '\n'
            'or let setup_benchmark.sh do all of it.\n'
        ))
    parser.add_argument('--version', action='version', version=f'bdcevalcli {__version__}')
    subparsers = parser.add_subparsers(dest='command')

    # -- init ----------------------------------------------------------
    init = subparsers.add_parser('init', help='write a starter experiment directory')
    init.add_argument('--exp-dir', required=True)
    init.add_argument('--name', default=None, help='experiment name (default: the directory)')
    init.add_argument('--time-limit', help='per pool, e.g. 01:30:00 or 90m')
    init.add_argument('--memory-limit', help='per pool, e.g. 8GB')
    init.add_argument('--force', action='store_true',
                      help='overwrite an existing exp-details.json')
    init.set_defaults(func=_init)

    # -- discover ------------------------------------------------------
    discover_parser = subparsers.add_parser(
        'discover', help='list the FI pools a configuration selects')
    _add_pool_arguments(discover_parser)
    discover_parser.add_argument('--exp-dir', default=None,
                                 help='apply this experiment\'s pool selection')
    discover_parser.add_argument('--list', action='store_true',
                                 help='print every pool path rather than a summary')
    discover_parser.set_defaults(func=_discover)

    # -- generate ------------------------------------------------------
    generate_parser = subparsers.add_parser(
        'generate', help='generate one run command per (experiment, pool) plus slurm arrays')
    generate_parser.add_argument('--exp-dir', required=True)
    generate_parser.add_argument('--sandbox-dir', required=True)
    _add_pool_arguments(generate_parser)
    generate_parser.add_argument('--venv-dir', default=None,
                                 help='virtualenv whose bdcevalcli the commands should call')
    generate_parser.add_argument('--skip-existing', action='store_true',
                                 help='skip pools that already have a result (resume a sweep)')
    generate_parser.add_argument('--local-jobs', type=int, default=4,
                                 help='parallelism baked into run_local.sh (default: 4)')
    generate_parser.set_defaults(func=_generate)

    # -- run -----------------------------------------------------------
    run_parser = subparsers.add_parser(
        'run', help='run ONE experiment over ONE pool (called by slurm)')
    run_parser.add_argument('--experiment', required=True, choices=EXPERIMENTS)
    run_parser.add_argument('--pool', action='append', required=True)
    run_parser.add_argument('--out', required=True)
    run_parser.add_argument('--overlap-out', default=None,
                            help='experiment b only: where the overlap rows go')
    run_parser.add_argument('--exp-dir', default=None,
                            help='take the knobs from this experiment (default: built-in)')
    _add_pool_arguments(run_parser, plans_dir=False)
    run_parser.add_argument('--dry-run', action='store_true',
                            help='cut the sweep down to a quick end-to-end check')
    run_parser.set_defaults(func=_run)

    # -- analyze -------------------------------------------------------
    analyze = subparsers.add_parser('analyze', help='summary tables and headline numbers')
    analyze.add_argument('--sandbox-dir', required=True)
    analyze.add_argument('--analysis-dir', default=None,
                         help='default: <sandbox>/analysis')
    analyze.set_defaults(func=_analyze)

    # -- report --------------------------------------------------------
    report = subparsers.add_parser('report', help='the paper figures')
    report.add_argument('--sandbox-dir', required=True)
    report.add_argument('--figures-dir', default=None, help='default: <sandbox>/figures')
    report.set_defaults(func=_report)

    return parser


def _add_pool_arguments(parser, plans_dir=True):
    if plans_dir:
        parser.add_argument('--plans-dir', default=common.DEFAULT_PLANS_DIR,
                            help='unpacked FI pools (default: data/fi-generated-plans-dir)')
    parser.add_argument('--ru-info-dir', default=common.DEFAULT_RU_INFO_DIR,
                        help='per-domain (:resource ...) declarations')
    parser.add_argument('--classical-domains', default=common.DEFAULT_CLASSICAL_DOMAINS,
                        help='classical-domains checkout (repo root or its classical/ dir)')


# ----------------------------------------------------------------------

def _init(args):
    path = os.path.join(os.path.abspath(os.path.expanduser(args.exp_dir)),
                        'exp-details.json')
    if os.path.exists(path) and not args.force:
        print(f'{path} already exists; pass --force to overwrite it', file=sys.stderr)
        return 1
    experiment = Experiment.default(
        args.name or os.path.basename(os.path.abspath(args.exp_dir)))
    if args.time_limit:
        experiment.cfgs['timelimit'] = args.time_limit
    if args.memory_limit:
        experiment.cfgs['memorylimit'] = args.memory_limit
    experiment.save(args.exp_dir)
    print(f'wrote {path}')
    print(f'  {experiment.summary()}')
    return 0


def _discover(args):
    from paperexps.discover import discover, summarise

    selection = Experiment.load(args.exp_dir).pools if args.exp_dir else None
    pools = discover(args.plans_dir, args.ru_info_dir, selection)
    if args.list:
        for pool in pools:
            print(pool.path)
        return 0 if pools else 1
    print(summarise(pools))
    by_domain = {}
    for pool in pools:
        by_domain.setdefault(pool.domain, []).append(pool)
    for domain in sorted(by_domain):
        group = by_domain[domain]
        print(f'  {domain:<26} {len(group):>5} pools  '
              f'median recorded behaviour-count '
              f'{sorted(p.behaviour_count or 0 for p in group)[len(group) // 2]}')
    return 0 if pools else 1


def _generate(args):
    from paperexps.generate import generate

    return generate(args)


def _run(args):
    """Translate the experiment's knobs into the runner's own arguments."""
    experiment = (Experiment.load(args.exp_dir) if args.exp_dir
                  else Experiment.default())
    knobs = experiment.knobs
    settings = experiment.experiment(args.experiment)

    argv = ['--out', args.out,
            '--classical-domains', args.classical_domains,
            '--ru-info-dir', args.ru_info_dir,
            '--seed', str(knobs.get('seed', 2026)),
            '--k-nn', str(knobs.get('k-nn', 3))]
    for path in args.pool:
        argv += ['--pool', path]
    if args.dry_run:
        argv.append('--dry-run')
    if not knobs.get('multiset-stability', True) and args.experiment in ('a', 'b'):
        argv.append('--set-stability')

    if args.experiment == 'a':
        from paperexps import exp_a as module
        argv += ['--k', str(settings.get('k', 20)),
                 '--repeats', str(settings.get('repeats', 5)),
                 '--pool-sizes', ','.join(str(v) for v in settings.get('pool-sizes', [])),
                 '--k-sweep', ','.join(str(v) for v in settings.get('k-sweep', []))]
    elif args.experiment == 'b':
        from paperexps import exp_b as module
        argv += ['--k', str(settings.get('k', 20)),
                 '--hidden-draws', str(settings.get('hidden-draws', 50))]
        if args.overlap_out:
            argv += ['--overlap-out', args.overlap_out]
    else:
        from paperexps import exp_c as module
        argv += ['--k', str(settings.get('k', 20)),
                 '--step', str(settings.get('step', 0.05))]
        if settings.get('require-varying-ru'):
            argv.append('--require-varying-ru')

    return module.main(argv)


def _sandbox_paths(sandbox_dir):
    from paperexps.generate import Sandbox

    sandbox = Sandbox(sandbox_dir)
    return sandbox, {
        'exp_a': sandbox.experiment_results_dir('a'),
        'exp_b': sandbox.experiment_results_dir('b'),
        'exp_b_overlap': sandbox.experiment_results_dir('b_overlap'),
        'exp_c': sandbox.experiment_results_dir('c'),
    }


def _analyze(args):
    from paperexps.aggregate import main as aggregate_main

    sandbox, paths = _sandbox_paths(args.sandbox_dir)
    argv = ['--analysis-dir', args.analysis_dir or sandbox.analysis_dir]
    for flag, key in (('--exp-a', 'exp_a'), ('--exp-b-scores', 'exp_b'),
                      ('--exp-b-overlap', 'exp_b_overlap'), ('--exp-c', 'exp_c')):
        if os.path.isdir(paths[key]):
            argv += [flag, paths[key]]
    return aggregate_main(argv)


def _report(args):
    from paperexps.plots import main as plots_main

    sandbox, paths = _sandbox_paths(args.sandbox_dir)
    argv = ['--figures-dir', args.figures_dir or sandbox.figures_dir]
    for flag, key in (('--exp-a', 'exp_a'), ('--exp-b-scores', 'exp_b'),
                      ('--exp-b-overlap', 'exp_b_overlap'), ('--exp-c', 'exp_c')):
        if os.path.isdir(paths[key]):
            argv += [flag, paths[key]]
    return plots_main(argv)


if __name__ == '__main__':
    sys.exit(main())
