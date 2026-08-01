"""Scan the FI pools and emit slurm job arrays, one run_pool command per pool.

The layout and job-array shape mirror ASPPlanners/benchmarks (aspbench):
job arrays rather than one sbatch file per pool -- a full sweep is thousands
of pools, and one array per requested-k keeps ``squeue`` readable and lets a
whole sweep be cancelled with one ``scancel``. Each array index picks its own
line out of a command file, so the sweep is one submission.

    sandbox/
      cmds/        one command file per requested-k group, plus generated_cmds.sh
      slurm/       the .sbatch arrays, submit_all.sh, logs/
      results/     one JSON per pool (run_pool --out)
      analysis/    aggregate.py output
    run_local.sh   the same commands through a local worker pool
"""

import argparse
import os
import shlex
import stat
import sys

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common


class Sandbox:
    """The directory layout every stage agrees on."""

    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.cmds_dir = os.path.join(self.root, 'cmds')
        self.slurm_dir = os.path.join(self.root, 'slurm')
        self.slurm_logs_dir = os.path.join(self.slurm_dir, 'logs')
        self.results_dir = os.path.join(self.root, 'results')
        self.analysis_dir = os.path.join(self.root, 'analysis')

    def create(self):
        for path in (self.root, self.cmds_dir, self.slurm_dir, self.slurm_logs_dir,
                     self.results_dir, self.analysis_dir):
            os.makedirs(path, exist_ok=True)


def collect_pools(plans_dir, k_filter, skip_existing, results_dir):
    """(pool path, filename fields) for every pool that holds plans."""
    pools, skipped_empty, skipped_done = [], 0, 0
    for name in sorted(os.listdir(plans_dir)):
        if not name.endswith('-fi-bc-results.json'):
            continue
        path = os.path.join(plans_dir, name)
        fields = common.parse_pool_filename(name)
        if fields is None:
            print(f'note: unrecognised pool file name {name}', file=sys.stderr)
            continue
        if k_filter and fields['k'] not in k_filter:
            continue
        if not common.pool_has_plans(path):
            skipped_empty += 1
            continue
        if skip_existing and os.path.exists(
                os.path.join(results_dir, name.replace('.json', '') + '.json')):
            skipped_done += 1
            continue
        pools.append((path, fields))
    return pools, skipped_empty, skipped_done


def run_pool_command(pool_path, sandbox, args):
    """The run_pool invocation for one pool: absolute, quoted, cwd-independent."""
    script = os.path.join(common.PAPER_EXPERIMENTS_DIR, 'paperexps', 'run_pool.py')
    out = os.path.join(sandbox.results_dir,
                       os.path.basename(pool_path).replace('.json', '') + '.json')
    parts = [
        args.python, script,
        '--pool', pool_path,
        '--out', out,
        '--classical-domains', os.path.abspath(args.classical_domains),
        '--ru-info-dir', os.path.abspath(args.ru_info_dir),
        '--subset-sizes', ','.join(map(str, args.subset_sizes)),
        '--samples', str(args.samples),
        '--select-k', ','.join(map(str, args.select_k)),
        '--exact-sizes', ','.join(map(str, args.exact_sizes)),
        '--exact-cap', str(args.exact_cap),
        '--seed', str(args.seed),
    ]
    if args.with_states:
        parts.append('--with-states')
    return ' '.join(shlex.quote(part) for part in parts)


def slurm_directives(args, job_name, sandbox):
    lines = [
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --output={sandbox.slurm_logs_dir}/%x_%A_%a.out',
        f'#SBATCH --error={sandbox.slurm_logs_dir}/%x_%A_%a.err',
        f'#SBATCH --cpus-per-task={args.cpus}',
        f'#SBATCH --mem={args.mem}',
        f'#SBATCH --time={args.time}',
    ]
    for flag, value in (('--partition', args.partition),
                        ('--account', args.account),
                        ('--qos', args.qos)):
        if value:
            lines.append(f'#SBATCH {flag}={value}')
    return lines


def write_slurm_arrays(sandbox, per_group, args):
    """One job array per group, split so no array exceeds --max-array-size."""
    scripts = []
    for tag, commands in per_group.items():
        if not commands:
            continue
        cmd_file = os.path.join(sandbox.cmds_dir, f'{tag}.txt')
        chunks = [(start, min(start + args.max_array_size, len(commands)))
                  for start in range(0, len(commands), args.max_array_size)]
        for number, (start, end) in enumerate(chunks, start=1):
            suffix = f'.{number}' if len(chunks) > 1 else ''
            job_name = f'paperexps-{tag}{suffix}'
            array = f'0-{end - start - 1}'
            if args.max_parallel_jobs > 0:
                array += f'%{args.max_parallel_jobs}'
            body = '\n'.join([
                '#!/bin/bash',
                *slurm_directives(args, job_name, sandbox),
                f'#SBATCH --array={array}',
                '',
                '# Each array index picks its own line out of the command file, so the',
                '# whole sweep is one submission instead of one per pool.',
                'set -uo pipefail',
                f'CMD_FILE={shlex.quote(cmd_file)}',
                f'OFFSET={start}',
                'LINE=$((OFFSET + SLURM_ARRAY_TASK_ID + 1))',
                'CMD=$(sed -n "${LINE}p" "$CMD_FILE")',
                'if [ -z "$CMD" ]; then',
                '    echo "no command at line $LINE of $CMD_FILE" >&2',
                '    exit 1',
                'fi',
                'echo "[$(date -Is)] host=$(hostname) line=$LINE"',
                'echo "$CMD"',
                'eval "$CMD"',
                'status=$?',
                'echo "[$(date -Is)] exit=$status"',
                '# A failed pool is reported but must not fail the rest of the array.',
                'exit 0',
                '',
            ])
            path = os.path.join(sandbox.slurm_dir, f'{job_name}.sbatch')
            with open(path, 'w') as handle:
                handle.write(body)
            make_executable(path)
            scripts.append(path)

    submit = os.path.join(sandbox.slurm_dir, 'submit_all.sh')
    with open(submit, 'w') as handle:
        handle.write('#!/bin/bash\n# Submit every generated job array.\nset -euo pipefail\n')
        for path in scripts:
            handle.write(f'sbatch {shlex.quote(path)}\n')
    make_executable(submit)
    return scripts


def write_command_files(sandbox, per_group):
    all_lines = []
    for tag, commands in per_group.items():
        if not commands:
            continue
        with open(os.path.join(sandbox.cmds_dir, f'{tag}.txt'), 'w') as handle:
            handle.write('\n'.join(commands) + '\n')
        all_lines.extend(commands)
    combined = os.path.join(sandbox.cmds_dir, 'generated_cmds.sh')
    with open(combined, 'w') as handle:
        handle.write('#!/bin/bash\n' + '\n'.join(all_lines) + '\n')
    make_executable(combined)
    return combined


def write_local_runner(sandbox, jobs):
    """A no-slurm fallback: the same commands through a local worker pool."""
    path = os.path.join(sandbox.root, 'run_local.sh')
    combined = os.path.join(sandbox.cmds_dir, 'generated_cmds.sh')
    with open(path, 'w') as handle:
        handle.write('\n'.join([
            '#!/bin/bash',
            '# Run the whole sweep locally, N pools at a time. Usage: ./run_local.sh [jobs]',
            'set -uo pipefail',
            f'JOBS=${{1:-{max(1, jobs)}}}',
            f'CMDS={shlex.quote(combined)}',
            '',
            'if command -v parallel >/dev/null 2>&1; then',
            '    grep -v "^#" "$CMDS" | grep -v "^[[:space:]]*$" | parallel -j "$JOBS" --halt never',
            '    exit 0',
            'fi',
            '',
            'while IFS= read -r cmd; do',
            '    case "$cmd" in ""|"#"*) continue ;; esac',
            '    while [ "$(jobs -rp | wc -l | tr -d " ")" -ge "$JOBS" ]; do sleep 0.5; done',
            '    bash -c "$cmd" &',
            'done < "$CMDS"',
            'wait',
            '',
        ]))
    make_executable(path)


def make_executable(path):
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def parse_int_list(text):
    return [int(part) for part in text.split(',') if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plans-dir', default=common.DEFAULT_PLANS_DIR)
    parser.add_argument('--sandbox-dir',
                        default=os.path.join(common.PAPER_EXPERIMENTS_DIR, 'sandbox'))
    parser.add_argument('--classical-domains', default=common.DEFAULT_CLASSICAL_DOMAINS)
    parser.add_argument('--ru-info-dir', default=common.DEFAULT_RU_INFO_DIR)
    parser.add_argument('--python',
                        default=os.path.join(common.PAPER_EXPERIMENTS_DIR, 'venv', 'bin', 'python'),
                        help='Interpreter the compute nodes run (default: the setup.sh venv).')
    parser.add_argument('--k-filter', type=parse_int_list, default=[],
                        help='Only pools with these requested-k values (e.g. 100). Default: all.')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip pools that already have a result JSON.')
    # Per-pool experiment parameters, passed straight through to run_pool.
    parser.add_argument('--subset-sizes', type=parse_int_list, default=[5, 10, 20])
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--select-k', type=parse_int_list, default=[5, 10])
    parser.add_argument('--exact-sizes', type=parse_int_list, default=[5, 10])
    parser.add_argument('--exact-cap', type=int, default=200_000)
    parser.add_argument('--with-states', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    # Slurm shape, defaults matching the paper's stated limits (90 min / 8 GB).
    parser.add_argument('--time', default='01:30:00')
    parser.add_argument('--mem', default='8G')
    parser.add_argument('--cpus', type=int, default=1)
    parser.add_argument('--partition', default=None)
    parser.add_argument('--account', default=None)
    parser.add_argument('--qos', default=None)
    parser.add_argument('--max-array-size', type=int, default=1000)
    parser.add_argument('--max-parallel-jobs', type=int, default=50)
    parser.add_argument('--local-jobs', type=int, default=8)
    args = parser.parse_args(argv)

    sandbox = Sandbox(args.sandbox_dir)
    sandbox.create()

    if not os.path.isdir(args.plans_dir):
        print(f'error: plans dir not found: {args.plans_dir} '
              f'(run setup.sh to unzip data/fi-generated-plans-dir.zip)', file=sys.stderr)
        return 1

    pools, skipped_empty, skipped_done = collect_pools(
        args.plans_dir, set(args.k_filter), args.skip_existing, sandbox.results_dir)
    if not pools:
        print('No pools matched; nothing to generate.')
        return 1

    # One array per requested-k: the k=1000 pools dominate the runtime, and a
    # group per k lets them be submitted, throttled or cancelled separately.
    per_group = {}
    for path, fields in pools:
        tag = f"k{fields['k']}"
        per_group.setdefault(tag, []).append(run_pool_command(path, sandbox, args))

    combined = write_command_files(sandbox, per_group)
    scripts = write_slurm_arrays(sandbox, per_group, args)
    write_local_runner(sandbox, args.local_jobs)

    print(f'Sandbox         : {sandbox.root}')
    print(f'Pools           : {len(pools)} with plans '
          f'({skipped_empty} empty/timeout pools skipped'
          + (f', {skipped_done} already done' if skipped_done else '') + ')')
    for tag in sorted(per_group):
        print(f'  {tag:<8} {len(per_group[tag]):>6} pools')
    print(f'Limits          : --time={args.time} --mem={args.mem} per pool')
    print(f'Commands        : {combined}')
    print(f'Slurm arrays    : {len(scripts)} scripts in {sandbox.slurm_dir}')
    print(f'Submit with     : bash {os.path.join(sandbox.slurm_dir, "submit_all.sh")}')
    print(f'Or run locally  : bash {os.path.join(sandbox.root, "run_local.sh")} {args.local_jobs}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
