"""Scan the pools an experiment selects and emit its run commands and job arrays.

Job arrays rather than one ``sbatch`` per pool: a full sweep is thousands of
pools, one submission each is slow to matter and hostile to schedulers with a
submission-rate limit. One array per experiment keeps ``squeue`` readable and
lets the whole sweep be cancelled with one ``scancel``; each array index picks
its own line out of a command file, so the sweep is one submission.

    <sandbox>/
      cmds/        one command file per experiment, plus generated_cmds.sh
      slurm/       the .sbatch arrays, submit_all.sh, logs/
      results/<experiment>/   one CSV per pool, concatenated by `analyze`
      analysis/    the summary tables
      figures/     the paper figures
    run_local.sh   the same commands through a local worker pool

Nothing here imports unified_planning, so a sweep can be generated anywhere.
"""

import os
import shlex
import stat
import sys

from paperexps import common
from paperexps.config import EXPERIMENTS, Experiment
from paperexps.discover import discover, summarise


class Sandbox:
    """The directory layout every stage agrees on."""

    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.cmds_dir = os.path.join(self.root, 'cmds')
        self.slurm_dir = os.path.join(self.root, 'slurm')
        self.slurm_logs_dir = os.path.join(self.slurm_dir, 'logs')
        self.results_dir = os.path.join(self.root, 'results')
        self.analysis_dir = os.path.join(self.root, 'analysis')
        self.figures_dir = os.path.join(self.root, 'figures')

    def create(self):
        for path in (self.root, self.cmds_dir, self.slurm_dir, self.slurm_logs_dir,
                     self.results_dir, self.analysis_dir, self.figures_dir):
            os.makedirs(path, exist_ok=True)
        for name in EXPERIMENTS + ['b_overlap']:
            os.makedirs(self.experiment_results_dir(name), exist_ok=True)

    def experiment_results_dir(self, name):
        return os.path.join(self.results_dir, f'exp_{name}')

    def result_file(self, name, task_id, suffix=''):
        return os.path.join(self.experiment_results_dir(name),
                            f'{task_id}{suffix}.csv')


def cli_command(venv_dir=None):
    """How to invoke the toolkit on a compute node."""
    if venv_dir:
        candidate = os.path.join(os.path.abspath(os.path.expanduser(venv_dir)),
                                 'bin', 'bdcevalcli')
        if os.path.exists(candidate):
            return [candidate]
        python = os.path.join(os.path.abspath(os.path.expanduser(venv_dir)),
                              'bin', 'python')
        if os.path.exists(python):
            return [python, '-m', 'paperexps.cli']
    return [sys.executable, '-m', 'paperexps.cli']


def run_command(experiment, name, pool, sandbox, args):
    """The ``bdcevalcli run`` invocation for one (experiment, pool) pair."""
    parts = cli_command(args.venv_dir) + [
        'run', '--experiment', name,
        '--pool', pool.path,
        '--out', sandbox.result_file(name, pool.task_id),
        '--exp-dir', os.path.abspath(args.exp_dir),
        '--classical-domains', os.path.abspath(args.classical_domains),
        '--ru-info-dir', os.path.abspath(args.ru_info_dir),
    ]
    if name == 'b':
        # Its own directory, so `analyze` can concatenate the score rows without
        # the overlap rows landing in the same table.
        parts += ['--overlap-out', sandbox.result_file('b_overlap', pool.task_id)]
    return ' '.join(shlex.quote(part) for part in parts)


def slurm_directives(experiment, job_name, sandbox):
    slurm = experiment.slurm
    lines = [
        f'#SBATCH --job-name={job_name}',
        f'#SBATCH --output={sandbox.slurm_logs_dir}/%x_%A_%a.out',
        f'#SBATCH --error={sandbox.slurm_logs_dir}/%x_%A_%a.err',
        f"#SBATCH --cpus-per-task={slurm.get('cpus-per-task', 1)}",
        f'#SBATCH --mem={experiment.slurm_memory()}',
        f'#SBATCH --time={experiment.slurm_time()}',
    ]
    for flag, key in (('--partition', 'partition'), ('--account', 'account'),
                      ('--qos', 'qos')):
        if slurm.get(key):
            lines.append(f'#SBATCH {flag}={slurm[key]}')
    lines.extend(slurm.get('extra-directives') or [])
    return lines


def write_command_files(sandbox, per_experiment):
    all_lines = []
    for name, commands in per_experiment.items():
        if not commands:
            continue
        with open(os.path.join(sandbox.cmds_dir, f'exp_{name}.txt'), 'w') as handle:
            handle.write('\n'.join(commands) + '\n')
        all_lines.extend(commands)
    combined = os.path.join(sandbox.cmds_dir, 'generated_cmds.sh')
    with open(combined, 'w') as handle:
        handle.write('#!/bin/bash\n' + '\n'.join(all_lines) + '\n')
    make_executable(combined)
    return combined


def write_slurm_arrays(sandbox, experiment, per_experiment):
    """One job array per experiment, split so no array exceeds max-array-size."""
    chunk_size = int(experiment.slurm.get('max-array-size') or 1000)
    throttle = int(experiment.slurm.get('max-parallel-jobs') or 0)
    scripts = []
    for name, commands in per_experiment.items():
        if not commands:
            continue
        cmd_file = os.path.join(sandbox.cmds_dir, f'exp_{name}.txt')
        chunks = [(start, min(start + chunk_size, len(commands)))
                  for start in range(0, len(commands), chunk_size)]
        for number, (start, end) in enumerate(chunks, start=1):
            suffix = f'.{number}' if len(chunks) > 1 else ''
            job_name = f'bdc-exp-{name}{suffix}'
            array = f'0-{end - start - 1}' + (f'%{throttle}' if throttle > 0 else '')
            body = '\n'.join([
                '#!/bin/bash',
                *slurm_directives(experiment, job_name, sandbox),
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


def write_local_runner(sandbox, jobs):
    """A no-slurm fallback: the same commands through a local worker pool."""
    path = os.path.join(sandbox.root, 'run_local.sh')
    combined = os.path.join(sandbox.cmds_dir, 'generated_cmds.sh')
    with open(path, 'w') as handle:
        handle.write('\n'.join([
            '#!/bin/bash',
            '# Run the whole sweep locally, N pools at a time. Usage: ./run_local.sh [jobs]',
            '#',
            '# Experiment A measures wall-clock, so running it alongside anything else',
            '# on the same machine inflates exactly what it is trying to measure. Use',
            '# one job for a timing run: ./run_local.sh 1',
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
    return path


def make_executable(path):
    mode = os.stat(path).st_mode
    os.chmod(path, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def generate(args):
    experiment = Experiment.load(args.exp_dir)
    sandbox = Sandbox(args.sandbox_dir)
    sandbox.create()

    pools = discover(args.plans_dir, args.ru_info_dir, experiment.pools)
    if not pools:
        print('No pools matched the configuration; nothing to generate.', file=sys.stderr)
        return 1

    enabled = experiment.enabled_experiments()
    if not enabled:
        print('Every experiment is disabled in the configuration; nothing to generate.',
              file=sys.stderr)
        return 1

    per_experiment, skipped_done = {}, 0
    for name in enabled:
        commands = []
        for pool in pools:
            if args.skip_existing and os.path.exists(sandbox.result_file(name, pool.task_id)):
                skipped_done += 1
                continue
            commands.append(run_command(experiment, name, pool, sandbox, args))
        per_experiment[name] = commands

    combined = write_command_files(sandbox, per_experiment)
    scripts = write_slurm_arrays(sandbox, experiment, per_experiment)
    local = write_local_runner(sandbox, args.local_jobs)

    print(f'Experiment      : {experiment.summary()}')
    print(f'Sandbox         : {sandbox.root}')
    print(f'Pools           : {summarise(pools)}')
    for name in enabled:
        print(f'  experiment {name:<3} {len(per_experiment[name]):>6} commands')
    if skipped_done:
        print(f'  ({skipped_done} skipped: results already present)')
    print(f'Limits          : {experiment.cfgs["timelimit"]} / '
          f'{experiment.cfgs["memorylimit"]} per pool '
          f'(slurm gets {experiment.slurm_time()} / {experiment.slurm_memory()})')
    print(f'Commands        : {combined}')
    print(f'Slurm arrays    : {len(scripts)} scripts in {sandbox.slurm_dir}')
    print(f'Submit with     : bash {os.path.join(sandbox.slurm_dir, "submit_all.sh")}')
    print(f'Or run locally  : bash {local} {args.local_jobs}')
    return 0
