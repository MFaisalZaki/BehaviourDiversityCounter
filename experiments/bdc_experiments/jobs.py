"""``bdcexp jobs``: the sweep as job arrays, after pyPMTEvalToolkit.

One command file per job kind under ``runs/<name>/slurm/cmds/``, one
``.sbatch`` job array per chunk of at most ``[slurm].max_array_size`` lines,
each index reading its own line of the file, a ``submit_all.sh`` that submits
the arrays in dependency order (pools, then the two task kinds, then the
report), and a ``run_local.sh`` that runs the same commands through GNU
parallel where there is no scheduler. Phase one is arrayed per domain, so the
per-domain instance cap keeps its sequential meaning inside one job.
"""

import shlex
import sys
from pathlib import Path

from bdc_experiments import generate, pools, runner
from bdc_experiments.config import results_root

SBATCH = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={logs}/%x_%A_%a.out
#SBATCH --error={logs}/%x_%A_%a.err
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={memory}M
#SBATCH --time={time}
#SBATCH --array=0-{last}%{throttle}
{directives}
set -uo pipefail
LINE=$(({offset} + SLURM_ARRAY_TASK_ID + 1))
CMD=$(sed -n "${{LINE}}p" {commands})
echo "$CMD"
eval "$CMD"
exit 0
"""

RUN_LOCAL = """#!/bin/bash
# The same commands without a scheduler: `run_local.sh [jobs]`, GNU parallel
# where it exists, bash background jobs otherwise.
set -uo pipefail
JOBS="${1:-4}"
cd "$(dirname "$0")"
for kind in {kinds}; do
    [ -s "cmds/$kind.txt" ] || continue
    if command -v parallel >/dev/null; then
        parallel -j "$JOBS" --halt never < "cmds/$kind.txt"
    else
        while IFS= read -r cmd; do
            while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 1; done
            eval "$cmd" &
        done < "cmds/$kind.txt"
        wait
    fi
done
"""


def _cli():
    """The ``bdcexp`` of the environment generating the jobs: the console
    script next to the interpreter, or the bare name on the PATH."""
    script = Path(sys.executable).parent / 'bdcexp'
    return str(script) if script.is_file() else 'bdcexp'


def _hms(seconds):
    return f'{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}'


def commands(cfg, skip_existing=False):
    """``kind -> [command, ...]`` for the three arrays of the sweep."""
    base = f"{_cli()} {{verb}} {shlex.quote(cfg['meta']['config_path'])} " \
           f"--results-dir {shlex.quote(str(results_root(cfg)))}"
    found = {'generate': [f"{base.format(verb='generate')} --domain {domain}"
                          for domain in cfg['benchmark']['domains']
                          if generate.archive(cfg) is not None and not (skip_existing and any(
                              path.parent.parent.name == domain for path in pools.pool_files(cfg)))]}
    for kind in runner.KINDS:
        found[kind] = [f"{base.format(verb='run')} {kind} --task {task_id}"
                       for task_id in runner.tasks(cfg, kind)
                       if not (skip_existing and runner.result_path(cfg, kind, task_id).is_file())]
    return found


def _limits(cfg):
    """``(time, memory_mb)`` of one array element: the task's own limit plus
    the configured headroom. Phase one only unpacks the archive, so every
    kind gets the same limits."""
    slurm = cfg['slurm']
    return (_hms(cfg['run']['time_limit_selection_s'] + slurm['time_headroom_s']),
            slurm['memory_mb'] + slurm['memory_headroom_mb'])


def write(cfg, skip_existing=False):
    """Write the command files, the job arrays and the two launchers; return
    the paths written."""
    slurm, root = cfg['slurm'], results_root(cfg) / 'slurm'
    logs, cmds = root / 'logs', root / 'cmds'
    for directory in (logs, cmds):
        directory.mkdir(parents=True, exist_ok=True)
    directives = [f"#SBATCH --{key}={slurm[key]}" for key in ('partition', 'account', 'qos')
                  if slurm.get(key)]
    directives += [d if d.startswith('#SBATCH') else f'#SBATCH {d}' for d in slurm['extra_directives']]
    written, arrays = [], {}
    for kind, lines in commands(cfg, skip_existing).items():
        path = cmds / f'{kind}.txt'
        path.write_text(''.join(line + '\n' for line in lines))
        written.append(path)
        arrays[kind] = []
        time, memory = _limits(cfg)
        for chunk, offset in enumerate(range(0, len(lines), slurm['max_array_size'])):
            size = min(slurm['max_array_size'], len(lines) - offset)
            script = root / f"bdcexp-{kind}{f'-{chunk}' if offset else ''}.sbatch"
            script.write_text(SBATCH.format(
                name=f'bdcexp-{kind}', logs=logs, cpus=slurm['cpus_per_task'], memory=memory,
                time=time, last=size - 1, throttle=slurm['max_parallel_jobs'],
                directives='\n'.join(directives), offset=offset, commands=shlex.quote(str(path))))
            arrays[kind].append(script)
            written.append(script)

    submit = ['#!/bin/bash', '# Submit every generated job array, each task kind waiting for the',
              '# pools, and the report waiting for every task.', 'set -euo pipefail', 'POOLS=""']
    if arrays['generate']:
        # The benchmark is cloned once here, in the submitting shell, and never
        # by the array elements, which would race for the same checkout. The
        # arrays then unpack the archive, one domain per element.
        submit.append(f"{_cli()} generate {shlex.quote(cfg['meta']['config_path'])} "
                      f"--results-dir {shlex.quote(str(results_root(cfg)))} --list > /dev/null")
    submit += [f'POOLS="$POOLS:$(sbatch --parsable {shlex.quote(str(s))})"' for s in arrays['generate']]
    submit += ['TASKS=""']
    for kind in runner.KINDS:
        submit += [f'TASKS="$TASKS:$(sbatch --parsable ${{POOLS:+--dependency=afterany$POOLS}} '
                   f'{shlex.quote(str(s))})"' for s in arrays[kind]]
    report = (f"{_cli()} report {shlex.quote(cfg['meta']['config_path'])} "
              f"--results-dir {shlex.quote(str(results_root(cfg)))} all")
    submit += ['[ -n "$TASKS" ] && sbatch --parsable --job-name=bdcexp-report '
               f'--dependency=afterany$TASKS --output={logs}/%x_%j.out '
               f"--cpus-per-task=1 --mem={slurm['memory_mb']}M "
               f"--time={_hms(cfg['run']['time_limit_selection_s'])} --wrap {shlex.quote(report)}", '']
    for name, text in (('submit_all.sh', '\n'.join(submit)),
                       ('run_local.sh', RUN_LOCAL.replace('{kinds}', ' '.join(['generate', *runner.KINDS])))):
        path = root / name
        path.write_text(text)
        path.chmod(0o755)
        written.append(path)
    return written
