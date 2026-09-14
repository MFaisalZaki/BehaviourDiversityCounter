"""``bdcexp jobs``: the sweep as job arrays and a local launcher, after
pyPMTEvalToolkit. Generated on the smoke pools, then run through the local
launcher, which executes the very command lines the arrays would."""

import subprocess

import pytest

from bdc_experiments import cli, runner
from bdc_experiments.config import load


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp('jobs')
    assert cli.main(['jobs', 'smoke', '--results-dir', str(root), '--skip-existing']) == 0
    return load('smoke', results_dir=root)


def test_the_arrays_are_split_and_throttled(generated):
    slurm = runner.results_root(generated) / 'slurm'
    lines = (slurm / 'cmds' / 'select.txt').read_text().splitlines()
    assert len(lines) == len(runner.tasks(generated, 'select')) > generated['slurm']['max_array_size']
    assert all(line.endswith(' --task ' + task_id) for line, task_id in zip(lines, runner.tasks(generated, 'select')))
    first, second = (slurm / 'bdcexp-select.sbatch').read_text(), (slurm / 'bdcexp-select-1.sbatch').read_text()
    size = generated['slurm']['max_array_size']
    assert f"#SBATCH --array=0-{size - 1}%{generated['slurm']['max_parallel_jobs']}" in first
    assert f"#SBATCH --array=0-{len(lines) - size - 1}%" in second
    assert 'LINE=$((0 + SLURM_ARRAY_TASK_ID + 1))' in first and f'LINE=$(({size} + ' in second
    assert '#SBATCH --partition' not in first          # empty in the config: the site's default
    assert first.rstrip().endswith('exit 0')


def test_the_arrays_are_the_two_task_kinds_and_the_report_waits_for_them(generated):
    slurm = runner.results_root(generated) / 'slurm'
    assert sorted(p.name for p in (slurm / 'cmds').iterdir()) == ['select.txt', 'time.txt']
    submit = (slurm / 'submit_all.sh').read_text()
    assert 'bdcexp-select.sbatch' in submit and 'bdcexp-time.sbatch' in submit
    assert '--dependency=afterany$TASKS' in submit and 'report' in submit


def test_the_arrays_need_the_pools_on_disk_first(tmp_path):
    from bdc_experiments import jobs
    with pytest.raises(SystemExit, match='run `bdcexp generate` first'):
        jobs.write(load('smoke', results_dir=tmp_path))


def test_the_local_launcher_runs_the_sweep_and_the_rerun_is_empty(generated):
    slurm = runner.results_root(generated) / 'slurm'
    done = subprocess.run(['bash', str(slurm / 'run_local.sh'), '2'], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    for kind in runner.KINDS:
        results = runner.load_results(generated, kind)
        assert len(results) == len(runner.tasks(generated, kind)), kind
        assert not [r for r in results if r.get('error')], kind
    assert cli.main(['jobs', 'smoke', '--results-dir', generated['run']['results_dir'],
                     '--skip-existing']) == 0
    assert all((slurm / 'cmds' / f'{kind}.txt').read_text() == '' for kind in runner.KINDS)
