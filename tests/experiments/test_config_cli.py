"""The config's key checking, and the three CLI subcommands."""

import json

import pytest

from bdc_experiments import cli, pools
from bdc_experiments.config import KEYS, load, resolve


class TestConfig:
    def test_the_packaged_configs_load(self):
        for name in ('default', 'smoke'):
            cfg = load(name)
            assert set(cfg) == set(KEYS) | {'meta'}
            assert cfg['meta']['config_hash'] and cfg['meta']['run_name']

    def test_a_path_wins_over_a_packaged_name(self, tmp_path):
        path = tmp_path / 'smoke.toml'
        path.write_text(resolve('smoke').read_text())
        assert resolve(str(path)) == path

    def test_the_results_dir_can_be_overridden(self, tmp_path):
        cfg = load('smoke', results_dir=tmp_path)
        assert cfg['run']['results_dir'] == str(tmp_path)

    @pytest.mark.parametrize('edit, message', [
        ('[nonsense]\nx = 1\n', 'unknown section'),
        ('[e2]\nrandom_subsets = 25\nextra = 1\n', 'unknown key'),
    ])
    def test_a_typo_is_an_error_not_a_silently_ignored_setting(self, tmp_path, edit, message):
        text = resolve('smoke').read_text()
        if message == 'unknown key':
            text = text.replace('[e2]\nrandom_subsets = 25', '[e2]\nrandom_subsets = 25\nextra = 1')
        else:
            text += '\n' + edit
        path = tmp_path / 'bad.toml'
        path.write_text(text)
        with pytest.raises(ValueError, match=message):
            load(str(path))

    def test_an_override_is_typed_hashed_and_recorded(self):
        plain = load('smoke')
        cfg = load('smoke', overrides=['slurm.partition=long', 'slurm.max_parallel_jobs=7'])
        assert cfg['slurm']['partition'] == 'long' and cfg['slurm']['max_parallel_jobs'] == 7
        assert cfg['meta']['overrides'] == ['slurm.partition=long', 'slurm.max_parallel_jobs=7']
        assert cfg['meta']['config_hash'] != plain['meta']['config_hash']

    @pytest.mark.parametrize('override', ['slurm.nonsense=1', 'e2.weight_settings=1', 'slurm.max_parallel_jobs=x'])
    def test_an_override_of_the_wrong_key_or_type_is_an_error(self, override):
        with pytest.raises(ValueError):
            load('smoke', overrides=[override])

    def test_a_missing_section_is_an_error(self, tmp_path):
        text = '\n'.join(line for line in resolve('smoke').read_text().splitlines()
                         if not line.startswith('[e3]') and 'repeats' not in line
                         and 'largest_pools' not in line)
        path = tmp_path / 'short.toml'
        path.write_text(text)
        with pytest.raises(ValueError, match='missing (section|key)'):
            load(str(path))


class TestCli:
    def test_generate_list_needs_no_planner_run(self, tmp_path, capsys, monkeypatch):
        # --list on a run whose benchmark is absent should say so rather than
        # silently cloning several hundred megabytes inside a test.
        monkeypatch.setattr('bdc_experiments.benchmark.ensure',
                            lambda cfg: (_ for _ in ()).throw(RuntimeError('would clone')))
        with pytest.raises(RuntimeError, match='would clone'):
            cli.main(['generate', 'smoke', '--results-dir', str(tmp_path), '--list'])

    def test_run_list_seeds_the_committed_pools_and_lists_tasks(self, tmp_path, capsys):
        assert cli.main(['run', 'smoke', 'select', '--results-dir', str(tmp_path), '--list']) == 0
        printed = capsys.readouterr().out.strip().splitlines()
        assert printed and all(line.startswith('select/') for line in printed)
        assert len(pools.pool_files(load('smoke', results_dir=tmp_path))) == 4

    def test_report_setup_writes_the_two_files_the_paper_consumes(self, tmp_path, capsys):
        assert cli.main(['report', 'smoke', 'setup', '--results-dir', str(tmp_path)]) == 0
        written = capsys.readouterr().out.split()
        assert any(path.endswith('benchmark.csv') for path in written)
        assert any(path.endswith('models.csv') for path in written)
        manifest = json.loads((tmp_path / 'reports' / 'setup' / 'manifest.json').read_text())
        assert manifest['config']['hash'] and manifest['tie_breaking']
        assert manifest['planner']['name'] == 'forbid-iterative' and manifest['planner']['archive'] == ''

    def test_reporting_with_no_results_says_so(self, tmp_path):
        with pytest.raises(SystemExit, match='run the select tasks first'):
            cli.main(['report', 'smoke', 'e1', '--results-dir', str(tmp_path)])
