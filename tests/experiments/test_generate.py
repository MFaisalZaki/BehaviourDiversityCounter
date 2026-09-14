"""Phase one: the pools out of the forbid-iterative archive.

A small archive is built from the committed smoke pool, in the archive's own
file naming and plan format, next to a one-instance benchmark checkout and a
one-instance ru-info tree, and unpacked through ``generate.run``.
"""

import json
import shutil
import zipfile

import pytest

from bdc_experiments import cli, generate, pools, runner
from bdc_experiments.config import CONFIG_DIR, load, resolve

SMOKE = CONFIG_DIR / 'smoke_pools'


def _archive_member(pool):
    """The smoke pool's plans as the archive writes them: action lines, then
    a ``;<cost> cost (unit)`` comment."""
    return {'plans': ['\n'.join(p['actions']) + f"\n;{p['cost']} cost (unit)\n" for p in pool['plans']],
            'info': {'domain': 'rovers/domain.pddl', 'problem': 'p02.pddl', 'planner': 'fi-bc',
                     'k': 1000, 'q': 2.0},
            'total-time-seconds': 12.5}


@pytest.fixture(scope='module')
def unpacked(tmp_path_factory):
    root = tmp_path_factory.mktemp('archive')
    classical = root / 'run' / 'benchmark' / 'classical' / 'rovers'
    classical.mkdir(parents=True)
    for name in ('domain.pddl', 'p02.pddl'):
        shutil.copy(SMOKE / 'pddl' / 'rovers' / name, classical / name)
    (classical / 'api.py').write_text(
        "domains = [{'name': 'rovers', 'ipc': 2006, "
        "'problems': [('rovers/domain.pddl', 'rovers/p02.pddl')]}]\n")
    (root / 'ru' / 'rovers').mkdir(parents=True)
    # Keyed by the number in the file name (p02 is 2); entry 1 is the sorted
    # position, and names a rover the problem does not have.
    (root / 'ru' / 'rovers' / 'rovers-2006.json').write_text(json.dumps(
        {'info': {'domain': 'rovers', 'year': '2006'},
         'instances': {'1': '(:resource rover7 100 0 5)', '2': '(:resource rover0 100 0 5)'}}))
    smoke_pool = json.loads((SMOKE / 'pools' / 'rovers' / 'p02' / 'topq-q2.0-N30.json').read_text())
    with zipfile.ZipFile(root / 'pools.zip', 'w') as bundle:
        bundle.writestr('fi-generated-plans-dir/2.0-1000-classical-2006-rovers-1-fi-bc-results.json',
                        json.dumps(_archive_member(smoke_pool)))
        bundle.writestr('fi-generated-plans-dir/1.0-1000-classical-2006-rovers-1-fi-bc-results.json',
                        json.dumps({'total-time-seconds': 1800.2}))            # killed: no plans key
        bundle.writestr('fi-generated-plans-dir/2.0-10-classical-2006-rovers-1-fi-bc-results.json',
                        json.dumps(_archive_member(smoke_pool)))              # another size
        bundle.writestr('__MACOSX/fi-generated-plans-dir/._2.0-1000-classical-2006-rovers-1-fi-bc-results.json', 'x')
    text = resolve('smoke').read_text()
    text = text.replace('archive = ""', f'archive = "{root / "pools.zip"}"')
    text = text.replace('resources = ""', f'resources = "{root / "ru"}"')
    text = text.replace('pool_sizes = [30]', 'pool_sizes = [1000]')
    text = text.replace('domains = ["rovers", "driverlog", "satellite"]', 'domains = ["rovers"]')
    (root / 'archive.toml').write_text(text)
    cfg = load(str(root / 'archive.toml'), results_dir=root / 'run')
    records = generate.run(cfg, log=lambda line: None)
    return cfg, records, smoke_pool


def test_the_archive_is_read_at_the_configured_sizes_and_bounds(unpacked):
    cfg, records, _ = unpacked
    assert sorted(generate.members(cfg)) == [('rovers', '2006', 1)]
    assert sorted(generate.members(cfg)[('rovers', '2006', 1)]) == [(1.0, 1000), (2.0, 1000)]
    assert [(r['q'], r['requested']) for r in records] == [(1.0, 1000), (2.0, 1000)]
    assert len(pools.pool_files(cfg)) == 2


def test_the_pool_record_carries_the_plans_the_optimum_and_the_declarations(unpacked):
    cfg, records, smoke_pool = unpacked
    full = next(r for r in records if r['q'] == 2.0)
    assert full['instance'] == 'rovers/2006/p02' and full['domain'] == 'rovers'
    assert [p['actions'] for p in full['plans']] == [p['actions'] for p in smoke_pool['plans']]
    assert full['optimal_cost'] == smoke_pool['optimal_cost']
    assert full['planner']['name'] == 'forbid-iterative' and full['wall_s'] == 12.5
    assert full['exhausted'] and not full['timed_out'] and full['cpu_s'] is None
    assert full['resources'] == '(:resource rover0 100 0 5)' and full['resources_entry'] == 2
    assert pools.Path(full['problem_file']).is_file()


def test_a_declaration_is_attached_only_where_it_names_the_problems_objects(unpacked):
    cfg, records, _ = unpacked
    instance = benchmark_instance(cfg)
    assert generate.resources_of(instance, {'1': '(:resource rover7 100 0 5)'}) == (None, None)
    assert generate.resources_of(instance, {'1': '(:resource rover0 100 0 5)'}) == ('(:resource rover0 100 0 5)', 1)
    assert generate.resources_of(instance, {'2': '(:resource rover0 100 0 5)\n(:resource rover1 100 0 5)'}) == (None, None)


def benchmark_instance(cfg):
    from bdc_experiments import benchmark
    return benchmark.instances(cfg)[0]


def test_a_timed_out_file_is_an_empty_pool_that_says_so(unpacked):
    cfg, records, _ = unpacked
    empty = next(r for r in records if r['q'] == 1.0)
    assert empty['timed_out'] and not empty['plans'] and empty['wall_s'] == 1800.2
    assert empty['optimal_cost'] is None and not empty['exhausted']


def test_a_short_pool_is_exhausted_inside_the_limit_and_timed_out_at_it(unpacked):
    cfg, _, smoke_pool = unpacked
    instance = benchmark_instance(cfg)
    data = _archive_member(smoke_pool)
    inside = generate.pool_record(instance, 2.0, 1000, dict(data, **{'total-time-seconds': 12.5}), 'm', None, time_limit_s=1800)
    at_limit = generate.pool_record(instance, 2.0, 1000, dict(data, **{'total-time-seconds': 1901.0}), 'm', None, time_limit_s=1800)
    assert inside['exhausted'] and not inside['timed_out']
    assert at_limit['timed_out'] and not at_limit['exhausted'] and len(at_limit['plans']) == 30
    full = generate.pool_record(instance, 2.0, 30, dict(data, **{'total-time-seconds': 1901.0}), 'm', None, time_limit_s=1800)
    assert not full['timed_out'] and not full['exhausted']      # every plan asked for arrived


def test_a_domain_model_is_skipped_where_the_instance_has_no_declarations(unpacked, tmp_path):
    cfg, _, _ = unpacked
    path = next(p for p in pools.pool_files(cfg) if 'q2.0' in p.name)
    pool = json.loads(path.read_text())
    stripped = pools.results_root(cfg) / 'pools' / 'rovers' / 'p02' / 'topq-q2.0-N9.json'
    stripped.write_text(json.dumps(dict(pool, resources=None, resources_entry=None)))
    try:
        result = runner.run_task(cfg, 'select', 'select/rovers/2006/p02/topq-q2.0-N9/rovers_astronaut')
        assert result['error'] is None and 'no (:resource' in result['extra']['skipped']
        generic = runner.run_task(cfg, 'select', 'select/rovers/2006/p02/topq-q2.0-N9/generic')
        assert generic['error'] is None and generic['rows']
    finally:
        stripped.unlink()


def test_unpacking_is_resumable(unpacked):
    cfg, _, _ = unpacked
    stamps = {p: p.stat().st_mtime_ns for p in pools.pool_files(cfg)}
    generate.run(cfg, log=lambda line: None)
    assert {p: p.stat().st_mtime_ns for p in pools.pool_files(cfg)} == stamps


def test_the_declarations_reach_the_astronaut_model(unpacked):
    cfg, _, _ = unpacked
    task_id = next(t for t in runner.tasks(cfg, 'select')
                   if t.endswith('/rovers_astronaut') and 'q2.0' in t)
    result = runner.run_task(cfg, 'select', task_id)
    assert result['error'] is None, result['error']
    by_key = {f['key']: f for f in result['model']['features']}
    assert by_key['rn']['objects'] == ['rover0']
    skipped = runner.run_task(cfg, 'select', task_id.replace('q2.0', 'q1.0'))
    assert 'timed out' in skipped['extra']['skipped']


def test_the_plan_reader_takes_both_cost_comments():
    assert generate.read_plan('(a x)\n(b y)\n;7 cost (unit)\n') == (['(a x)', '(b y)'], 7)
    assert generate.read_plan('(a x)\n; cost = 3 (unit cost)\n') == (['(a x)'], 3)
    assert generate.read_plan('(a x)\n') == (['(a x)'], None)


def test_a_config_without_an_archive_cannot_generate(tmp_path):
    with pytest.raises(SystemExit, match='archive is empty'):
        cli.main(['generate', 'smoke', '--results-dir', str(tmp_path), '--domain', 'rovers'])
