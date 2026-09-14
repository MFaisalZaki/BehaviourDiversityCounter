"""Pool loading: the round trip, the two drop lists, the order and the cache."""

import json

import pytest

from bdc_experiments import models, pools, runner
from bdc_experiments.config import load


@pytest.fixture
def smoke(tmp_path):
    """The smoke config, seeded with the committed pools, in a temp run dir."""
    cfg = load('smoke', results_dir=tmp_path / 'run')
    pools.ensure_pools(cfg)
    return cfg


@pytest.fixture
def rovers_pool(smoke):
    return next(p for p in pools.pool_files(smoke) if 'rovers' in str(p) and 'q2.0' in str(p))


def counter_for(cfg, pool, name='generic'):
    task = pools.task_of(pool)
    return models.build_counter(models.registry(cfg)[name], task, runner.instance_info(cfg, pool)), task


def loaded_pool(cfg, path):
    pool = pools.read_pool(path)
    return pools.load_pool(path, *counter_for(cfg, pool))


def dump_for(cfg, path, name='generic'):
    pool = pools.read_pool(path)
    counter, task = counter_for(cfg, pool, name)
    loaded = pools.load_pool(path, counter, task)
    record = models.model_record(models.registry(cfg)[name], counter, task, {'id': pool['instance']})
    where = loaded['record']
    path = pools.dump_path(cfg, record['hash'], where['domain'], where['instance'], where['pool_stem'])
    return pools.behaviour_dump(cfg, counter, loaded, record), path, loaded


class TestRoundTrip:
    def test_relative_pddl_paths_resolve_against_the_pool_file(self, rovers_pool):
        pool = pools.read_pool(rovers_pool)
        assert pool['domain_file'].endswith('.pddl')
        assert pools.Path(pool['domain_file']).is_file()
        assert pools.Path(pool['problem_file']).is_file()

    def test_the_pool_record_carries_what_the_report_needs(self, smoke, rovers_pool):
        loaded = loaded_pool(smoke, rovers_pool)
        record = loaded['record']
        for key in ('instance', 'domain', 'pool_stem', 'mode', 'q', 'requested', 'size',
                    'dropped_replay', 'dropped_cost', 'optimal_cost', 'parse_s', 'replay_s'):
            assert key in record, key
        assert record['size'] == len(loaded['plans']) == len(loaded['costs'])


class TestDropping:
    def test_a_plan_that_cannot_be_replayed_is_listed_not_raised(self, smoke, rovers_pool, tmp_path):
        pool = pools.read_pool(rovers_pool)
        # A first action that no state satisfies: the replay fails, the pool does not.
        broken = dict(pool)
        broken['plans'] = list(pool['plans']) + [
            {'actions': ['(communicate_soil_data rover0 general waypoint0 waypoint0 waypoint0)'],
             'cost': 1}]
        path = tmp_path / 'broken.json'
        path.write_text(json.dumps(broken))
        loaded = loaded_pool(smoke, path)
        assert len(loaded['record']['dropped_replay']) == 1
        assert loaded['record']['size'] == len(pool['plans'])

    def test_a_plan_over_the_quality_bound_is_dropped(self, smoke, rovers_pool, tmp_path):
        pool = pools.read_pool(rovers_pool)
        tightened = dict(pool, q=1.0)          # the bound becomes c* itself
        tightened['plans'] = list(pool['plans'])
        path = tmp_path / 'tight.json'
        path.write_text(json.dumps(tightened))
        loaded = loaded_pool(smoke, path)
        assert loaded['record']['cost_bound'] == pytest.approx(float(pool['optimal_cost']))
        assert all(cost <= pool['optimal_cost'] for cost in loaded['costs'])


class TestOrderAndCache:
    def test_the_pool_is_cost_sorted(self, smoke, rovers_pool):
        costs = loaded_pool(smoke, rovers_pool)['costs']
        assert costs == sorted(costs)

    def test_the_behaviour_dump_is_written_once_and_read_back(self, smoke, rovers_pool):
        first, path, _ = dump_for(smoke, rovers_pool)
        stamp = path.stat().st_mtime_ns
        second, _, _ = dump_for(smoke, rovers_pool)
        assert path.stat().st_mtime_ns == stamp, 'the second load rewrote the dump'
        assert first == second

    def test_the_dump_holds_the_raw_material_for_a_recomputation(self, smoke, rovers_pool):
        dump, _, loaded = dump_for(smoke, rovers_pool)
        b = len(dump['distinct'])
        assert len(dump['plans']) == loaded['record']['size']
        assert len(dump['matrix']) == b and all(len(row) == b for row in dump['matrix'])
        assert all(dump['matrix'][i][i] == 0.0 for i in range(b))
        assert all(dump['matrix'][i][j] == dump['matrix'][j][i]
                   for i in range(b) for j in range(b))
        # Definiteness: distinct behaviours are at a positive distance.
        assert all(dump['matrix'][i][j] > 0 for i in range(b) for j in range(b) if i != j)
        # The per-plan behaviour is the tuple the matrix is indexed by.
        for entry in dump['plans']:
            assert dump['distinct'][entry['distinct']] == entry['behaviour']

    def test_the_stability_dump_carries_the_action_sets_and_no_matrix(self, smoke, rovers_pool):
        """Under the stability model b runs to the pool size, so the matrix
        is left out; the behaviour tuple is the action set a reader recomputes
        the stability distance from."""
        dump, _, loaded = dump_for(smoke, rovers_pool, 'stability')
        assert dump['matrix'] is None and dump['features'] == ['stability']
        assert 1 <= len(dump['distinct']) <= loaded['record']['size']
        for entry in dump['plans']:
            actions = set(entry['behaviour'][0].split(' ; '))
            assert actions == {str(a) for a in loaded['plans'][entry['index']].actions}
