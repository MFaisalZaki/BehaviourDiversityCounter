"""Pool loading: the round trip, the two drop lists, the order and the cache."""

import json

import pytest

from bdc_experiments import models, pools
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
    info = {'id': pool['instance'], 'domain': pool['domain'], 'optimal_cost': pool['optimal_cost'],
            'q': pool['q'], 'resource_dir': pools.results_root(cfg) / 'resources'}
    return models.build_counter(models.registry(cfg)[name], task, info), task


class TestRoundTrip:
    def test_relative_pddl_paths_resolve_against_the_pool_file(self, rovers_pool):
        pool = pools.read_pool(rovers_pool)
        assert pool['domain_file'].endswith('.pddl')
        assert pools.Path(pool['domain_file']).is_file()
        assert pools.Path(pool['problem_file']).is_file()

    def test_the_pool_record_carries_what_the_report_needs(self, smoke, rovers_pool):
        loaded = pools.load_pool(rovers_pool)
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
        loaded = pools.load_pool(path)
        assert len(loaded['record']['dropped_replay']) == 1
        assert loaded['record']['size'] == len(pool['plans'])

    def test_a_plan_over_the_quality_bound_is_dropped(self, smoke, rovers_pool, tmp_path):
        pool = pools.read_pool(rovers_pool)
        tightened = dict(pool, q=1.0)          # the bound becomes c* itself
        tightened['plans'] = list(pool['plans'])
        path = tmp_path / 'tight.json'
        path.write_text(json.dumps(tightened))
        loaded = pools.load_pool(path)
        assert loaded['record']['cost_bound'] == pytest.approx(float(pool['optimal_cost']))
        assert all(cost <= pool['optimal_cost'] for cost in loaded['costs'])

    def test_a_top_k_pool_has_no_quality_bound(self, rovers_pool, tmp_path):
        pool = dict(pools.read_pool(rovers_pool), mode='topk')
        path = tmp_path / 'topk.json'
        path.write_text(json.dumps(pool))
        loaded = pools.load_pool(path)
        assert loaded['record']['cost_bound'] is None and not loaded['record']['dropped_cost']


class TestOrderAndCache:
    def test_the_pool_is_cost_sorted(self, rovers_pool):
        costs = pools.load_pool(rovers_pool)['costs']
        assert costs == sorted(costs)

    def test_the_behaviour_dump_is_written_once_and_read_back(self, smoke, rovers_pool):
        pool = pools.read_pool(rovers_pool)
        counter, task = counter_for(smoke, pool)
        loaded = pools.load_pool(rovers_pool, counter=counter, task=task)
        record = models.model_record(models.registry(smoke)['generic'], task,
                                     {'id': pool['instance']})
        first = pools.behaviour_dump(smoke, counter, loaded, record)
        path = pools.dump_path(smoke, record['hash'], loaded)
        stamp = path.stat().st_mtime_ns
        second = pools.behaviour_dump(smoke, counter, loaded, record)
        assert path.stat().st_mtime_ns == stamp, 'the second load rewrote the dump'
        assert first == second

    def test_the_dump_holds_the_raw_material_for_a_recomputation(self, smoke, rovers_pool):
        pool = pools.read_pool(rovers_pool)
        counter, task = counter_for(smoke, pool)
        loaded = pools.load_pool(rovers_pool, counter=counter, task=task)
        record = models.model_record(models.registry(smoke)['generic'], task,
                                     {'id': pool['instance']})
        dump = pools.behaviour_dump(smoke, counter, loaded, record)
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
