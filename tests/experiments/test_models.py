"""The diversity models: resource resolution, building, hashing, sizing."""

import subprocess
import sys

import pytest

from bdc_experiments import models, pools
from bdc_experiments.config import load


@pytest.fixture
def smoke(tmp_path):
    cfg = load('smoke', results_dir=tmp_path / 'run')
    pools.ensure_pools(cfg)
    return cfg


def info_for(cfg, pool):
    return {'id': pool['instance'], 'domain': pool['domain'],
            'optimal_cost': pool['optimal_cost'], 'q': pool['q'],
            'resource_dir': pools.results_root(cfg) / 'resources'}


class TestResourceObjects:
    def test_a_typed_domain_is_read_by_its_user_type(self, smoke):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke) if 'rovers' in str(p)))
        task = pools.task_of(pool)
        assert models.resource_objects(task, ('rover',)) == ['rover0']

    @pytest.mark.parametrize('domain, types, expected', [
        ('driverlog', ('driver',), ['driver1']),
        ('satellite', ('satellite',), ['satellite0']),
    ])
    def test_an_untyped_strips_domain_is_read_by_its_unary_predicate(
            self, smoke, domain, types, expected):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke) if domain in str(p)))
        task = pools.task_of(pool)
        found = models.resource_objects(task, types)
        assert set(expected) <= set(found) and found == sorted(found)

    def test_an_absent_type_yields_nothing_rather_than_guessing(self, smoke):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke) if 'rovers' in str(p)))
        assert models.resource_objects(pools.task_of(pool), ('submarine',)) == []

    def test_the_declaration_file_lists_one_resource_per_object(self, tmp_path):
        path = models.write_resource_file(tmp_path / 'r.txt', ['tr1', 'tr2'])
        assert path.read_text() == '(:resource tr1 0 1 1)\n(:resource tr2 0 1 1)\n'


class TestBuilding:
    def test_every_registered_model_builds_on_the_pool_of_a_domain_it_claims(self, smoke):
        built = 0
        for path in pools.pool_files(smoke):
            pool = pools.read_pool(path)
            task = pools.task_of(pool)
            for spec in models.models_for(smoke, pool['domain']):
                counter = models.build_counter(spec, task, info_for(smoke, pool))
                assert set(counter.dimensions) == {f.key for f in spec.features}
                assert models.space_size(counter) is None or models.space_size(counter) > 0
                built += 1
        assert built >= 8

    def test_a_model_whose_resources_are_absent_fails_loudly(self, smoke):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke) if 'rovers' in str(p)))
        spec = models.ModelSpec('nowhere', None,
                                (models.FeatureSpec('ru', {'types': ('submarine',)}, 1.0),))
        with pytest.raises(ValueError, match='no objects of type'):
            models.build_counter(spec, pools.task_of(pool), info_for(smoke, pool))

    def test_the_model_record_names_the_objects_and_the_goal_atoms(self, smoke):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke) if 'rovers' in str(p)))
        record = models.model_record(models.registry(smoke)['rovers_astronaut'],
                                     pools.task_of(pool), info_for(smoke, pool))
        by_key = {feature['key']: feature for feature in record['features']}
        assert by_key['rn']['objects'] == ['rover0']
        assert len(by_key['go']['goal_atoms']) == 3
        assert record['weight_convention'] == 'declared'

    def test_the_cost_bin_is_a_single_bin_at_q_one(self, smoke):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke)
                                    if 'rovers' in str(p) and 'q1.0' in str(p)))
        counter = models.build_counter(models.registry(smoke)['generic'],
                                       pools.task_of(pool), info_for(smoke, pool))
        assert models.dimension_size(counter, 'cbin') == 1


class TestHashing:
    def test_the_knobs_are_part_of_the_hash(self, smoke):
        base = models.model_hash(models.generic_spec(smoke))
        assert base != models.model_hash(models.generic_spec(smoke, goal_cap=99))
        assert base != models.model_hash(models.generic_spec(smoke, cost_bin_width=0.9))
        assert base == models.model_hash(models.generic_spec(smoke))

    def test_the_hash_is_stable_across_processes(self, smoke):
        here = models.model_hash(models.registry(smoke)['rovers_astronaut'])
        code = ('from bdc_experiments import models\n'
                'from bdc_experiments.config import load\n'
                "print(models.model_hash(models.registry(load('smoke'))['rovers_astronaut']))")
        elsewhere = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                   check=True).stdout.strip()
        assert here == elsewhere


class TestDimensionSizes:
    def test_the_space_size_is_the_product_of_the_dimension_sizes(self, smoke):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke)
                                    if 'rovers' in str(p) and 'q2.0' in str(p)))
        counter = models.build_counter(models.generic_spec(smoke, goal_cap=3, cost_bin_width=0.25),
                                       pools.task_of(pool), info_for(smoke, pool))
        # 3! orderings of the three capped goal atoms, and (2.0 - 1) / 0.25 = 4 cost bins.
        assert models.dimension_size(counter, 'go') == 6
        assert models.dimension_size(counter, 'cbin') == 4
        assert models.space_size(counter) == 24

    def test_a_count_dimension_is_reported_unbounded_rather_than_guessed(self, smoke):
        pool = pools.read_pool(next(p for p in pools.pool_files(smoke) if 'rovers' in str(p)))
        spec = models.ModelSpec('counts', None, (models.FeatureSpec('rc', {'types': ('rover',)}, 1.0),))
        counter = models.build_counter(spec, pools.task_of(pool), info_for(smoke, pool))
        assert models.dimension_size(counter, 'rc') is None
        assert models.space_size(counter) is None
