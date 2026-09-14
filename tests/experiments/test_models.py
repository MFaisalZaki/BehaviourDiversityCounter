"""The diversity models: resource resolution, building, hashing, sizing."""

import subprocess
import sys

import pytest

from bdc_experiments import models, pools, runner
from bdc_experiments.config import load


@pytest.fixture
def smoke(tmp_path):
    cfg = load('smoke', results_dir=tmp_path / 'run')
    pools.ensure_pools(cfg)
    return cfg


def pool_named(cfg, *parts):
    return pools.read_pool(next(p for p in pools.pool_files(cfg) if all(x in str(p) for x in parts)))


def built(cfg, pool, spec):
    task = pools.task_of(pool)
    return models.build_counter(spec, task, runner.instance_info(cfg, pool)), task


class TestResourceObjects:
    def test_a_typed_domain_is_read_by_its_user_type(self, smoke):
        task = pools.task_of(pool_named(smoke, 'rovers'))
        assert models.resource_objects(task, ('rover',)) == ['rover0']

    @pytest.mark.parametrize('domain, types, expected', [
        ('driverlog', ('driver',), ['driver1']),
        ('satellite', ('satellite',), ['satellite0']),
    ])
    def test_an_untyped_strips_domain_is_read_by_its_unary_predicate(
            self, smoke, domain, types, expected):
        found = models.resource_objects(pools.task_of(pool_named(smoke, domain)), types)
        assert set(expected) <= set(found) and found == sorted(found)

    def test_an_absent_type_yields_nothing_rather_than_guessing(self, smoke):
        assert models.resource_objects(pools.task_of(pool_named(smoke, 'rovers')), ('submarine',)) == []

    def test_the_declaration_file_lists_one_resource_per_object(self, tmp_path):
        path = models.write_resource_file(tmp_path / 'r.txt', ['tr1', 'tr2'])
        assert path.read_text() == '(:resource tr1 0 1 1)\n(:resource tr2 0 1 1)\n'


class TestRegistry:
    def test_every_registered_model_builds_on_a_pool_of_a_domain_it_claims(self, smoke):
        count = 0
        for path in pools.pool_files(smoke):
            pool = pools.read_pool(path)
            for spec in models.selection_specs(smoke, pool['domain']) \
                    + models.timing_specs(smoke, pool['domain']):
                counter, _ = built(smoke, pool, spec)
                assert set(counter.dimensions) == {f.key for f in spec.features}
                assert spec.name in models.registry(smoke)
                count += 1
        assert count >= 14

    def test_the_selection_sweep_runs_the_stability_model_on_every_domain(self, smoke):
        for domain in ('rovers', 'driverlog', 'satellite'):
            names = [s.name for s in models.selection_specs(smoke, domain)]
            assert 'stability' in names and 'generic' in names
            assert models.domain_model(smoke, domain).name in names

    def test_the_weight_variants_belong_to_rovers_alone(self, smoke):
        assert [s.name for s in models.weight_variants(smoke)] == ['rovers_astronaut-w0.25-0.75']
        assert not [s for s in models.selection_specs(smoke, 'driverlog')
                    if s.name.startswith('rovers_astronaut')]

    def test_the_timing_models_have_one_two_and_three_features(self, smoke):
        assert [len(s.features) for s in models.timing_specs(smoke, 'rovers')] == [1, 2, 3]
        assert [len(s.features) for s in models.timing_specs(smoke, 'gripper')] == [1, 2]

    def test_a_model_whose_resources_are_absent_fails_loudly(self, smoke):
        spec = models.ModelSpec('nowhere', None,
                                (models.FeatureSpec('ru', {'types': ('submarine',)}, 1.0),))
        with pytest.raises(ValueError, match='no objects of type'):
            built(smoke, pool_named(smoke, 'rovers'), spec)


class TestRecord:
    def test_the_model_record_names_the_objects_the_goal_atoms_and_the_sizes(self, smoke):
        pool = pool_named(smoke, 'rovers', 'q2.0')
        spec = models.registry(smoke)['rovers_astronaut']
        counter, task = built(smoke, pool, spec)
        record = models.model_record(spec, counter, task, runner.instance_info(smoke, pool))
        by_key = {feature['key']: feature for feature in record['features']}
        assert by_key['rn']['objects'] == ['rover0'] and by_key['rn']['size'] == 2
        assert len(by_key['go']['goal_atoms']) == 3 and by_key['go']['size'] == 6
        assert record['space_size'] == 12 and record['weight_convention'] == 'declared'

    def test_the_space_size_is_the_product_of_the_dimension_sizes(self, smoke):
        pool = pool_named(smoke, 'rovers', 'q2.0')
        spec = models.generic_spec(smoke)
        counter, task = built(smoke, pool, spec)
        # 3! orderings of the three capped goal atoms, and (2.0 - 1) / 0.25 = 4 cost bins.
        record = models.model_record(spec, counter, task, runner.instance_info(smoke, pool))
        assert [f['size'] for f in record['features']] == [6, 4] and record['space_size'] == 24

    def test_the_cost_bin_is_a_single_bin_at_q_one(self, smoke):
        pool = pool_named(smoke, 'rovers', 'q1.0')
        counter, _ = built(smoke, pool, models.generic_spec(smoke))
        assert models.dimension_size(counter, 'cbin') == 1

    def test_the_stability_model_is_unbounded(self, smoke):
        pool = pool_named(smoke, 'rovers', 'q2.0')
        counter, task = built(smoke, pool, models.STABILITY)
        record = models.model_record(models.STABILITY, counter, task, runner.instance_info(smoke, pool))
        assert record['space_size'] is None and record['features'][0]['size'] is None


class TestHashing:
    def test_the_knobs_and_the_weights_are_part_of_the_hash(self, smoke):
        base = models.model_hash(models.generic_spec(smoke))
        coarser = dict(smoke, models={'generic': dict(smoke['models']['generic'], goal_cap=99)})
        assert base != models.model_hash(models.generic_spec(coarser))
        assert base == models.model_hash(models.generic_spec(smoke))
        astronaut = models.model_hash(models.registry(smoke)['rovers_astronaut'])
        assert astronaut != models.model_hash(models.weight_variants(smoke)[0])

    def test_the_hash_is_stable_across_processes(self, smoke):
        here = models.model_hash(models.registry(smoke)['rovers_astronaut'])
        code = ('from bdc_experiments import models\n'
                'from bdc_experiments.config import load\n'
                "print(models.model_hash(models.registry(load('smoke'))['rovers_astronaut']))")
        elsewhere = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                   check=True).stdout.strip()
        assert here == elsewhere
