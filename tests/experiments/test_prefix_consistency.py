"""The selection sweep records one run to k_max and every report reads the
selection at a smaller k off its prefix. That is only sound if the selection
functions are prefix-consistent: step k+1 extends step k.

Checked here once, on the committed smoke pools, against the library itself.
k = 1 is deliberately exempt: the three dissimilarity-based rules return the
first plan of the pool at k = 1 and open on the farthest pair at k = 2, which
need not contain it. The paper reads these indicators at a fixed set size of
at least two, and every k a report reads is at least two.
"""

import pytest

from bdc_experiments import models, pools, runner
from bdc_experiments.config import load

KAPPAS = (1, 2, 3)


@pytest.fixture(scope='module')
def seeded(tmp_path_factory):
    cfg = load('smoke', results_dir=tmp_path_factory.mktemp('prefix'))
    pools.ensure_pools(cfg)
    return cfg


def cases(cfg):
    """Every (pool, model) of the smoke sweep, loaded once."""
    for path in pools.pool_files(cfg):
        pool = pools.read_pool(path)
        task = pools.task_of(pool)
        info = runner.instance_info(cfg, pool)
        trace, loaded = {}, None
        for spec in models.selection_specs(cfg, pool['domain']):
            counter = models.build_counter(spec, task, info, trace_cache=trace)
            if loaded is None:
                loaded = pools.load_pool(path, counter=counter, task=task)
            counter.b_coverage(loaded['plans'])
            yield pool['instance'], spec.name, counter, loaded['plans']


def test_a_run_to_k_max_equals_the_runs_to_each_k(seeded):
    """The prefix of the long run is the short run, plan for plan."""
    compared = 0
    for instance, model, counter, plans in cases(seeded):
        k_max = min(6, len(plans))
        for indicator in runner.INDICATORS:
            for kappa in KAPPAS:
                if indicator != 'bnovelty' and kappa != KAPPAS[0]:
                    continue        # only B-Novelty reads kappa
                long_run = counter.extract(plans, k_max, indicator=indicator, k_nn=kappa)
                for k in range(2, k_max + 1):
                    short = counter.extract(plans, k, indicator=indicator, k_nn=kappa)
                    assert [id(p) for p in short] == [id(p) for p in long_run[:k]], (
                        f'{instance} {model} {indicator} kappa={kappa}: the run to k={k} '
                        f'is not the first {k} of the run to k={k_max}')
                    compared += 1
    assert compared > 100, f'only {compared} comparisons: the smoke pools are not exercising this'


def test_every_run_returns_exactly_min_k_pool(seeded):
    """Even where the indicator fell during selection, k plans come back."""
    for instance, model, counter, plans in cases(seeded):
        for indicator in runner.INDICATORS:
            for k in (1, 2, 3, 5, len(plans) + 3):
                picked = counter.extract(plans, k, indicator=indicator, k_nn=2)
                assert len(picked) == min(k, len(plans)), (
                    f'{instance} {model} {indicator} k={k}: got {len(picked)} plans')
                assert len({id(p) for p in picked}) == len(picked), 'a plan came back twice'


def test_b_coverage_and_b_maxsum_never_fall_along_a_prefix(seeded):
    """The monotone half of claim C4, on real pools rather than random spaces."""
    for instance, model, counter, plans in cases(seeded):
        for indicator in ('bcoverage', 'bmaxsum'):
            picked = counter.extract(plans, min(6, len(plans)), indicator=indicator, k_nn=2)
            values = [runner.indicators(counter, picked[:k], 2)[indicator]
                      for k in range(1, len(picked) + 1)]
            assert values == sorted(values), (
                f'{instance} {model}: {indicator} fell along its own prefix: {values}')
