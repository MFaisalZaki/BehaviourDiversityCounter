"""E3's task: what the second phase costs, on both clocks, repeatedly.

A mapping sample builds a fresh counter with no trace cache and maps the whole
pool, so it pays the replay and the feature extraction. A selection sample
runs one selection on a counter mapped once and untimed, with its
behaviour-distance cache cleared first, so it pays the b^2 distances and the
greedy and the replay not at all. The planner's own time is in the pool
record, so the comparison is against the pool that was measured.
"""

import time

from bdc_experiments import models, runner

PROTOCOL = __doc__.split('\n\n')[1].replace('\n', ' ')

#: Why the planner is compared on the wall clock and not on the CPU one.
GENERATION_CLOCK = ('The planner runs as a subprocess, so the pool record measures its CPU with '
                    "getrusage(RUSAGE_CHILDREN) around the call; generation_cpu_s is the planner's "
                    'own CPU. Both ratios are in e3_timing.csv; the table and the figure use the '
                    'wall clock, which is the clock a user waits on.')


def tasks(cfg):
    return runner.pool_tasks(cfg, 'time', set(cfg['e3']['pool_sizes']),
                             lambda pool: models.timing_specs(cfg, pool['domain']))


def run_task(task_id, cfg):
    ctx = runner.context(cfg, task_id)
    spec = models.registry(cfg)[ctx['model']]
    task, counter, loaded, record, dump = runner.setup(cfg, ctx, spec)
    plans, pool = loaded['plans'], loaded['record']
    info = runner.instance_info(cfg, loaded['pool'])
    kappa = cfg['selection']['kappa_values'][0]
    samples = []
    for repeat in range(cfg['e3']['repeats']):
        wall, cpu = time.perf_counter(), time.process_time()
        models.build_counter(spec, task, info).b_coverage(plans)     # a cold replay
        samples.append({'phase': 'mapping', 'indicator': None, 'k': None, 'kappa': None,
                        'repeat': repeat, 'wall_s': time.perf_counter() - wall,
                        'cpu_s': time.process_time() - cpu})
    for k in cfg['selection']['k_values']:
        for indicator in runner.INDICATORS:
            for repeat in range(cfg['e3']['repeats']):
                counter._behaviour_distance_cache.clear()      # every sample pays b^2
                _, wall, cpu = runner.select(counter, plans, min(k, len(plans)), indicator, kappa)
                samples.append({'phase': 'selection', 'indicator': indicator,
                                'k': min(k, len(plans)), 'kappa': kappa, 'repeat': repeat,
                                'wall_s': wall, 'cpu_s': cpu})
    base = {**runner.base_row(loaded, dump, record), 'features': len(spec.features),
            'generation_wall_s': pool['generation_wall_s'],
            'generation_cpu_s': pool['generation_cpu_s'], 'exhausted': pool['exhausted']}
    rows = [{**base, **sample,
             'wall_over_generation': _ratio(sample['wall_s'], pool['generation_wall_s']),
             'cpu_over_generation': _ratio(sample['cpu_s'], pool['generation_cpu_s'])}
            for sample in samples]
    return {'pool': pool, 'model': record, 'rows': rows,
            'extra': {'samples': samples, 'protocol': PROTOCOL,
                      'generation_clock': GENERATION_CLOCK, 'kappa': kappa,
                      'kappa_note': 'kappa reaches B-Novelty alone and only through '
                                    'min(kappa, b - 1), so every selection is timed at the first '
                                    'configured kappa'}}


def _ratio(value, reference):
    """value / reference, or None where the reference is missing or zero."""
    return None if not reference else value / reference
