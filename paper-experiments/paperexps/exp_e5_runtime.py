"""E5 -- the cost of the selection phase.

On the largest pools: feature extraction (mapping, simulation included) and
each of the four selections, timed for every pool-size prefix and k of the
grid, with the pair-distance cache and without it, repeated; the report takes
medians.  Wall clock and CPU are both recorded.
"""

import os

from harness import INDICATORS, Timer, flatten, load_pool, make_counter, task_record
from model import build, default_dimensions
from report import (PALETTE, group_by, latex_table, matplotlib_or_none, save_figure, write_csv,
                    write_manifest)
from stats import median
from utils import pool_plan_count

NAME = 'E5'


def select_tasks(tasks, params):
    """The ``largest-pools`` pools by plan count, preferring the generators in
    ``generator-preference`` order.  Only the largest tenth of the files by
    size are opened to count their plans: a pool file's size grows with its
    plans, so the largest pools by count are among them."""
    preference = params.get('generator-preference') or []
    present = {t['generator'] for t in tasks}
    chosen = next((g for g in preference if g in present), None)
    candidates = [t for t in tasks if chosen is None or t['generator'] == chosen]
    by_bytes = sorted(candidates, key=lambda t: os.path.getsize(t['pool_file']), reverse=True)
    opened = by_bytes[:max(params['largest-pools'], len(by_bytes) // 10)]
    counted = sorted(opened, key=lambda t: (-pool_plan_count(t['pool_file']), t['task_id']))
    return counted[:params['largest-pools']]


def uncached(counter):
    """The same counter with pair distances recomputed on every call."""
    dimensions = list(counter.dimensions.values())
    counter._pair_distance = lambda b1, b2: sum(d.distance(b1, b2) for d in dimensions)
    return counter


def run_task(taskdetails, params):
    pool = load_pool(taskdetails)
    out = {'pool': pool.record(), 'rows': []}
    if not pool.plans:
        return out
    dimensions = default_dimensions(pool, params)
    _, out['model'] = build(pool, dimensions)
    k_nn = params['k-nn']
    sizes = [s for s in params['pool-sizes'] if s <= len(pool.plans)]
    if len(pool.plans) not in sizes:
        sizes.append(len(pool.plans))
    for size in sizes:
        prefix = pool.prefix(size)
        for repetition in range(params['repeats']):
            # Mapping from scratch: a fresh trace so the simulation is timed too.
            from behaviour_diversity_counter import BehaviourDiversityCounter
            fresh = BehaviourDiversityCounter(pool.task, dimensions, trace_cache={})
            with Timer() as timer:
                fresh.behaviours(prefix)
            out['rows'].append({'pool_size': len(prefix), 'k': None, 'phase': 'mapping', 'indicator': None,
                                'cached': None, 'repetition': repetition, 'cpu_s': timer.cpu, 'wall_s': timer.wall,
                                'generation_s': pool.generation_s})
            cached_counter = make_counter(pool, dimensions)
            cached_counter.behaviours(prefix)
            for k in [k for k in params['k-values'] if k <= len(prefix)]:
                for indicator in INDICATORS:
                    with Timer() as timer:
                        cached_counter.extract(prefix, k, indicator=indicator, k_nn=k_nn)
                    out['rows'].append({'pool_size': len(prefix), 'k': k, 'phase': 'selection',
                                        'indicator': indicator, 'cached': True, 'repetition': repetition,
                                        'cpu_s': timer.cpu, 'wall_s': timer.wall, 'generation_s': pool.generation_s})
                    plain = uncached(make_counter(pool, dimensions))
                    plain.behaviours(prefix)
                    with Timer() as timer:
                        plain.extract(prefix, k, indicator=indicator, k_nn=k_nn)
                    out['rows'].append({'pool_size': len(prefix), 'k': k, 'phase': 'selection',
                                        'indicator': indicator, 'cached': False, 'repetition': repetition,
                                        'cpu_s': timer.cpu, 'wall_s': timer.wall, 'generation_s': pool.generation_s})
    return out


def report(results, params, paths, config_file, started):
    rows = flatten(results)
    outputs, notes = [], []
    outputs.append(write_csv(os.path.join(paths['results'], 'E5_runtime_raw.csv'), rows))
    medians = []
    for key, group in sorted(group_by(rows, ('task_id', 'pool_size', 'k', 'phase', 'indicator', 'cached')).items(),
                             key=lambda item: tuple(str(x) for x in item[0])):
        medians.append({'task_id': key[0], 'domain': group[0]['domain'], 'pool_size': key[1], 'k': key[2],
                        'phase': key[3], 'indicator': key[4], 'cached': key[5],
                        'cpu_s': median([r['cpu_s'] for r in group]), 'wall_s': median([r['wall_s'] for r in group]),
                        'generation_s': group[0]['generation_s'], 'repeats': len(group)})
    outputs.append(write_csv(os.path.join(paths['results'], 'E5_runtime.csv'), medians))
    sizes = sorted({r['pool_size'] for r in medians})
    largest = max(sizes) if sizes else 0
    if largest < max(params['pool-sizes']):
        notes.append(f'largest pool on disk holds {largest} plans; the {params["pool-sizes"]} grid is cut there')
    table_rows = []
    for (size, k, indicator, cached), group in sorted(group_by([r for r in medians if r['phase'] == 'selection'],
                                                              ('pool_size', 'k', 'indicator', 'cached')).items(),
                                                      key=lambda item: (item[0][0], item[0][1], INDICATORS.index(item[0][2]), item[0][3])):
        table_rows.append({'pool_size': size, 'k': k, 'indicator': indicator, 'cached': cached,
                           'median_cpu_s': median([r['cpu_s'] for r in group]), 'pools': len(group)})
    mapping_rows = []
    for size, group in sorted(group_by([r for r in medians if r['phase'] == 'mapping'], ('pool_size',)).items()):
        mapping_rows.append({'pool_size': size[0], 'median_mapping_cpu_s': median([r['cpu_s'] for r in group]),
                             'median_generation_s': median([r['generation_s'] for r in group if r['generation_s'] is not None]),
                             'pools': len(group)})
    outputs.append(write_csv(os.path.join(paths['results'], 'E5_selection_medians.csv'), table_rows))
    outputs.append(write_csv(os.path.join(paths['results'], 'E5_mapping_vs_generation.csv'), mapping_rows))
    outputs.append(latex_table(os.path.join(paths['tables'], 'E5_selection.tex'),
                               [('pool_size', 'pool'), ('k', '$k$'), ('indicator', 'indicator'), ('cached', 'cache'),
                                ('median_cpu_s', 'median CPU s'), ('pools', 'pools')], table_rows,
                               'Selection CPU time (median over pools of medians over repetitions).', 'tab:e5-selection'))
    outputs.append(latex_table(os.path.join(paths['tables'], 'E5_mapping.tex'),
                               [('pool_size', 'pool'), ('median_mapping_cpu_s', 'mapping CPU s'),
                                ('median_generation_s', 'generation s'), ('pools', 'pools')], mapping_rows,
                               'Feature extraction against pool generation time.', 'tab:e5-mapping'))
    plt = matplotlib_or_none()
    if plt is None:
        notes.append('matplotlib is not installed: figures skipped, their data is in the CSVs')
    elif table_rows:
        k_plot = params.get('plot-k', 100)
        pts = [r for r in table_rows if r['k'] == k_plot and r['cached']]
        if not pts:
            k_plot = max(r['k'] for r in table_rows)
            pts = [r for r in table_rows if r['k'] == k_plot and r['cached']]
            notes.append(f'no selections at k = {params.get("plot-k", 100)} for the log-log plot; drew k = {k_plot}')
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for colour, indicator in zip(PALETTE, INDICATORS):
            series = sorted((r['pool_size'], r['median_cpu_s']) for r in pts if r['indicator'] == indicator)
            ax.loglog([s[0] for s in series], [max(s[1], 1e-6) for s in series], marker='o', label=indicator, color=colour)
        ax.set_xlabel('pool size'); ax.set_ylabel(f'selection CPU s (k = {k_plot})'); ax.legend(fontsize=7)
        outputs.append(save_figure(plt, fig, os.path.join(paths['figures'], 'E5_selection_loglog.pdf')))
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        labels = [str(r['pool_size']) for r in mapping_rows]
        x = range(len(labels))
        ax.bar([i - 0.2 for i in x], [r['median_mapping_cpu_s'] for r in mapping_rows], width=0.4, label='mapping', color=PALETTE[0])
        ax.bar([i + 0.2 for i in x], [r['median_generation_s'] or 0 for r in mapping_rows], width=0.4, label='generation', color=PALETTE[4])
        ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_yscale('log'); ax.set_xlabel('pool size'); ax.set_ylabel('seconds'); ax.legend(fontsize=7)
        outputs.append(save_figure(plt, fig, os.path.join(paths['figures'], 'E5_mapping_vs_generation.pdf')))
    outputs.append(write_manifest(os.path.join(paths['results'], 'E5_manifest.json'), NAME,
                                  config_file, params, started, outputs, notes))
    return outputs, notes
