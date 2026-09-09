"""E4 -- pool generators and reachable behaviours.

For every (instance, q) the pools of all generators present are cut to the
same size N and compared on how many distinct behaviours they expose and on
what each selection returns for every k <= N.  Each task is one pool; it
finds its siblings by file name.
"""

import glob
import os

from harness import (INDICATORS, Timer, coverage_rows, flatten, load_pool, mean_cost_ratio, scores,
                     select_all)
from model import build, default_dimensions
from report import (PALETTE, group_by, latex_table, matplotlib_or_none, save_figure, write_csv,
                    write_manifest)
from stats import median
from utils import RESULTS_SUFFIX, generator_name, pool_plan_count

NAME = 'E4'


def sibling_sizes(taskdetails):
    """``{generator tag: plan count}`` of every pool of this (q, k, instance)."""
    t = taskdetails
    pattern = os.path.join(os.path.dirname(t['pool_file']),
                           f"{t['q']}-{t['k']}-{t['track']}-{t['year']}-{t['domain']}-{t['inst']}-*{RESULTS_SUFFIX}")
    sizes = {}
    prefix = f"{t['q']}-{t['k']}-{t['track']}-{t['year']}-{t['domain']}-{t['inst']}-"
    for path in sorted(glob.glob(pattern)):
        tag = os.path.basename(path)[len(prefix):-len(RESULTS_SUFFIX)]
        sizes[tag] = pool_plan_count(path)
    return sizes


def run_task(taskdetails, params):
    sizes = sibling_sizes(taskdetails)
    cap = min(params['pool-cap-factor'] * taskdetails['k'], params['pool-cap'])
    shared = min([cap] + [size for size in sizes.values()])
    out = {'pool': None, 'rows': [], 'extra': {'sibling_sizes': sizes, 'generators_present': sorted(sizes),
                                              'shared_size': shared, 'cap': cap}}
    pool = load_pool(taskdetails)
    out['pool'] = pool.record()
    plans = pool.prefix(shared)
    out['extra']['pool_size'] = len(plans)
    if not plans:
        return out
    counter, out['model'] = build(pool, default_dimensions(pool, params))
    with Timer() as mapping:
        counter.behaviours(plans)
    out['mapping'] = {'cpu_s': mapping.cpu, 'wall_s': mapping.wall}
    k_nn = params['k-nn']
    distinct = counter.b_coverage(plans)
    for k in [k for k in params['k-values'] if k <= len(plans)]:
        selected, timing = select_all(counter, plans, k, k_nn)
        for indicator in INDICATORS:
            out['rows'].append({
                'pool_size': len(plans), 'distinct_behaviours': distinct, 'k': k, 'selector': indicator,
                **scores(counter, selected[indicator], k_nn),
                'generation_cpu_s': pool.generation_s,
                'mean_cost_ratio': mean_cost_ratio(selected[indicator], pool.optimal_cost),
                'selection_cpu_s': timing[indicator]['cpu_s'],
                'shared_size': shared, 'available_size': len(pool.plans)})
    return out


def report(results, params, paths, config_file, started):
    rows = flatten(results)
    outputs, notes = [], []
    outputs.append(write_csv(os.path.join(paths['results'], 'E4_generators.csv'), rows))
    outputs.append(write_csv(os.path.join(paths['results'], 'E4_coverage.csv'), coverage_rows(results)))
    generators = sorted({r['generator_name'] for r in rows})
    if len(generators) < 2:
        notes.append(f'only {generators or "no"} pools on disk: the generator comparison has a single column')
    ratio_rows = []
    for (domain, generator), group in sorted(group_by(rows, ('domain', 'generator_name')).items()):
        per_pool = {r['task_id']: r['distinct_behaviours'] / r['pool_size'] for r in group if r['pool_size']}
        ratio_rows.append({'domain': domain, 'generator': generator, 'median_ratio': median(list(per_pool.values())),
                           'pools': len(per_pool)})
    outputs.append(write_csv(os.path.join(paths['results'], 'E4_distinct_ratio.csv'), ratio_rows))
    outputs.append(latex_table(os.path.join(paths['tables'], 'E4_distinct_ratio.tex'),
                               [('domain', 'domain'), ('generator', 'generator'),
                                ('median_ratio', 'median distinct / pool size'), ('pools', 'pools')], ratio_rows,
                               'Distinct behaviours per plan in pools of equal size, by domain and generator.',
                               'tab:e4-distinct'))
    curve = []
    for (generator, k), group in sorted(group_by([r for r in rows if r['selector'] == 'bcoverage'],
                                                 ('generator_name', 'k')).items()):
        values = [r['score_bcov'] for r in group]
        curve.append({'generator': generator, 'k': k, 'mean_bcov': sum(values) / len(values), 'pools': len(values)})
    outputs.append(write_csv(os.path.join(paths['results'], 'E4_bcov_by_k.csv'), curve))
    plt = matplotlib_or_none()
    if plt is None:
        notes.append('matplotlib is not installed: the B-Coverage line plot is in E4_bcov_by_k.csv')
    elif curve:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for colour, generator in zip(PALETTE, generators):
            pts = sorted((r['k'], r['mean_bcov']) for r in curve if r['generator'] == generator)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', label=generator, color=colour)
        ax.plot([r['k'] for r in curve], [r['k'] for r in curve], linestyle=':', color='grey', label='k')
        ax.set_xlabel('k'); ax.set_ylabel('B-Coverage after select_coverage'); ax.set_xscale('log'); ax.legend(fontsize=7)
        outputs.append(save_figure(plt, fig, os.path.join(paths['figures'], 'E4_bcov_by_k.pdf')))
    outputs.append(write_manifest(os.path.join(paths['results'], 'E4_manifest.json'), NAME,
                                  config_file, params, started, outputs, notes))
    return outputs, notes
