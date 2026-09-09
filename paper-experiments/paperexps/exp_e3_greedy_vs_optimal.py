"""E3 -- greedy against optimal, on pools with 8 <= b <= 20 distinct
behaviours and k in {3, 4, 5}: the optimum of every indicator by exhaustive
enumeration over subsets of exactly k behaviours, and for B-MaxMin also over
subsets of at most k, which a pair attains.
"""

import itertools
import json
import os

import numpy as np

from harness import INDICATORS, SCORE_OF, Timer, flatten, load_pool, representatives, scores
from model import build, default_dimensions
from report import group_by, latex_table, matplotlib_or_none, save_figure, summary, write_csv, write_manifest

NAME = 'E3'


def optima(distance, k, k_nn):
    """Optimal value of each indicator over subsets of exactly ``k`` of the
    ``b`` behaviours whose distance matrix is ``distance``."""
    b = len(distance)
    combos = np.array(list(itertools.combinations(range(b), k)))
    sub = distance[combos[:, :, None], combos[:, None, :]]           # C x k x k
    rows, cols = np.triu_indices(k, k=1)
    pairs = sub[:, rows, cols]                                        # C x P
    k_prime = min(k_nn, k - 1)
    with_inf = sub.copy()
    with_inf[:, np.arange(k), np.arange(k)] = np.inf
    novelty = np.partition(with_inf, k_prime - 1, axis=2)[:, :, :k_prime].mean(axis=2).mean(axis=1)
    return {'bcoverage': float(k), 'bmaxsum': float(pairs.sum(axis=1).max()),
            'bmaxmin': float(pairs.min(axis=1).max()), 'bnovelty': float(novelty.max())}, len(combos)


def run_task(taskdetails, params):
    pool = load_pool(taskdetails)
    out = {'pool': pool.record(), 'rows': [], 'extra': {}}
    if not pool.plans:
        return out
    counter, out['model'] = build(pool, default_dimensions(pool, params))
    counter.behaviours(pool.plans)
    reps = representatives(counter, pool.plans)
    b = len(reps)
    low, high = params['b-range']
    out['extra']['b'] = b
    if not low <= b <= high:
        out['extra']['skipped'] = f'b = {b} outside [{low}, {high}]'
        return out
    k_nn = params['k-nn']
    distinct = list(reps)
    distance = counter._behaviour_distance_matrix(distinct)
    at_most_k_maxmin = float(distance.max())
    for k in [k for k in params['k-values'] if k <= b]:
        with Timer() as timer:
            best, n_subsets = optima(distance, k, k_nn)
        for indicator in INDICATORS:
            selected = counter.extract(pool.plans, k, indicator=indicator, k_nn=k_nn)
            greedy = scores(counter, selected, k_nn)[SCORE_OF[indicator]]
            optimal = best[indicator]
            ratio = greedy / optimal if optimal > 0 else (1.0 if greedy == optimal else None)
            row = {'k': k, 'indicator': indicator, 'b': b, 'greedy_value': greedy,
                   'optimal_value_exact_k': optimal, 'ratio_exact_k': ratio,
                   'optimal_value_at_most_k': at_most_k_maxmin if indicator == 'bmaxmin' else None,
                   'n_subsets': n_subsets, 'enumeration_cpu_s': timer.cpu,
                   'selected_behaviours': counter.b_coverage(selected)}
            if indicator == 'bmaxmin':
                row['at_most_k_only_below_k'] = at_most_k_maxmin > optimal + 1e-9
            row['below_half'] = indicator in ('bmaxsum', 'bmaxmin') and ratio is not None and ratio < 0.5 - 1e-9
            out['rows'].append(row)
    return out


def report(results, params, paths, config_file, started):
    rows = flatten(results)
    outputs, notes = [], []
    outputs.append(write_csv(os.path.join(paths['results'], 'E3_greedy_vs_optimal.csv'), rows))
    summary_rows = []
    for (indicator, k), group in sorted(group_by(rows, ('indicator', 'k')).items(),
                                        key=lambda item: (INDICATORS.index(item[0][0]), item[0][1])):
        stats = summary([r['ratio_exact_k'] for r in group])
        summary_rows.append({'indicator': indicator, 'k': k, **stats})
    outputs.append(write_csv(os.path.join(paths['results'], 'E3_ratio_summary.csv'), summary_rows))
    outputs.append(latex_table(os.path.join(paths['tables'], 'E3_ratios.tex'),
                               [('indicator', 'indicator'), ('k', '$k$'), ('n', 'pools'), ('min', 'min'),
                                ('p5', '5th pct.'), ('median', 'median')], summary_rows,
                               'Greedy value over the optimum on subsets of exactly $k$ behaviours, on pools '
                               f"with {params['b-range'][0]} to {params['b-range'][1]} distinct behaviours.",
                               'tab:e3-ratios'))
    checks = {
        'pools_in_range': len({r['task_id'] for r in rows}),
        'pools_skipped': sum(1 for r in results if (r.get('extra') or {}).get('skipped')),
        'bcoverage_ratios_all_one': all(abs(r['ratio_exact_k'] - 1.0) < 1e-9 for r in rows if r['indicator'] == 'bcoverage'),
        'below_half': [{'task_id': r['task_id'], 'k': r['k'], 'indicator': r['indicator'], 'ratio': r['ratio_exact_k']}
                       for r in rows if r.get('below_half')],
        'bmaxmin_at_most_k_only_below_k': sum(1 for r in rows if r['indicator'] == 'bmaxmin' and r.get('at_most_k_only_below_k')),
        'bmaxmin_rows': sum(1 for r in rows if r['indicator'] == 'bmaxmin'),
    }
    if not checks['bcoverage_ratios_all_one']:
        notes.append('acceptance check failed: a B-Coverage greedy/optimal ratio differs from 1')
    if checks['below_half']:
        notes.append(f"{len(checks['below_half'])} B-MaxSum/B-MaxMin ratios fall below 1/2 -- flagged in E3_checks.json")
    checks_path = os.path.join(paths['results'], 'E3_checks.json')
    with open(checks_path, 'w') as handle:
        json.dump(checks, handle, indent=2)
    outputs.append(checks_path)
    plt = matplotlib_or_none()
    if plt is None:
        notes.append('matplotlib is not installed: box plots skipped, their data is in the CSV')
    elif rows:
        fig, axes = plt.subplots(1, len(INDICATORS), figsize=(9, 3), sharey=True)
        for ax, indicator in zip(axes, INDICATORS):
            ks = sorted({r['k'] for r in rows if r['indicator'] == indicator})
            data = [[r['ratio_exact_k'] for r in rows if r['indicator'] == indicator and r['k'] == k
                     and r['ratio_exact_k'] is not None] for k in ks]
            ax.boxplot(data, labels=[str(k) for k in ks])
            ax.set_title(indicator)
            ax.set_xlabel('k')
        axes[0].set_ylabel('greedy / optimal')
        outputs.append(save_figure(plt, fig, os.path.join(paths['figures'], 'E3_ratios.pdf')))
    outputs.append(write_manifest(os.path.join(paths['results'], 'E3_manifest.json'), NAME,
                                  config_file, params, started, outputs, notes))
    return outputs, notes
