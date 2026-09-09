"""E2 -- cross-evaluation of the four indicators.

(a) each selection scored under all four indicators; (b) random subsets of k
distinct behaviours scored under all four, and Kendall's tau between every
pair of indicators over those subsets.
"""

import itertools
import math
import os
import random

from harness import (INDICATORS, Timer, flatten, load_pool, representatives, scores, select_all,
                     selection_k_values)
from model import build, default_dimensions
from report import group_by, latex_table, matrix_table, relative_score_matrix, write_csv, write_manifest
from stats import kendall_tau, median

NAME = 'E2'
SCORE_COLUMNS = ('score_bcov', 'score_bmaxsum', 'score_bmaxmin', 'score_bnov')
PAIRS = list(itertools.combinations(SCORE_COLUMNS, 2))


def random_subsets(population, size, count, rng):
    """``count`` distinct subsets of ``size`` elements -- every one when there
    are no more than ``count``."""
    if size >= len(population):
        return [tuple(population)]
    total = math.comb(len(population), size)
    if total <= count:
        return list(itertools.combinations(population, size))
    drawn = set()
    while len(drawn) < count:
        drawn.add(tuple(sorted(rng.sample(range(len(population)), size))))
    return [tuple(population[i] for i in subset) for subset in sorted(drawn)]


def run_task(taskdetails, params):
    pool = load_pool(taskdetails)
    out = {'pool': pool.record(), 'rows': [], 'rows_subsets': []}
    if not pool.plans:
        return out
    counter, out['model'] = build(pool, default_dimensions(pool, params))
    with Timer() as mapping:
        counter.behaviours(pool.plans)
    out['mapping'] = {'cpu_s': mapping.cpu, 'wall_s': mapping.wall}
    k_nn = params['k-nn']
    reps = representatives(counter, pool.plans)
    behaviours = sorted(reps)
    rng = random.Random(f"{params['seed']}:{taskdetails['task_id']}")

    for k in selection_k_values(params, len(pool.plans)):
        selected, timing = select_all(counter, pool.plans, k, k_nn)
        for indicator in INDICATORS:
            out['rows'].append({'k': k, 'selector': indicator, **scores(counter, selected[indicator], k_nn),
                                'selection_cpu_s': timing[indicator]['cpu_s']})
        subsets = random_subsets(behaviours, min(k, len(behaviours)), params['random-subsets'], rng)
        scored = [scores(counter, [reps[b] for b in subset], k_nn) for subset in subsets]
        for first, second in PAIRS:
            out['rows_subsets'].append({
                'k': k, 'pair': f'{first[6:]}/{second[6:]}', 'n_subsets': len(subsets),
                'b': len(behaviours),
                'kendall_tau': kendall_tau([s[first] for s in scored], [s[second] for s in scored])})
    return out


def report(results, params, paths, config_file, started):
    rows, subset_rows = flatten(results), flatten(results, 'rows_subsets')
    outputs, notes = [], []
    outputs.append(write_csv(os.path.join(paths['results'], 'E2_cross_eval.csv'), rows))
    outputs.append(write_csv(os.path.join(paths['results'], 'E2_kendall.csv'), subset_rows))
    headers = {'score_bcov': 'B-Cov', 'score_bmaxsum': 'B-MaxSum', 'score_bmaxmin': 'B-MaxMin', 'score_bnov': 'B-Nov'}
    matrix, counts = relative_score_matrix(rows, ('task_id', 'k'), 'selector', INDICATORS, SCORE_COLUMNS)
    outputs.append(matrix_table(os.path.join(paths['tables'], 'E2_relative_scores.tex'), matrix, INDICATORS,
                                SCORE_COLUMNS, 'Mean relative score of each selection (rows) under each '
                                'indicator (columns), normalised per instance and $k$ by the best selection; '
                                f'{max(counts.values()) if counts else 0} (instance, $k$) pairs.',
                                'tab:e2-relative', headers))
    flat = all(v is not None and v > 0.95 for sel in matrix.values() for v in sel.values())
    if flat and rows:
        notes.append('the 4x4 relative-score matrix is nearly flat (every entry above 0.95)')

    tau_rows = []
    for (k, pair), group in sorted(group_by(subset_rows, ('k', 'pair')).items()):
        taus = [r['kendall_tau'] for r in group if r['kendall_tau'] is not None]
        tau_rows.append({'k': k, 'pair': pair, 'median_tau': median(taus), 'n_pools': len(group),
                         'n_defined': len(taus)})
    outputs.append(write_csv(os.path.join(paths['results'], 'E2_kendall_summary.csv'), tau_rows))
    outputs.append(latex_table(os.path.join(paths['tables'], 'E2_kendall.tex'),
                               [('k', '$k$'), ('pair', 'indicator pair'), ('median_tau', r'median $\tau$'),
                                ('n_defined', 'pools')], tau_rows,
                               f"Median Kendall's $\\tau$ between indicator pairs over up to "
                               f"{params['random-subsets']} random subsets of $k$ distinct behaviours per pool.",
                               'tab:e2-kendall'))
    outputs.append(write_manifest(os.path.join(paths['results'], 'E2_manifest.json'), NAME,
                                  config_file, params, started, outputs, notes))
    return outputs, notes
