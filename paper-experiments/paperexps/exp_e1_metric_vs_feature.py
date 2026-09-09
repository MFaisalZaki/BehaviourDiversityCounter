"""E1 -- metric-based against feature-based diversity.

On every pool, for every k: the three plan-level greedy selections and the
four behaviour-space selections, every returned set scored under all seven
measures, plus the fraction of distinct behaviours and the mean cost ratio.
"""

import os

from baseline import METRICS, PlanMetric, greedy_select, plan_set_score
from harness import (INDICATORS, Timer, coverage_rows, flatten, load_pool, mean_cost_ratio,
                     scores, select_all, selection_k_values)
from model import build, default_dimensions
from report import (group_by, latex_table, matplotlib_or_none, matrix_table, paired_wilcoxon_rows,
                    relative_score_matrix, save_figure, write_csv, write_manifest, PALETTE)

NAME = 'E1'
SELECTORS = METRICS + INDICATORS
SCORE_COLUMNS = ('score_stability', 'score_state', 'score_uniqueness',
                 'score_bcov', 'score_bmaxsum', 'score_bmaxmin', 'score_bnov')


def run_task(taskdetails, params):
    pool = load_pool(taskdetails)
    out = {'pool': pool.record(), 'rows': []}
    if not pool.plans:
        return out
    counter, out['model'] = build(pool, default_dimensions(pool, params))
    with Timer() as mapping:
        counter.behaviours(pool.plans)
    out['mapping'] = {'cpu_s': mapping.cpu, 'wall_s': mapping.wall}
    k_nn = params['k-nn']

    with Timer() as timer:
        matrices = {name: PlanMetric(name, pool.plans, trace=pool.trace).matrix() for name in METRICS}
    out['plan_metric_matrices'] = {'cpu_s': timer.cpu, 'wall_s': timer.wall}

    for k in selection_k_values(params, len(pool.plans)):
        selected, timing = {}, {}
        for name in METRICS:
            with Timer() as timer:
                indices = greedy_select(matrices[name], k)
            selected[name] = [pool.plans[i] for i in indices]
            timing[name] = {'cpu_s': timer.cpu, 'wall_s': timer.wall}
        bspace, bspace_timing = select_all(counter, pool.plans, k, k_nn)
        selected.update(bspace)
        timing.update(bspace_timing)
        for selector in SELECTORS:
            plans = selected[selector]
            indices = [pool.index[id(plan)] for plan in plans]
            row = {'k': k, 'selector': selector, 'n_selected': len(plans)}
            for name in METRICS:
                row[f'score_{name}'] = plan_set_score(matrices[name], indices)
            row.update(scores(counter, plans, k_nn))
            row['distinct_fraction'] = row['score_bcov'] / k
            row['mean_cost_ratio'] = mean_cost_ratio(plans, pool.optimal_cost)
            row['selection_cpu_s'] = timing[selector]['cpu_s']
            out['rows'].append(row)
    return out


def report(results, params, paths, config_file, started):
    rows = flatten(results)
    outputs, notes = [], []
    outputs.append(write_csv(os.path.join(paths['results'], 'E1_metric_vs_feature.csv'), rows))
    outputs.append(write_csv(os.path.join(paths['results'], 'E1_coverage.csv'), coverage_rows(results)))

    headers = {'score_stability': 'stability', 'score_state': 'state', 'score_uniqueness': 'uniqueness',
               'score_bcov': 'B-Cov', 'score_bmaxsum': 'B-MaxSum', 'score_bmaxmin': 'B-MaxMin',
               'score_bnov': 'B-Nov'}
    matrix, counts = relative_score_matrix(rows, ('task_id', 'k'), 'selector', SELECTORS, SCORE_COLUMNS)
    outputs.append(matrix_table(os.path.join(paths['tables'], 'E1_relative_scores.tex'), matrix,
                                SELECTORS, SCORE_COLUMNS,
                                'Mean relative score of each selector (rows) under each measure '
                                '(columns), normalised per instance and $k$ by the best selector; '
                                f'{min(counts.values()) if counts else 0}--{max(counts.values()) if counts else 0} '
                                '(instance, $k$) pairs per column.', 'tab:e1-relative', headers))
    for k, group in sorted(group_by(rows, ('k',)).items()):
        matrix_k, counts_k = relative_score_matrix(group, ('task_id',), 'selector', SELECTORS, SCORE_COLUMNS)
        outputs.append(matrix_table(os.path.join(paths['tables'], f'E1_relative_scores_k{k[0]}.tex'), matrix_k,
                                    SELECTORS, SCORE_COLUMNS,
                                    f'As the main table, for $k = {k[0]}$ ({max(counts_k.values()) if counts_k else 0} instances).',
                                    f'tab:e1-relative-k{k[0]}', headers))

    comparisons = [(f'q={key[0]}, k={key[1]}', group) for key, group in sorted(group_by(rows, ('q', 'k')).items())]
    tests = paired_wilcoxon_rows(rows, ('task_id',), 'selector', 'stability', 'bcoverage',
                                 'distinct_fraction', comparisons)
    outputs.append(write_csv(os.path.join(paths['results'], 'E1_distinct_fraction_wilcoxon.csv'), tests))
    outputs.append(latex_table(os.path.join(paths['tables'], 'E1_distinct_fraction_wilcoxon.tex'),
                               [('comparison', 'pools'), ('n', '$n$'), ('median_diff', 'median diff.'),
                                ('iqr_diff', 'IQR'), ('p', '$p$'), ('p_holm', '$p$ (Holm)')], tests,
                               'Distinct fraction of the stability selection minus that of '
                               'select\\_coverage, paired by instance: two-sided Wilcoxon signed-rank '
                               'test, Holm-corrected across the rows.', 'tab:e1-wilcoxon'))

    scatter = params.get('scatter', {'k': 10, 'q': 2.0})
    points = [r for r in rows if r['k'] == scatter['k'] and r['q'] == scatter['q']]
    if not points:
        available = sorted({(r['q'], r['k']) for r in rows})
        fallback = next(((q, k) for q, k in available if k == scatter['k']), available[0] if available else None)
        if fallback is not None:
            notes.append(f'no rows at q={scatter["q"]}, k={scatter["k"]} for the scatter; drew q={fallback[0]}, k={fallback[1]} instead')
            points = [r for r in rows if r['k'] == fallback[1] and r['q'] == fallback[0]]
    scatter_rows = [{'task_id': r['task_id'], 'selector': r['selector'], 'q': r['q'], 'k': r['k'],
                     'score_stability': r['score_stability'], 'bcov_over_k': r['score_bcov'] / r['k']}
                    for r in points]
    outputs.append(write_csv(os.path.join(paths['results'], 'E1_scatter.csv'), scatter_rows))
    plt = matplotlib_or_none()
    if plt is None:
        notes.append('matplotlib is not installed: figures skipped, their data is in the CSVs')
    elif scatter_rows:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        for colour, selector in zip(PALETTE, SELECTORS):
            pts = [r for r in scatter_rows if r['selector'] == selector]
            ax.scatter([r['score_stability'] for r in pts], [r['bcov_over_k'] for r in pts],
                       s=12, alpha=0.7, label=selector, color=colour)
        ax.set_xlabel('stability score of the selected set')
        ax.set_ylabel('B-Coverage / k')
        ax.legend(fontsize=7)
        outputs.append(save_figure(plt, fig, os.path.join(paths['figures'], 'E1_scatter.pdf')))
    outputs.append(write_manifest(os.path.join(paths['results'], 'E1_manifest.json'), NAME,
                                  config_file, params, started, outputs, notes))
    return outputs, notes
