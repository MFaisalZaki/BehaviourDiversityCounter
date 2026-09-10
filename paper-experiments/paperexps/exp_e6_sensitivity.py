"""E6 -- sensitivity to the model's parameters, on the two-feature model
(subgoal ordering + cost bin): the weight of the first feature, the novelty
neighbourhood and the cost-bin width, each varied around the default; every
setting reruns the four selections and records the indicator values and the
overlap with the default setting's selection.
"""

import json
import os

from harness import (INDICATORS, SCORE_OF, behaviour_set, candidates, flatten, jaccard, load_pool, scores,
                     select_all)
from model import build, default_dimensions, with_weights
from report import group_by, latex_table, write_csv, write_manifest

NAME = 'E6'


def settings(params):
    default = {'weight': params['default-weight'], 'k_nn': params['k-nn'], 'bin_width': params['cost-bin-width']}
    out = [('default', None, default)]
    for w in params['weights']:
        out.append(('weight', w, {**default, 'weight': w}))
    for k_nn in params['k-nn-values']:
        out.append(('k_nn', k_nn, {**default, 'k_nn': k_nn}))
    for width in params['bin-widths']:
        out.append(('bin_width', width, {**default, 'bin_width': width}))
    return out


def run_task(taskdetails, params):
    pool = load_pool(taskdetails)
    out = {'pool': pool.record(), 'rows': [], 'extra': {}}
    k = params['k']
    plans = candidates(pool, params, k)              # N_max = min(10 k, 1000) plans
    out['extra']['pool_size'] = len(plans)
    if len(plans) < k:
        out['extra']['skipped'] = f'pool of {len(plans)} plans is smaller than k = {k}'
        return out
    default_counter, out['model'] = build(pool, default_dimensions(pool, params, with_resources=False))
    default_counter.behaviours(plans)
    default_selection = {}
    for parameter, value, setting in settings(params):
        dimensions = with_weights(default_dimensions(pool, params, bin_width=setting['bin_width'], with_resources=False),
                                  [setting['weight'], 1 - setting['weight']])
        counter, _ = build(pool, dimensions)
        counter.behaviours(plans)
        selected, timing = select_all(counter, plans, k, setting['k_nn'])
        for indicator in INDICATORS:
            chosen = selected[indicator]   # never rebind ``plans``: it is the candidate pool of every setting
            own = scores(counter, chosen, setting['k_nn'])
            # Behaviours under the *default* model, so sets are comparable
            # across bin widths, whose tokens differ.
            under_default = behaviour_set([type('P', (), {'behaviour': b})() for b in default_counter._behaviours_of(chosen)])
            plan_ids = {plan.pool_index for plan in chosen}
            if parameter == 'default':
                default_selection[indicator] = (under_default, plan_ids)
            reference = default_selection[indicator]
            out['rows'].append({
                'parameter': parameter, 'value': value, 'selector': indicator, 'k': k, 'pool_size': len(plans),
                'score': own[SCORE_OF[indicator]], **own,
                'jaccard_to_default': jaccard(under_default, reference[0]),
                'jaccard_plans_to_default': jaccard(plan_ids, reference[1]),
                'behaviours_under_default': under_default,
                'selection_cpu_s': timing[indicator]['cpu_s']})
    return out


def report(results, params, paths, config_file, started):
    rows = flatten(results)
    outputs, notes = [], []
    outputs.append(write_csv(os.path.join(paths['results'], 'E6_sensitivity.csv'),
                             [{k: v for k, v in r.items() if k != 'behaviours_under_default'} for r in rows]))
    table_rows = []
    for (parameter, value, selector), group in sorted(group_by(rows, ('parameter', 'value', 'selector')).items(),
                                                       key=lambda item: (item[0][0], str(item[0][1]), INDICATORS.index(item[0][2]))):
        table_rows.append({'parameter': parameter, 'value': value, 'selector': selector,
                           'mean_score': sum(r['score'] for r in group) / len(group),
                           'mean_jaccard': sum(r['jaccard_to_default'] for r in group) / len(group),
                           'pools': len(group)})
    outputs.append(write_csv(os.path.join(paths['results'], 'E6_summary.csv'), table_rows))
    outputs.append(latex_table(os.path.join(paths['tables'], 'E6_sensitivity.tex'),
                               [('parameter', 'parameter'), ('value', 'value'), ('selector', 'selection'),
                                ('mean_score', 'mean own score'), ('mean_jaccard', 'mean Jaccard to default'),
                                ('pools', 'pools')], table_rows,
                               f"Sensitivity of the selections to the model's parameters at $k = {params['k']}$: "
                               'the mean value of the selection\'s own indicator and the mean Jaccard overlap '
                               'between its behaviour set and the default setting\'s.', 'tab:e6-sensitivity'))
    violations = []
    for task_id, group in group_by([r for r in rows if r['selector'] == 'bcoverage'
                                    and r['parameter'] in ('default', 'weight')], ('task_id',)).items():
        sets = {tuple(r['behaviours_under_default']) for r in group}
        values = {r['score_bcov'] for r in group}
        if len(sets) > 1 or len(values) > 1:
            violations.append(task_id[0])
    checks = {'weight_settings_leave_coverage_unchanged': not violations, 'violations': violations,
              'pools': len({r['task_id'] for r in rows})}
    if violations:
        notes.append(f'acceptance check failed: select_coverage changed under the weight settings on {len(violations)} pools')
    if rows and all(r['q'] == 1.0 for r in rows):
        notes.append('every pool is at q = 1.0, where the cost-bin feature is constant: the bin-width settings cannot change any selection')
    checks_path = os.path.join(paths['results'], 'E6_checks.json')
    with open(checks_path, 'w') as handle:
        json.dump(checks, handle, indent=2)
    outputs.append(checks_path)
    outputs.append(write_manifest(os.path.join(paths['results'], 'E6_manifest.json'), NAME,
                                  config_file, params, started, outputs, notes))
    return outputs, notes
