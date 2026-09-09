"""E0 -- the case study: the four indicators on one real Rovers instance under
the paper's own model (number of rovers used + subgoal ordering, weights 1/2).

Every Rovers pool at the case-study q is a candidate task; each records
whether it is eligible (at least ``min-rovers`` rovers, ``min-goals`` goal
atoms and ``min-distinct`` distinct behaviours) and its four selections at
k = 3.  The report picks the smallest eligible instance.
"""

import json
import os

from harness import INDICATORS, Timer, load_pool, plan_record, scores, select_all
from model import build, rovers_dimensions
from report import latex_table, write_manifest

NAME = 'E0'


def run_task(taskdetails, params):
    pool = load_pool(taskdetails)
    out = {'pool': pool.record(), 'rows': [], 'extra': {'eligible': False}}
    if not pool.plans or not pool.info['resources-file']:
        out['extra']['reason'] = 'empty pool' if not pool.plans else 'no rover declarations'
        return out
    counter, model = build(pool, rovers_dimensions(pool, params))
    out['model'] = model
    with Timer() as mapping:
        counter.behaviours(pool.plans)
    k, k_nn = params['k'], params['k-nn']
    extra = out['extra']
    extra.update({
        'num_rovers': len(model['agents']),
        'num_goals': model['goal-atoms'],
        'num_objects': len(pool.task.all_objects),
        'distinct': counter.b_coverage(pool.plans),
        'pool_size': len(pool.plans),
        'mapping_cpu_s': mapping.cpu,
    })
    extra['eligible'] = (extra['num_rovers'] >= params['min-rovers']
                         and extra['num_goals'] >= params['min-goals']
                         and extra['distinct'] >= params['min-distinct']
                         and extra['pool_size'] >= k)
    selected, timing = select_all(counter, pool.plans, k, k_nn)
    for indicator in INDICATORS:
        plans = selected[indicator]
        out['rows'].append({'k': k, 'selector': indicator, **scores(counter, plans, k_nn),
                            'plans': [plan_record(plan) for plan in plans],
                            'selection_cpu_s': timing[indicator]['cpu_s']})
    out['extra']['pool_scores'] = scores(counter, pool.plans, k_nn)
    return out


def report(results, params, paths, config_file, started):
    eligible = [r for r in results if not r.get('error') and r.get('extra', {}).get('eligible')]
    chosen = min(eligible, key=lambda r: (r['extra']['num_objects'], r['task']['inst'],
                                          r['task']['year']), default=None)
    notes = []
    if chosen is None:
        notes.append(f'no eligible Rovers instance among {len(results)} candidate pools '
                     f'(need >= {params["min-rovers"]} rovers, >= {params["min-goals"]} goal atoms, '
                     f'>= {params["min-distinct"]} distinct behaviours at q = {params.get("pool-q-values")})')
    out_json = os.path.join(paths['results'], 'E0_case_study.json')
    os.makedirs(paths['results'], exist_ok=True)
    with open(out_json, 'w') as handle:
        json.dump({'chosen': chosen, 'candidates': [
            {'task_id': r['task']['task_id'], **{k: v for k, v in r.get('extra', {}).items() if k != 'pool_scores'}}
            for r in results if not r.get('error')],
            'errors': [r['task']['task_id'] for r in results if r.get('error')]}, handle, indent=2)
    outputs = [out_json]
    if chosen is not None:
        rows = []
        for row in chosen['rows']:
            for position, plan in enumerate(row['plans']):
                rows.append({'selector': row['selector'] if position == 0 else '',
                             'plan': position + 1, 'behaviour': plan['behaviour'], 'cost': plan['cost'],
                             'bcov': row['score_bcov'] if position == 0 else None,
                             'bmaxsum': row['score_bmaxsum'] if position == 0 else None,
                             'bmaxmin': row['score_bmaxmin'] if position == 0 else None,
                             'bnov': row['score_bnov'] if position == 0 else None})
        table = latex_table(
            os.path.join(paths['tables'], 'E0_case_study.tex'),
            [('selector', 'selection'), ('plan', r'\#'), ('behaviour', 'behaviour'), ('cost', 'cost'),
             ('bcov', 'B-Cov'), ('bmaxsum', 'B-MaxSum'), ('bmaxmin', 'B-MaxMin'), ('bnov', 'B-Nov')],
            rows, f"Case study on {chosen['task']['task_id']} ({chosen['extra']['num_rovers']} rovers, "
                  f"{chosen['extra']['num_goals']} goal atoms, {chosen['extra']['distinct']} distinct "
                  f"behaviours in a pool of {chosen['extra']['pool_size']}), k = {params['k']}.",
            'tab:e0-case-study')
        outputs.append(table)
    outputs.append(write_manifest(os.path.join(paths['results'], 'E0_manifest.json'), NAME,
                                  config_file, params, started, outputs, notes))
    return outputs, notes
