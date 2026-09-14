"""E1 -- the case study: reading the differences componentwise, and what a
plan-level distance sees and misses on the same plans.

One instance per configured domain under the domain's own model, chosen from
the selection results by ``RULE``. Everything is read off that pool's
selection results and behaviour dump: the behaviours the pool exhibits and how
many plans each holds (twinning in practice), the plans the four selections
return at K, the four indicators of each returned set, and every returned pair
feature by feature. The stability reading takes the set B-MaxSum selects under
the stability model on the same pool -- the post-hoc selection of Katz and
Sohrabi (2020) -- and reads it under the domain's model, and reads each
domain-model selection by its pairwise stability values, recomputed from the
action strings the results hold. The report states the numbers and ranks
neither model.
"""

import json
from itertools import combinations

from bdc_experiments import models, reference, runner
from bdc_experiments import report as rp
from bdc_experiments.config import results_root

#: The case study's grid, fixed by the paper.
K, KAPPA, Q = 3, 1, 2.0

RULE = ("per domain, the smallest instance (smallest pool size, then the instance id) whose pool "
        "at q = 2.0 exhibits at least min_behaviours behaviours under the domain's model and at "
        "least two distinct values on the model's agents feature; failing that, the smallest with "
        'min_behaviours behaviours alone; failing that, the pool with the most behaviours')

CLAUSES = ('the behaviour count and a varying agents feature', 'the behaviour count alone',
           'neither clause: the richest pool available, below min_behaviours')

BASE = list(runner.BASE_FIELDS)


def _survey(results):
    """Every candidate pool with the two numbers the rule reads."""
    rows = []
    for result in results:
        dump = result['dump']
        key = next((f for f in dump['features'] if f in ('rn', 'ru', 'rc')), dump['features'][0])
        column = dump['features'].index(key)
        rows.append({'domain': result['pool']['domain'], 'instance': result['pool']['instance'],
                     'pool_stem': result['pool']['pool_stem'], 'pool_size': result['pool']['size'],
                     'b': len(dump['distinct']), 'agents_feature': key,
                     'agents_values': len({tuple(t)[column] for t in dump['distinct']}),
                     'task_id': result['task_id']})
    return rows


def _choose(cfg, survey):
    """The rule on one domain's candidates, and which of its clauses decided."""
    order = lambda row: (row['pool_size'], row['instance'])
    enough = [row for row in survey if row['b'] >= cfg['e1']['min_behaviours']]
    varying = [row for row in enough if row['agents_values'] >= 2]
    for candidates, clause in ((varying, CLAUSES[0]), (enough, CLAUSES[1])):
        if candidates:
            return min(candidates, key=order), clause
    return (min(survey, key=lambda row: (-row['b'], *order(row))), CLAUSES[2]) if survey else (None, None)


def _components(features, values):
    return {f'f_{key}': value for key, value in zip(features, values)}


def _stability(actions_of_plans):
    """Every unordered pair's stability distance, by position in the set."""
    return {(i, j): reference.ref_stability(actions_of_plans[i], actions_of_plans[j])
            for i, j in combinations(range(len(actions_of_plans)), 2)}


def _rows(result, katz):
    """Behaviour, selection, pair and stability rows for one instance."""
    dump, features, matrix = result['dump'], result['dump']['features'], result['dump']['matrix']
    rows, base = [], rp.head(result, K, KAPPA)
    for index, values in enumerate(dump['distinct']):
        shown = [plan for plan in dump['plans'] if plan['distinct'] == index]
        rows.append({**rp.head(result), 'kind': 'behaviour', 'distinct': index,
                     'plans': len(shown), 'cheapest_cost': min(p['cost'] for p in shown),
                     **_components(features, values)})
    # The four domain-model selections, and the stability selection read under
    # the domain model: every set is indexed into the same cost-sorted pool.
    sets = {indicator: rp.entry(result, indicator, KAPPA) for indicator in runner.INDICATORS}
    sets['stability'] = rp.entry(katz, 'bmaxsum', KAPPA)
    for selector, entry in sets.items():
        indices = entry['indices'][:K]
        chosen = [dump['plans'][i]['distinct'] for i in indices]
        values = {f'set_{n}': v for n, v in rp.score(dump, chosen, KAPPA).items()}
        for position, index in enumerate(indices):
            rows.append({**base, 'kind': 'selection', 'selector': selector, 'position': position,
                         'plan': index, 'cost': dump['plans'][index]['cost'], 'distinct': chosen[position],
                         **_components(features, dump['plans'][index]['behaviour']), **values})
        stability = _stability(entry['actions'][:K])
        for i, j in combinations(range(len(indices)), 2):
            left, right = dump['distinct'][chosen[i]], dump['distinct'][chosen[j]]
            differing = [key for key, a, b in zip(features, left, right) if a != b]
            rows += [{**base, 'kind': 'pair', 'selector': selector, 'plan_i': indices[i],
                      'plan_j': indices[j], 'feature': key, 'value_i': a, 'value_j': b,
                      'differs': a != b, 'psi': matrix[chosen[i]][chosen[j]],
                      'differing_features': ' '.join(differing)}
                     for key, a, b in zip(features, left, right)]
            rows.append({**base, 'kind': 'stability', 'selector': selector, 'plan_i': indices[i],
                         'plan_j': indices[j], 'stability': stability[(i, j)],
                         'same_behaviour': chosen[i] == chosen[j], 'psi': matrix[chosen[i]][chosen[j]],
                         'covered': len(set(chosen)), 'attainable': min(K, len(dump['distinct']))})
    return rows


def _fullest_cell(cfg, result, rows):
    """Two plans of the behaviour most plans share, with their action strings:
    what the model's features do not record."""
    fullest = max((r for r in rows if r['kind'] == 'behaviour'), key=lambda r: r['plans'])
    pool = result['pool']
    path = (results_root(cfg) / 'pools' / pool['domain'] / pool['instance'].split('/')[-1]
            / f"{pool['pool_stem']}.json")
    plans = json.loads(path.read_text())['plans'] if path.is_file() else []
    members = [p for p in result['dump']['plans'] if p['distinct'] == fullest['distinct']][:2]
    return fullest, [(p['index'], plans[p['original_index']]['actions']) for p in members if plans]


def _note(cfg, cases):
    lines = ['# E1 -- reading the differences componentwise', '',
             f'Instance-selection rule: {RULE}.', '',
             f"q = {Q}, k = {K}, kappa = {KAPPA}, min_behaviours = {cfg['e1']['min_behaviours']}. "
             f'{rp.TIE_RULE}']
    for domain, (survey, chosen, clause, result, rows) in cases.items():
        lines += ['', f'## {domain}', '', f'Pools surveyed: {len(survey)}.', '',
                  '| instance | pool | pool_size | b | distinct agents values |',
                  '| --- | --- | --- | --- | --- |']
        lines += [f"| {r['instance']} | {r['pool_stem']} | {r['pool_size']} | {r['b']} "
                  f"| {r['agents_values']} |" for r in sorted(survey, key=lambda r: (r['pool_size'], r['instance']))]
        feature = chosen['agents_feature']
        lines += ['', f"Chosen: **{chosen['instance']}**, pool `{chosen['pool_stem']}` (pool_size "
                      f"{chosen['pool_size']}, b = {chosen['b']} under `{result['model']['name']}`); "
                      f'the clause that decided: {clause}.',
                  f'The `{feature}` feature is '
                  + ('CONSTANT here: it takes one value, so no returned pair can differ on it.'
                     if chosen['agents_values'] < 2 else
                     f"not constant: it takes {chosen['agents_values']} values."),
                  'Model: ' + ', '.join(f"{f['key']} (weight {f['weight']}, {f['params']})"
                                        for f in result['model']['features'])]
        fullest, shown = _fullest_cell(cfg, result, rows)
        lines += ['', f"The fullest behaviour (id {fullest['distinct']}) holds {fullest['plans']} of "
                      f"the pool's {chosen['pool_size']} plans; two of them, which the model treats "
                      'as the same:']
        for index, actions in shown:
            lines += ['', f'Plan {index}:', '```'] + actions + ['```']
        katz = [r for r in rows if r['kind'] == 'stability' and r['selector'] == 'stability']
        lines += ['', f"The stability selection covers {katz[0]['covered']} of the "
                      f"{katz[0]['attainable']} behaviours attainable under the domain's model, and "
                      f"{sum(r['same_behaviour'] for r in katz)} of its {len(katz)} pairs share a "
                      'behaviour.' if katz else '']
    return '\n'.join(lines) + '\n'


def report(cfg):
    kept = {models.domain_model(cfg, d).name for d in cfg['e1']['domains']} | {'stability'}
    pools = rp.by_pool([r for r in rp.selections(cfg, keep=lambda name: name in kept)
                        if float(r['pool']['q']) == Q])
    cases = {}
    for domain in cfg['e1']['domains']:
        model = models.domain_model(cfg, domain).name
        results = [found[model] for (instance, _), found in pools.items()
                   if instance.split('/')[0] == domain and model in found and 'stability' in found]
        survey = _survey(results)
        chosen, clause = _choose(cfg, survey)
        if chosen is None:
            raise SystemExit(f'E1: no selection result for {domain} at q = {Q} under {model} and '
                             'stability; run the select tasks first')
        result = next(r for r in results if r['task_id'] == chosen['task_id'])
        katz = pools[(result['pool']['instance'], result['pool']['pool_stem'])]['stability']
        cases[domain] = (survey, chosen, clause, result, _rows(result, katz))

    out = rp.report_dir(cfg, 'e1')
    rows = [row for *_, case_rows in cases.values() for row in case_rows]
    kinds = {kind: [r for r in rows if r['kind'] == kind]
             for kind in ('behaviour', 'selection', 'pair', 'stability')}
    features = sorted({key for r in rows for key in r if key.startswith('f_')})
    written = [
        rp.write_csv(out / 'e1_behaviours.csv', kinds['behaviour'],
                     BASE + ['distinct', 'plans', 'cheapest_cost'] + features),
        rp.write_csv(out / 'e1_selections.csv', kinds['selection'],
                     BASE + ['selector', 'position', 'plan', 'cost', 'distinct'] + features
                     + [f'set_{n}' for n in runner.INDICATORS]),
        rp.write_csv(out / 'e1_pairwise_diffs.csv', kinds['pair'],
                     BASE + ['selector', 'plan_i', 'plan_j', 'feature', 'value_i', 'value_j',
                             'differs', 'psi', 'differing_features']),
        rp.write_csv(out / 'e1_stability.csv', kinds['stability'],
                     BASE + ['selector', 'plan_i', 'plan_j', 'stability', 'same_behaviour', 'psi',
                             'covered', 'attainable']),
    ]
    for domain, (survey, chosen, clause, result, case_rows) in cases.items():
        columns = [f'f_{key}' for key in result['dump']['features']]
        tuple_of = lambda row: '; '.join(f'{key[2:]}={row[key]}' for key in columns)
        selections = rp.group([r for r in case_rows if r['kind'] == 'selection'], ('selector',))
        pairs = [r for r in case_rows if r['kind'] == 'pair']
        stability = {(r['selector'], r['plan_i'], r['plan_j']): r['stability']
                     for r in case_rows if r['kind'] == 'stability'}
        written += [
            rp.table(out / 'tables' / f'e1_behaviours_{domain}.tex', f'tab:e1-behaviours-{domain}',
                     f'The behaviours the chosen {domain} pool exhibits under its model, '
                     'componentwise, with how many plans exhibit each and the cheapest such plan.',
                     ['id', 'behaviour', 'plans', 'cheapest cost'],
                     [[r['distinct'], tuple_of(r), r['plans'], r['cheapest_cost']]
                      for r in case_rows if r['kind'] == 'behaviour'], digits=0, aligns='rlrr'),
            rp.table(out / 'tables' / f'e1_selections_{domain}.tex', f'tab:e1-selections-{domain}',
                     f'The plans each selection returns on the chosen {domain} instance, and the '
                     "four indicators of the returned set under the domain's model. The stability "
                     'row is the set B-MaxSum selects under the stability model, read under the '
                     "domain's model. " + rp.TIE_RULE,
                     ['selection', 'plans', 'costs', 'B-Coverage', 'B-MaxSum', 'B-MaxMin', 'B-Novelty'],
                     [[selector, ' '.join(str(r['plan']) for r in group),
                       ' '.join(str(r['cost']) for r in group)]
                      + [group[0][f'set_{n}'] for n in runner.INDICATORS]
                      for (selector,), group in selections.items()], aligns='lll' + 'r' * 4),
            rp.table(out / 'tables' / f'e1_pairwise_{domain}.tex', f'tab:e1-pairwise-{domain}',
                     f'Every pair of returned plans on the chosen {domain} instance: the features on '
                     "which the two differ, their dissimilarity under the domain's model, and their "
                     'stability distance.',
                     ['selection', 'plan i', 'plan j', 'differing features', 'psi', 'stability'],
                     [[r['selector'], r['plan_i'], r['plan_j'], r['differing_features'] or '(none)',
                       r['psi'], stability[(r['selector'], r['plan_i'], r['plan_j'])]]
                      for r in pairs if r['feature'] == result['dump']['features'][0]],
                     aligns='lrrlrr'),
        ]
    note = out / 'e1_note.md'
    note.write_text(_note(cfg, cases))
    written.append(note)
    written.append(rp.manifest(cfg, 'e1', written, [r for f in pools.values() for r in f.values()], extra={
        'rule': RULE, 'k': K, 'kappa': KAPPA, 'q': Q,
        'chosen': {d: {'instance': c['instance'], 'pool_stem': c['pool_stem'], 'clause': clause,
                       'b': c['b'], 'pool_size': c['pool_size'], 'agents_values': c['agents_values']}
                   for d, (_, c, clause, _, _) in cases.items()},
        'survey': {d: s for d, (s, *_) in cases.items()},
        'stability_reading': 'the stability selection is B-MaxSum under the stability model, read '
                             "under the domain's model; every stability value is recomputed from "
                             'the action strings of the selected plans'}))
    return written
