"""E1 -- the case study behind claim C1: two behaviours are compared
componentwise, so the features on which two plans differ can be read off.

One instance, picked from the pools by the rule in ``RULE``, read under the two
domain-specific models of its domain: the pool's behaviours, the four
selections at the configured k, and every returned pair feature by feature.
"""

from itertools import combinations
from pathlib import Path

from bdc_experiments import models, pools, report as reports, runner

#: Quoted verbatim in the note, which has to say why this instance and no other.
RULE = ('the smallest instance (smallest pool size, then the instance id) whose pool exhibits '
        'at least min_behaviours behaviours under the first model and at least two distinct '
        'values on that model resource feature; failing that, the smallest satisfying the '
        'behaviour count alone')

BASE = ['instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b']

#: One CSV per kind of row, and its columns before and after the feature columns.
OUTPUTS = {
    'behaviour': ('e1_behaviours.csv', BASE, ['distinct', 'plans', 'cheapest_cost']),
    'selection': ('e1_selections.csv', BASE + ['indicator', 'position', 'plan', 'cost', 'distinct'],
                  [f'set_{name}' for name in runner.INDICATORS] + ['wall_s', 'cpu_s']),
    'pair': ('e1_pairwise_diffs.csv',
             BASE + ['indicator', 'plan_i', 'plan_j', 'feature', 'value_i', 'value_j'],
             ['differs', 'contribution', 'psi', 'differing_features']),
}


def _specs(cfg):
    """The domain-specific models of the configured domain, in registry order.
    The generic control is not part of E1."""
    chosen = [s for s in models.models_for(cfg, cfg['e1']['domain']) if s.name != 'generic']
    if not chosen:
        raise ValueError(f"no domain-specific model for domain '{cfg['e1']['domain']}'")
    return chosen[:2]


def _survey(cfg):
    """Every pool of the configured domain and q, with the two numbers the rule
    reads: b, and how many values the first model's resource feature takes."""
    spec = _specs(cfg)[0]
    key = spec.features[0].key      # 'rn', the rover-count feature, on rovers
    found = []
    for path in pools.pool_files(cfg):
        pool = pools.read_pool(path)
        if pool['domain'] != cfg['e1']['domain'] or float(pool['q']) != float(cfg['e1']['q']):
            continue
        ctx = runner.context(cfg, f"e1/{pool['instance']}/{path.stem}")
        dump = runner.load_dump(cfg, models.model_hash(spec), pool['domain'],
                                Path(pool['instance']).name, path.stem)
        if dump is None:
            dump = runner.setup(cfg, ctx, spec)[4]
        column = dump['features'].index(key)
        found.append({'instance': pool['instance'], 'task_id': ctx['task_id'],
                      'pool_stem': path.stem, 'model': spec.name, 'resource_feature': key,
                      'pool_size': len(dump['plans']), 'b': len(dump['distinct']),
                      'resource_values': len({b[column] for b in dump['distinct']})})
    return found


def _choose(cfg, found):
    """The rule, and which of its clauses decided."""
    order = lambda row: (row['pool_size'], row['instance'])
    enough = [row for row in found if row['b'] >= cfg['e1']['min_behaviours']]
    varying = [row for row in enough if row['resource_values'] >= 2]
    if varying:
        return min(varying, key=order), 'behaviour count and a varying resource feature'
    if enough:
        return min(enough, key=order), 'the behaviour count alone'
    if found:
        # No clause holds: the richest pool is still the best case study there is,
        # and the note says the behaviour count was not met.
        return sorted(found, key=lambda row: (-row['b'], *order(row)))[0], 'no clause applied'
    return None, 'no pool of the configured domain and q'


def tasks(cfg):
    """One task: the chosen instance, run under both models."""
    chosen = _choose(cfg, _survey(cfg))[0]
    return [chosen['task_id']] if chosen else []


def _components(features, values):
    """A behaviour tuple as one column per feature, so two models with different
    features share one CSV."""
    return {f'f_{key}': value for key, value in zip(features, values)}


def _behaviour_rows(base, dump):
    """(a) every distinct behaviour, componentwise, with its plan count and its
    cheapest plan."""
    rows = []
    for index, values in enumerate(dump['distinct']):
        shown = [plan for plan in dump['plans'] if plan['distinct'] == index]
        rows.append({**base, 'kind': 'behaviour', 'distinct': index, 'plans': len(shown),
                     'cheapest_cost': min(plan['cost'] for plan in shown),
                     **_components(dump['features'], values)})
    return rows


def _selection_rows(base, dump, indicator, entry):
    """(b) the plans one selection returns, each carrying the four indicators of
    the returned set."""
    return [{**base, 'kind': 'selection', 'indicator': indicator, 'position': position,
             'plan': index, 'cost': entry['costs'][position],
             'distinct': entry['distinct'][position],
             **_components(dump['features'], entry['behaviours'][position]),
             **{f'set_{name}': value for name, value in entry['indicators'].items()},
             'wall_s': entry['wall_s'], 'cpu_s': entry['cpu_s']}
            for position, index in enumerate(entry['indices'])]


def _pair_rows(base, counter, plans, indicator, entry):
    """(c) each unordered pair of returned plans read componentwise: the two
    values of every feature, that feature's own term of psi_M, and the total."""
    rows = []
    for i, j in combinations(entry['indices'], 2):
        b1, b2 = plans[i].behaviour, plans[j].behaviour
        values = {key: (dim.payload(b1), dim.payload(b2))
                  for key, dim in counter.dimensions.items()}
        terms = {key: dim.distance(b1, b2) for key, dim in counter.dimensions.items()}
        differing = [key for key, (left, right) in values.items() if left != right]
        rows += [{**base, 'kind': 'pair', 'indicator': indicator, 'plan_i': i, 'plan_j': j,
                  'feature': key, 'value_i': left, 'value_j': right, 'differs': left != right,
                  'contribution': terms[key], 'psi': sum(terms.values()),
                  'differing_features': ' '.join(differing)}
                 for key, (left, right) in values.items()]
    return rows


def run_task(task_id, cfg):
    """The whole case study on one instance, under each model in turn."""
    ctx = runner.context(cfg, task_id)
    k, kappa = cfg['e1']['k'], cfg['e1']['kappa']
    found = _survey(cfg)
    chosen, clause = _choose(cfg, found)
    trace, loaded, rows, records, selections = {}, None, [], [], {}
    for spec in _specs(cfg):
        task, counter, loaded, record, dump = runner.setup(cfg, ctx, spec, trace_cache=trace,
                                                           loaded=loaded)
        plans = loaded['plans']
        base = {'instance': loaded['record']['instance'], 'domain': loaded['record']['domain'],
                'q': loaded['record']['q'], 'N': loaded['record']['requested'],
                'model': spec.name, 'k': k, 'kappa': kappa,
                'pool_size': loaded['record']['size'], 'b': len(dump['distinct'])}
        records.append(record)
        rows += _behaviour_rows(base, dump)
        selections[spec.name] = {}
        for indicator in runner.INDICATORS:
            selected, wall, cpu = runner.select(counter, plans, k, indicator, kappa)
            entry = runner.selection_record(loaded, dump, selected, wall, cpu)
            entry['indicators'] = runner.indicators(counter, selected, kappa)
            selections[spec.name][indicator] = entry
            rows += _selection_rows(base, dump, indicator, entry)
            rows += _pair_rows(base, counter, plans, indicator, entry)
    return {'pool': loaded['record'], 'model': records[0], 'rows': rows,
            'extra': {'k': k, 'kappa': kappa, 'rule': RULE, 'clause': clause, 'survey': found,
                      'chosen': chosen, 'models': records, 'selections': selections}}


def _note(cfg, extra):
    """The short markdown note the paper's subsection quotes."""
    chosen, settings = extra.get('chosen'), cfg['e1']
    feature = chosen['resource_feature'] if chosen else None
    lines = ['# E1 -- reading the differences componentwise', '',
             f'Instance-selection rule: {RULE}.', '',
             f"Domain `{settings['domain']}` at q = {settings['q']}, min_behaviours = "
             f"{settings['min_behaviours']}. Candidates surveyed:", '',
             f'| instance | pool | pool_size | b | distinct `{feature}` values |',
             '| --- | --- | --- | --- | --- |']
    lines += [f"| {row['instance']} | {row['pool_stem']} | {row['pool_size']} | {row['b']} "
              f"| {row['resource_values']} |" for row in extra.get('survey', [])]
    if chosen:
        lines += ['', f"Chosen: **{chosen['instance']}**, pool `{chosen['pool_stem']}` (pool_size "
                      f"{chosen['pool_size']}, b = {chosen['b']} under {chosen['model']}); the "
                      f"clause that decided: {extra['clause']}.",
                  f"The `{feature}` feature (the resource count, which on rovers is the rover "
                  'count) is ' + ('CONSTANT here -- it takes one value, so no returned pair can '
                                  'differ on it.' if chosen['resource_values'] < 2 else
                                  f"not constant: it takes {chosen['resource_values']} values.")]
    lines += ['', f"k = {extra.get('k')}, kappa = {extra.get('kappa')}, the latter passed "
                  'explicitly to every indicator and every selection.',
              '', f'Tie-breaking: {reports.TIE_RULE}', '',
              'Models (the two applying to the domain, in registry order; the generic control is '
              'not part of E1):']
    lines += [f"- `{record['name']}` [{record['hash']}]: "
              + ', '.join(f"{f['key']} (weight {f['weight']}, {f['params']})"
                          for f in record['features'])
              + f", weights {record['weight_convention']}" for record in extra.get('models', [])]
    return '\n'.join(lines) + '\n'


def report(cfg, results):
    """The three CSVs, the note, the two tables and the manifest."""
    out = reports.report_dir(cfg, 'e1')
    usable = [r for r in results if not r.get('error') and not r.get('extra', {}).get('skipped')]
    rows = [row for result in usable for row in result['rows']]
    kinds, written = {}, []
    for kind, (name, head, tail) in OUTPUTS.items():
        kinds[kind] = [row for row in rows if row['kind'] == kind]
        features = sorted({key for row in kinds[kind] for key in row if key.startswith('f_')})
        written.append(reports.write_csv(out / name, kinds[kind], [*head, *features, *tail]))

    note = out / 'e1_note.md'
    note.write_text(_note(cfg, usable[0]['extra'] if usable else {}))
    written.append(note)

    tuple_of = lambda row: '; '.join(f'{key[2:]}={row[key]}' for key in sorted(row)
                                     if key.startswith('f_') and row[key] is not None)
    written.append(reports.table(
        out / 'tables' / 'e1_behaviours.tex', 'tab:e1-behaviours',
        'The behaviours the chosen pool exhibits under each model, componentwise, with how many '
        'plans exhibit each and the cheapest such plan.',
        ['model', 'id', 'behaviour', 'plans', 'cheapest cost'],
        [[row['model'], row['distinct'], tuple_of(row), row['plans'], row['cheapest_cost']]
         for row in kinds['behaviour']], digits=0, aligns='lrlrr'))

    sets = {}
    for row in kinds['selection']:
        entry = sets.setdefault((row['model'], row['indicator']), {'plans': [], 'costs': []})
        entry['plans'].append(row['plan'])
        entry['costs'].append(row['cost'])
        entry.update({name: row[f'set_{name}'] for name in runner.INDICATORS})
    written.append(reports.table(
        out / 'tables' / 'e1_selections.tex', 'tab:e1-selections',
        'The plans each of the four selections returns on the chosen instance, and the four '
        'indicators of the returned set. ' + reports.TIE_RULE,
        ['model', 'indicator', 'plans', 'costs', 'B-Coverage', 'B-MaxSum', 'B-MaxMin', 'B-Novelty'],
        [[model, indicator, ' '.join(map(str, entry['plans'])), ' '.join(map(str, entry['costs']))]
         + [entry[name] for name in runner.INDICATORS]
         for (model, indicator), entry in sets.items()], aligns='llll' + 'r' * 4))

    written.append(reports.manifest(cfg, 'e1', written, results))
    return written
