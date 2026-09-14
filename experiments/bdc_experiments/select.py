"""The selection sweep: the four selections on one pool under one model.

Each is run once, to the largest k any experiment reads, and the order in
which it took its plans is dumped whole. The selection functions extend their
answer one plan at a time (checked in tests/experiments/test_prefix_consistency.py),
so the set selected at any smaller k >= 2 is a prefix of this run, and E1 and
E2 read every k they need off the same result. B-Coverage, B-MaxSum and
B-MaxMin do not read kappa, so each is selected once; B-Novelty once per kappa.
"""

from bdc_experiments import models, runner


def k_max(cfg):
    """The largest k any report reads."""
    return max(cfg['selection']['k_values'])


def tasks(cfg):
    return runner.pool_tasks(cfg, 'select', lambda pool: models.selection_specs(cfg, pool['domain']))


def run_task(task_id, cfg):
    ctx = runner.context(cfg, task_id)
    task, counter, loaded, record, dump = runner.setup(cfg, ctx, models.registry(cfg)[ctx['model']])
    plans, kappas = loaded['plans'], cfg['selection']['kappa_values']
    k = min(k_max(cfg), len(plans))
    rows, selections = [], []
    for indicator in runner.INDICATORS:
        for kappa in (kappas if indicator == 'bnovelty' else kappas[:1]):
            selected, wall, cpu = runner.select(counter, plans, k, indicator, kappa)
            entry = {'indicator': indicator, 'kappa': kappa, 'k': k,
                     **runner.selection_record(loaded, dump, selected, wall, cpu),
                     'values': runner.indicators(counter, selected, kappa)}
            selections.append(entry)
            rows.append({**runner.base_row(loaded, dump, record, k=k, kappa=kappa),
                         'indicator': indicator, 'wall_s': wall, 'cpu_s': cpu, **entry['values']})
    return {'pool': loaded['record'], 'model': record, 'rows': rows,
            'extra': {'selections': selections, 'k_max': k,
                      'prefixes': 'the selection at any k >= 2 below k_max is the first k plans '
                                  'of the run recorded here',
                      'kappa': 'B-Coverage, B-MaxSum and B-MaxMin do not read kappa and are '
                               'selected once, at the first configured kappa'}}
