"""Ground rule 8, enforced: no row holds a number that cannot be recomputed
from its own result file plus the behaviour dump.

Every selection an experiment records goes through ``runner.selection_record``,
so this walks the result JSONs for those records -- wherever an experiment
chose to nest them -- resolves which model each belongs to from the model
records the file carries, and recomputes the indicator values recorded beside
them from the dissimilarity matrix alone. No counter, no pool, no planner.
"""

from itertools import combinations

import pytest

from bdc_experiments import cli, runner
from bdc_experiments.config import load

EXPERIMENTS = ('e1', 'e2', 'e3', 'e4', 'e5', 'e6')

#: The keys ``runner.selection_record`` writes; enough to recognise one.
SELECTION_KEYS = {'indices', 'costs', 'behaviours', 'distinct', 'actions'}

#: Where an experiment may have put the four indicator values of a set.
VALUE_KEYS = ('values', 'indicators', 'scores')

#: Row fields that hold the value of the row's own ``indicator``. An experiment
#: either names the four values after the indicators (E5) or reports the one
#: its row is about under a name of its own (E3's greedy value, E4's prefix
#: value and the same value recomputed from the matrix).
OWN_VALUE_FIELDS = ('greedy', 'value', 'matrix_value')

INDICATORS = frozenset(runner.INDICATORS)


@pytest.fixture(scope='module')
def swept(tmp_path_factory):
    root = tmp_path_factory.mktemp('recompute')
    for experiment in EXPERIMENTS:
        assert cli.main(['run', 'smoke', experiment, '--results-dir', str(root)]) == 0
    return load('smoke', results_dir=root)


def model_hashes(result):
    """``model name -> hash`` for every model record anywhere in the file.

    A task that runs several models -- E1's two, E5's resolution variants,
    E6's feature counts -- has to say which dump each of its numbers belongs
    to, or the recomputation guarantee is not met. This is that mapping.
    """
    found = {}

    def collect(node):
        if isinstance(node, dict):
            if {'name', 'hash', 'features'} <= set(node):
                found[node['name']] = node['hash']
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(result)
    return found


def selections(result):
    """``(selection record, scope)`` for every selection in the file.

    The scope carries what the enclosing structure says about the selection:
    which model's dump it is indexed into, which kappa it was made at, and the
    indicator values recorded beside it.
    """
    names = model_hashes(result)
    found = []

    def walk(node, scope):
        if isinstance(node, dict):
            scope = dict(scope)
            model = node.get('model')
            if isinstance(model, dict) and 'hash' in model:
                scope['hash'] = model['hash']
            elif isinstance(model, str) and model in names:
                scope['hash'] = names[model]
            if isinstance(node.get('kappa'), int):
                scope['kappa'] = node['kappa']
            # E4 records one run to k_max and reads every smaller k off its
            # prefixes, so k_max is that selection's own k.
            for key in ('k_max', 'k'):
                if isinstance(node.get(key), int):
                    scope['k'] = node[key]
                    break
            if node.get('indicator') in INDICATORS:
                scope['indicator'] = node['indicator']
            for key in VALUE_KEYS:
                values = node.get(key)
                if isinstance(values, dict) and values and set(values) <= INDICATORS:
                    scope['values'] = values
            if SELECTION_KEYS <= set(node):
                found.append((node, scope))
            for key, value in node.items():
                # A key can carry the scope too: E1 files its selections under
                # the model name and then the indicator name.
                child = dict(scope, hash=names[key]) if key in names else scope
                child = dict(child, indicator=key) if key in INDICATORS else child
                walk(value, child)
        elif isinstance(node, list):
            for item in node:
                walk(item, scope)

    walk(result, {'hash': result['model']['hash'] if result.get('model') else None})
    return found


def indicators_from(matrix, chosen, kappa):
    """The paper's four indicators over a set of distinct behaviour indices,
    read straight off the b x b matrix."""
    distinct = sorted(set(chosen))
    pairs = list(combinations(distinct, 2))
    if len(distinct) < 2:
        return {'bcoverage': float(len(distinct)), 'bmaxsum': 0.0,
                'bmaxmin': 0.0, 'bnovelty': 0.0}
    k_prime = min(kappa, len(distinct) - 1)
    means = [sum(sorted(matrix[i][j] for j in distinct if j != i)[:k_prime]) / k_prime
             for i in distinct]
    return {'bcoverage': float(len(distinct)),
            'bmaxsum': sum(matrix[i][j] for i, j in pairs),
            'bmaxmin': min(matrix[i][j] for i, j in pairs),
            'bnovelty': sum(means) / len(distinct)}


def dump_by_hash(cfg, result, model_hash):
    pool = result['pool']
    return runner.load_dump(cfg, model_hash, pool['domain'],
                            pool['instance'].split('/')[-1], pool['pool_stem'])


def longest_run(index, key):
    """The selection a prefix row belongs to: same model, kappa and indicator,
    run to the largest k that reaches this row's."""
    model_hash, k, kappa, indicator = key
    candidates = [(other[1], selection) for other, selection in index.items()
                  if other[0] == model_hash and other[2] == kappa
                  and other[3] == indicator and (other[1] or 0) >= k]
    return max(candidates)[1] if candidates else None


def usable(cfg, experiment):
    for result in runner.load_results(cfg, experiment):
        if result.get('error') or (result.get('extra') or {}).get('skipped'):
            continue
        yield result


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_every_recorded_selection_recomputes(swept, experiment):
    """The behaviours, costs and indicator values of every selection, rebuilt
    from the dump and compared with what the task wrote."""
    checked = recomputed = 0
    for result in usable(swept, experiment):
        for selection, scope in selections(result):
            dump = dump_by_hash(swept, result, scope['hash'])
            assert dump is not None, (
                f"{result['task_id']}: a selection names no reachable behaviour dump "
                f"(hash {scope['hash']}), so its numbers cannot be recomputed")
            by_index = {entry['index']: entry for entry in dump['plans']}
            checked += 1
            for position, index in enumerate(selection['indices']):
                entry = by_index[index]
                assert selection['behaviours'][position] == entry['behaviour']
                assert selection['costs'][position] == entry['cost']
                assert selection['distinct'][position] == entry['distinct']
            if 'values' not in scope or 'kappa' not in scope:
                continue
            rebuilt = indicators_from(dump['matrix'], selection['distinct'], scope['kappa'])
            for name, reported in scope['values'].items():
                assert rebuilt[name] == pytest.approx(reported, abs=1e-9), (
                    f"{result['task_id']} {name} at kappa={scope['kappa']}: recomputed "
                    f'{rebuilt[name]} against the reported {reported}')
                recomputed += 1
    assert checked, f'{experiment} recorded no selection to recompute'
    if experiment in ('e1', 'e2'):  # the two that file the values beside the selection
        assert recomputed, f'{experiment} recorded no indicator value to recompute'


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_every_row_value_recomputes_from_its_own_selection(swept, experiment):
    """The other half of ground rule 8: the numbers in the ROWS.

    A row names its model, its k, its kappa and its indicator, and the file
    records the selection those name. Take that selection's first k
    behaviours, recompute the indicators from the matrix, and compare with
    whatever the row reported.
    """
    checked = 0
    for result in usable(swept, experiment):
        names = model_hashes(result)
        index = {}
        for selection, scope in selections(result):
            key = (scope.get('hash'), scope.get('k'), scope.get('kappa'), scope.get('indicator'))
            index.setdefault(key, selection)
        for row in result['rows']:
            indicator = row.get('indicator') or row.get('selector')
            key = (names.get(row.get('model')), row.get('k'), row.get('kappa'), indicator)
            if key[0] is None or key[1] is None or indicator not in INDICATORS:
                continue
            selection = index.get(key) or longest_run(index, key)
            if selection is None:
                continue
            dump = dump_by_hash(swept, result, key[0])
            # E4's rows are the prefixes of one run to k_max, which is sound
            # because the greedy rules are prefix-consistent -- proved in
            # test_e4_prefix_consistency.py. Every other row's k is the length
            # of the selection it names, so the slice is a no-op there.
            rebuilt = indicators_from(dump['matrix'], selection['distinct'][:row['k']],
                                      row['kappa'])
            expected = {field: rebuilt[name] for name in INDICATORS
                        for field in (name, f'set_{name}', f'score_{name}') if field in row}
            expected.update({field: rebuilt[indicator]
                             for field in OWN_VALUE_FIELDS if row.get(field) is not None})
            for field, value in expected.items():
                if row.get(field) is None:
                    continue
                assert value == pytest.approx(row[field], abs=1e-9), (
                    f"{result['task_id']} row {key}: {field} reported as {row[field]}, "
                    f'recomputed from the dump as {value}')
                checked += 1
    if experiment != 'e6':          # E6's rows are timings, not indicator values
        assert checked, f'{experiment} reported no row value traceable to a selection'


@pytest.mark.parametrize('experiment', EXPERIMENTS)
def test_the_dump_alone_determines_the_behaviour_count(swept, experiment):
    """`b` on a row is the distinct-behaviour count of that row's own model."""
    checked = 0
    for result in usable(swept, experiment):
        names = model_hashes(result)
        for row in result['rows']:
            if row.get('b') is None or row.get('model') not in names:
                continue
            dump = dump_by_hash(swept, result, names[row['model']])
            assert dump is not None, f"{result['task_id']}: row names an unreachable dump"
            assert row['b'] == len(dump['distinct']), (
                f"{result['task_id']}: a {row['model']} row says b={row['b']}, "
                f"its dump has {len(dump['distinct'])}")
            checked += 1
    assert checked, f'{experiment} reported no row whose b could be traced to a dump'
