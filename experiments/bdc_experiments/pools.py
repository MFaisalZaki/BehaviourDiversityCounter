"""Reading a pool file back: parse, replay, filter, sort -- and the behaviour
dump, which is both the cache the later tasks read and the raw material a
reader needs to recompute every indicator by hand.
"""

import json
import re
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from unified_planning.io import PDDLReader
from unified_planning.shortcuts import SequentialSimulator

from behaviour_diversity_counter import InapplicablePlanError
from behaviour_diversity_counter.simulation import simulate
from bdc_experiments import SCHEMA_VERSION
from bdc_experiments.config import results_root

#: A behaviour string is per-dimension tokens 'name:value' joined by this.
TOKEN_SEPARATOR = ' $$ '


@lru_cache(maxsize=8)
def load_task(domain_file, problem_file):
    """The unified-planning task. Cached: parsing is seconds, and every pool of
    an instance wants the same task object."""
    return PDDLReader().parse_problem(domain_file, problem_file)


def read_pool(path):
    """The pool file, with its PDDL paths resolved.

    A relative path is resolved against the pool file's own directory, so the
    committed smoke pools travel with the PDDL they were generated from.
    """
    path = Path(path)
    pool = json.loads(path.read_text())
    for key in ('domain_file', 'problem_file'):
        resolved = Path(pool[key])
        pool[key] = str(resolved if resolved.is_absolute() else (path.parent / resolved).resolve())
    return pool


def task_of(pool):
    return load_task(pool['domain_file'], pool['problem_file'])


def _lookup(names):
    """``normalised token -> canonical name``, raising on an ambiguous key."""
    table = {}
    for name in names:
        key = name.lower().replace('-', '_')
        if key in table and table[key] != name:
            raise ValueError(f"ambiguous name '{key}': both {table[key]} and {name}")
        table[key] = name
    return table


def normalise(task, actions):
    """Plan lines rewritten against the task's own action and object names.

    Fast Downward lowercases, and may write '-' where the task writes '_' or
    add a numeric suffix. Every token is resolved against the task; an
    ambiguous one raises rather than being guessed at.
    """
    action_names = _lookup([a.name for a in task.actions])
    object_names = _lookup([o.name for o in task.all_objects])
    rewritten = []
    for line in actions:
        tokens = line.strip().strip('()').split()
        if not tokens:
            continue
        resolved = []
        for position, token in enumerate(tokens):
            table = action_names if position == 0 else object_names
            key = token.lower().replace('-', '_')
            if key not in table:
                # `_12` on the tail is Fast Downward disambiguating a grounded name.
                key = re.sub(r'_\d+$', '', key)
            if key not in table:
                raise ValueError(f"plan token '{token}' matches no "
                                 f"{'action' if position == 0 else 'object'} of the task")
            resolved.append(table[key])
        rewritten.append('(' + ' '.join(resolved) + ')')
    return rewritten


def parse_plans(task, pool):
    """``(plans, dropped)``: the pool's plans as unified-planning plans."""
    reader, plans, dropped = PDDLReader(), [], []
    for index, entry in enumerate(pool['plans']):
        text = '\n'.join(entry['actions'])
        try:
            plan = reader.parse_plan_string(task, text)
        except Exception:
            try:
                plan = reader.parse_plan_string(task, '\n'.join(normalise(task, entry['actions'])))
            except Exception as second:
                dropped.append({'original_index': index, 'reason': f'{type(second).__name__}: {second}'})
                continue
        plans.append((index, plan))
    return plans, dropped


def load_pool(path, counter=None, task=None):
    """A pool ready for selection: parsed, replayed, cost-filtered, cost-sorted.

    ``counter`` replays through the library (so its behaviour cache is warm and
    the pool is walked once); without one the plans are replayed directly.
    Returns the raw pool, the plans in canonical order, their costs, and the
    record every result file carries.
    """
    pool = read_pool(path)
    task = task if task is not None else task_of(pool)

    clock = time.perf_counter()
    parsed, dropped_parse = parse_plans(task, pool)
    parse_s = time.perf_counter() - clock

    clock = time.perf_counter()
    simulator = SequentialSimulator(problem=task) if counter is None else None
    replayed, dropped_replay = [], list(dropped_parse)
    for index, plan in parsed:
        try:
            if counter is None:
                cost = simulate(task, plan, simulator)[1]
            else:
                counter.b_coverage([plan])      # replays, caches, sets plan.cost
                cost = plan.cost
        except InapplicablePlanError as failure:
            dropped_replay.append({'original_index': index, 'reason': str(failure)})
            continue
        replayed.append((index, plan, cost))
    replay_s = time.perf_counter() - clock

    # The bound of phase one, re-applied on the way in. Only a top-quality pool
    # has one: `topk` takes the k cheapest plans whatever they cost.
    dropped_cost, bound = [], None
    optimal = pool.get('optimal_cost')
    if optimal is not None and pool.get('mode') == 'topq':
        bound = Fraction(str(pool.get('q', 1.0))) * Fraction(optimal)
        kept = []
        for index, plan, cost in replayed:
            if Fraction(cost) <= bound:
                kept.append((index, plan, cost))
            else:
                dropped_cost.append({'original_index': index, 'cost': cost, 'bound': float(bound)})
        replayed = kept
    replayed.sort(key=lambda entry: (entry[2], entry[0]))

    return {
        'pool': pool,
        'path': str(path),
        'stem': Path(path).stem,
        'plans': [plan for _, plan, _ in replayed],
        'costs': [cost for _, _, cost in replayed],
        'original_indices': [index for index, _, _ in replayed],
        'record': {
            'instance': pool['instance'], 'domain': pool['domain'], 'ipc': pool.get('ipc'),
            'pool_stem': Path(path).stem, 'mode': pool['mode'], 'q': pool['q'],
            'requested': pool['requested'], 'written': len(pool['plans']),
            'size': len(replayed),
            'dropped_replay': dropped_replay, 'dropped_cost': dropped_cost,
            'duplicates_dropped': pool.get('duplicates_dropped', 0),
            'optimal_cost': optimal, 'optimal_cost_source': pool.get('optimal_cost_source'),
            'cost_bound': (float(bound) if bound is not None else None),
            'timed_out': pool.get('timed_out'), 'exhausted': pool.get('exhausted'),
            'generation_wall_s': pool.get('wall_s'), 'generation_cpu_s': pool.get('cpu_s'),
            'parse_s': parse_s, 'replay_s': replay_s,
        },
    }


# ----------------------------------------------------------------------
# The behaviour dump
# ----------------------------------------------------------------------

def payloads(behaviour):
    """A library behaviour string as the paper's tuple of feature values."""
    return [token.split(':', 1)[1] if ':' in token else token
            for token in behaviour.split(TOKEN_SEPARATOR)]


def model_distance(counter, b1, b2):
    """psi_M(a, b) = sum_i w_i psi_i, summed over the counter's own features."""
    return sum(dimension.distance(b1, b2) for dimension in counter.dimensions.values())


def dump_path(cfg, model_hash, loaded):
    record = loaded['record']
    return (results_root(cfg) / 'behaviours' / model_hash / record['domain']
            / Path(record['instance']).name / f"{record['pool_stem']}.json")


def behaviour_dump(cfg, counter, loaded, model_record, force=False):
    """Per-plan behaviour and cost plus the b x b dissimilarity matrix.

    Written once per (model, pool) and read by every later task. The model hash
    in the path is the whole of the invalidation logic: a changed model is a
    changed path.
    """
    path = dump_path(cfg, model_record['hash'], loaded)
    if path.is_file() and not force:
        return json.loads(path.read_text())

    plans = loaded['plans']
    counter.b_coverage(plans)                     # fills .behaviour on every plan
    strings = [plan.behaviour for plan in plans]
    distinct = list(dict.fromkeys(strings))
    position = {behaviour: index for index, behaviour in enumerate(distinct)}
    matrix = [[0.0] * len(distinct) for _ in distinct]
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            matrix[i][j] = matrix[j][i] = model_distance(counter, distinct[i], distinct[j])

    dump = {
        'schema': 'behaviours', 'version': SCHEMA_VERSION,
        'instance': loaded['record']['instance'], 'domain': loaded['record']['domain'],
        'pool_stem': loaded['record']['pool_stem'], 'model': model_record,
        'features': list(counter.dimensions),
        'plans': [{'index': index, 'original_index': loaded['original_indices'][index],
                   'cost': loaded['costs'][index], 'behaviour': payloads(behaviour),
                   'distinct': position[behaviour]}
                  for index, behaviour in enumerate(strings)],
        'distinct': [payloads(behaviour) for behaviour in distinct],
        'matrix': matrix,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump))
    return dump


def pool_files(cfg):
    """Every pool file of the run, in a stable order."""
    root = results_root(cfg) / 'pools'
    return sorted(root.rglob('*.json')) if root.is_dir() else []


def ensure_pools(cfg):
    """Seed an empty run from pools committed next to its config file.

    That is how the smoke sweep runs without a planner: ``configs/smoke.toml``
    sits beside ``configs/smoke_pools/pools/``, so ``bdcexp run smoke e3`` on a
    clean clone copies those in and proceeds. A config with no such directory,
    or a run that already holds pools, is left alone.
    """
    if pool_files(cfg):
        return []
    seed = Path(cfg['meta']['config_path']).parent / 'smoke_pools' / 'pools'
    if not seed.is_dir():
        return []
    target = results_root(cfg) / 'pools'
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted(seed.rglob('*.json')):
        # The committed pool points at the PDDL beside it with a relative path;
        # resolve it now, so the copy in the run directory stands on its own.
        pool = read_pool(source)
        destination = target / source.relative_to(seed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(pool, indent=1))
    return pool_files(cfg)
