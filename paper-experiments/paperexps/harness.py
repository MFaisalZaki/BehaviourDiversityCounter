"""What every experiment shares: loading a pool, running the task list, and
the small measurements the rows are made of.

A task is one pool file.  ``load_pool`` replays every plan of it once, drops
the ones that cannot be replayed, filters by the cost bound and sorts the rest
by cost, which is what the brief's selection procedures assume (the cheapest
plan then represents every behaviour, under every selection rule, because the
library breaks ties towards the lowest index).  The state traces are kept in a
``trace_cache`` that every counter built over the pool shares, so a task that
scores the pool under several models simulates it once.
"""

import json
import os
import sys
import time
import traceback
from fractions import Fraction

import numpy as np
from unified_planning.shortcuts import SequentialSimulator

from behaviour_diversity_counter import BehaviourDiversityCounter
from behaviour_diversity_counter.simulation import InapplicablePlanError, simulate
from utils import construct_task, dumpfile_name

INDICATORS = ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty')
SCORE_OF = {'bcoverage': 'score_bcov', 'bmaxsum': 'score_bmaxsum',
            'bmaxmin': 'score_bmaxmin', 'bnovelty': 'score_bnov'}
TASK_COLUMNS = ('task_id', 'domain', 'year', 'inst', 'generator', 'generator_name',
                'q', 'pool_k')


class Timer:
    """CPU and wall-clock seconds of a ``with`` block."""

    def __enter__(self):
        self._cpu, self._wall = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *_):
        self.cpu = time.process_time() - self._cpu
        self.wall = time.perf_counter() - self._wall


class Pool:
    """One task's plans, replayed, cost-bounded and sorted by cost."""

    def __init__(self, taskdetails):
        self.details = taskdetails
        with Timer() as timer:
            self.task, plans, self.info = construct_task(taskdetails)
        self.parse_s = timer.wall

        simulator = SequentialSimulator(problem=self.task)
        self.trace = {}
        kept, self.inapplicable = [], []
        with Timer() as timer:
            for plan in plans:
                try:
                    states, cost = simulate(self.task, plan, simulator)
                except InapplicablePlanError as error:
                    self.inapplicable.append({'index': plan.pool_index, 'error': str(error)})
                    continue
                self.trace[id(plan)] = (states, cost)
                plan.cost = cost
                kept.append(plan)
        self.simulate_s = timer.wall

        # c* is the cheapest plan of the pool: the generators are cost-optimal
        # in their first plan, so at q = 1.0 this is exact and at q > 1 it is
        # the best bound the pool itself gives.
        self.optimal_cost = min((plan.cost for plan in kept), default=None)
        bound = None if self.optimal_cost is None else Fraction(str(self.info['q'])) * self.optimal_cost
        self.cost_filtered = [plan.pool_index for plan in kept if plan.cost > bound]
        kept = [plan for plan in kept if plan.cost <= bound]
        # Brief, select_coverage: sort P by cost ascending before scanning.
        # Stable, so plans of equal cost keep their generation order.
        kept.sort(key=lambda plan: (plan.cost, plan.pool_index))
        self.plans = kept
        self.index = {id(plan): position for position, plan in enumerate(kept)}
        self.generation_s = self.info.get('planning-time')

    def prefix(self, size):
        """The first ``size`` plans in generation order, still sorted by cost."""
        return [plan for plan in self.plans if plan.pool_index < size]

    def record(self):
        return {
            'size': len(self.plans),
            'raw-size': len(self.plans) + len(self.inapplicable) + len(self.cost_filtered)
                        + len(self.info['parse-failures']),
            'duplicates-dropped': self.info['duplicates-dropped'],
            'parse-failures': self.info['parse-failures'],
            'inapplicable': self.inapplicable,
            'cost-filtered': self.cost_filtered,
            'optimal-cost': self.optimal_cost,
            'optimal-cost-source': 'cheapest plan of the pool',
            'generation-time-s': self.generation_s,
            'parse-time-s': self.parse_s,
            'simulate-time-s': self.simulate_s,
        }


def load_pool(taskdetails):
    return Pool(taskdetails)


def make_counter(pool, dimensions):
    return BehaviourDiversityCounter(pool.task, dimensions, trace_cache=pool.trace)


# ---------------------------------------------------------------- measures --

def scores(counter, plans, k_nn):
    """The four indicators of a plan set, under the row names the CSVs use."""
    return {
        'score_bcov': counter.b_coverage(plans),
        'score_bmaxsum': counter.b_maxsum(plans),
        'score_bmaxmin': counter.b_maxmin(plans),
        'score_bnov': counter.b_novelty(plans, k_nn=k_nn),
    }


def select_all(counter, plans, k, k_nn):
    """The four behaviour-space selections, each timed."""
    selected, timing = {}, {}
    for indicator in INDICATORS:
        with Timer() as timer:
            selected[indicator] = counter.extract(plans, k, indicator=indicator, k_nn=k_nn)
        timing[indicator] = {'cpu_s': timer.cpu, 'wall_s': timer.wall}
    return selected, timing


def behaviour_set(plans):
    return sorted({plan.behaviour for plan in plans})


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def mean_cost_ratio(plans, optimal_cost):
    if not plans or not optimal_cost:
        return None
    return float(sum(Fraction(plan.cost) for plan in plans) / len(plans) / Fraction(optimal_cost))


def representatives(counter, plans):
    """One plan per distinct behaviour, the first (cheapest) exhibiting it."""
    reps = {}
    for plan, behaviour in zip(plans, counter._behaviours_of(plans)):
        reps.setdefault(behaviour, plan)
    return reps


def selection_k_values(params, size):
    return [k for k in params['k-values'] if k <= size]


def plan_record(plan):
    return {'cost': str(plan.cost), 'behaviour': plan.behaviour, 'pool_index': plan.pool_index,
            'plan': plan.plan_str}


# --------------------------------------------------------------- the runner --

def task_record(taskdetails):
    return {
        'task_id': taskdetails['task_id'],
        'domain': taskdetails['domain'],
        'year': taskdetails['year'],
        'inst': taskdetails['inst'],
        'generator': taskdetails['generator'],
        'generator_name': taskdetails['generator_name'],
        'q': taskdetails['q'],
        'pool_k': taskdetails['k'],
        'track': taskdetails['track'],
        'pool_file': os.path.basename(taskdetails['pool_file']),
        'resources_file': taskdetails['resources'],
    }


def _jsonable(value):
    if isinstance(value, Fraction):
        return str(value) if value.denominator != 1 else int(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not JSON serialisable: {type(value)}')


def run_tasks(tasks, basedir, run_task, params, force=False):
    """Run ``run_task(taskdetails, params)`` for every task, one result file
    each.  A task whose result file exists is skipped unless ``force``; a task
    that raises writes its traceback in place of a result, so a failure is
    recorded and never silently counted as anything else.
    """
    for taskdetails in tasks:
        dumpfile = os.path.join(basedir, dumpfile_name(taskdetails))
        if os.path.exists(dumpfile) and not force:
            print(f"| skipping {taskdetails['task_id']}: {dumpfile} exists")
            continue
        print(f"| Running {params['name']} on {taskdetails['task_id']}")
        result = {'task': task_record(taskdetails), 'experiment': params['name'],
                  'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'error': None,
                  'rows': []}
        with Timer() as timer:
            try:
                result.update(run_task(taskdetails, params))
            except Exception:                                   # noqa: BLE001
                result['error'] = traceback.format_exc()
                print(result['error'], file=sys.stderr)
        result['timing'] = {'cpu_s': timer.cpu, 'wall_s': timer.wall}
        result['ended'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with open(dumpfile, 'w') as handle:
            json.dump(result, handle, indent=2, default=_jsonable)


def load_results(dump_dir):
    """Every result file of an experiment, as written by :func:`run_tasks`."""
    results = []
    for name in sorted(os.listdir(dump_dir)):
        if name.endswith('.json'):
            with open(os.path.join(dump_dir, name)) as handle:
                results.append(json.load(handle))
    return results


def flatten(results, key='rows'):
    """One flat dict per row, the task's columns prepended."""
    rows = []
    for result in results:
        task = {column: result['task'].get(column) for column in TASK_COLUMNS}
        for row in result.get(key) or []:
            rows.append({**task, **row})
    return rows


def coverage_rows(results):
    """Instances with a non-empty pool per (domain, generator, q), and the
    number of pools that failed for each reason."""
    groups = {}
    for result in results:
        task = result['task']
        key = (task['domain'], task['generator_name'], task['q'])
        entry = groups.setdefault(key, {'domain': key[0], 'generator': key[1], 'q': key[2],
                                        'pools': 0, 'non_empty': 0, 'errors': 0,
                                        'instances_non_empty': set()})
        entry['pools'] += 1
        if result.get('error'):
            entry['errors'] += 1
        elif (result.get('pool') or {}).get('size', 0) > 0:
            entry['non_empty'] += 1
            entry['instances_non_empty'].add(task['inst'])
    rows = []
    for entry in groups.values():
        entry['instances_non_empty'] = len(entry['instances_non_empty'])
        rows.append(entry)
    return sorted(rows, key=lambda row: (row['domain'], row['generator'], row['q']))
