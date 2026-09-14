"""Task ids, the shared setup every task opens with, and running one task
with its traceback captured rather than lost.

Two kinds of task exist. ``select`` runs the four selections on one pool
under one model, once, to the largest k any experiment reads; E1 and E2 are
reports over those results, since the selection functions extend their
answer one plan at a time and every smaller k is a prefix. ``time`` is E3's
repeated, cold-clock timing of the second phase.
"""

import importlib
import json
import signal
import subprocess
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from bdc_experiments import SCHEMA_VERSION, models, pools
from bdc_experiments.config import load, results_root

#: The run-stage task kinds, and the reports built over their results.
KINDS = {'select': 'select', 'time': 'timing'}
REPORTS = {'e1': 'e1_case_study', 'e2': 'e2_separation', 'e3': 'e3_cost'}

INDICATORS = ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty')

#: The nine fields every row of every result carries, so that the reports can
#: group and pair on them without knowing which task wrote them.
BASE_FIELDS = ('instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b')


class SkipTask(Exception):
    """A task with nothing to do. Recorded as skipped with its reason, which is
    not the same thing as a failure and not the same thing as a zero."""


def module(name):
    table = {**KINDS, **REPORTS}
    if name not in table:
        raise ValueError(f"unknown task kind or report '{name}'; valid: {sorted(table)}")
    return importlib.import_module(f'bdc_experiments.{table[name]}')


def now():
    return datetime.now(timezone.utc).isoformat()


def git_revision():
    """The commit the code is running from, and whether the tree was dirty."""
    here = Path(__file__).resolve().parent
    def git(*args):
        done = subprocess.run(['git', '-C', str(here), *args], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None
    return {'revision': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain'))}


# ----------------------------------------------------------------------
# Tasks
# ----------------------------------------------------------------------

def tasks(cfg, kind):
    return module(kind).tasks(cfg)


def pool_tasks(cfg, kind, sizes, specs_for):
    """One task per (pool of a requested size, model): ``<kind>/<instance>/<pool>/<model>``."""
    ids = []
    for path in pools.pool_files(cfg):
        pool = pools.read_pool(path)
        if pool['requested'] in sizes:
            ids += [f"{kind}/{pool['instance']}/{path.stem}/{spec.name}"
                    for spec in specs_for(pool)]
    return ids


def context(cfg, task_id):
    """The pieces of a task id: ``<kind>/<domain>/<ipc>/<stem>/<pool>/<model>``."""
    kind, domain, ipc, stem, pool_stem, model = task_id.split('/')
    return {'kind': kind, 'task_id': task_id, 'instance': f'{domain}/{ipc}/{stem}',
            'domain': domain, 'pool_stem': pool_stem, 'model': model,
            'pool_path': results_root(cfg) / 'pools' / domain / stem / f'{pool_stem}.json'}


def instance_info(cfg, pool):
    return {'id': pool['instance'], 'domain': pool['domain'],
            'optimal_cost': pool['optimal_cost'], 'q': pool['q'],
            'resource_dir': results_root(cfg) / 'resources'}


def setup(cfg, ctx, spec, trace_cache=None):
    """Task, counter, cost-sorted pool, model record and behaviour dump."""
    pool = pools.read_pool(ctx['pool_path'])
    if not pool['plans']:
        raise SkipTask(f"the pool holds no plans"
                       f"{' (the planner timed out)' if pool.get('timed_out') else ''}, "
                       f'so there is nothing to select from')
    task = pools.task_of(pool)
    info = instance_info(cfg, pool)
    counter = models.build_counter(spec, task, info, trace_cache=trace_cache)
    loaded = pools.load_pool(ctx['pool_path'], counter=counter, task=task)
    if not loaded['plans']:
        raise SkipTask('no plan of the pool could be replayed against the task')
    record = models.model_record(spec, counter, task, info)
    return task, counter, loaded, record, pools.behaviour_dump(cfg, counter, loaded, record)


# ----------------------------------------------------------------------
# Selection, timed and dumped whole
# ----------------------------------------------------------------------

def indicators(counter, plans, kappa):
    """All four indicators of one plan set, at the given kappa."""
    return {'bcoverage': float(counter.b_coverage(plans)),
            'bmaxsum': float(counter.b_maxsum(plans)),
            'bmaxmin': float(counter.b_maxmin(plans)),
            'bnovelty': float(counter.b_novelty(plans, k_nn=kappa))}


def select(counter, plans, k, indicator, kappa):
    """``(selected, wall_s, cpu_s)``. kappa is always passed: the library's
    DEFAULT_K_NN is never what an experiment means."""
    wall, cpu = time.perf_counter(), time.process_time()
    selected = counter.extract(plans, k, indicator=indicator, k_nn=kappa)
    return selected, time.perf_counter() - wall, time.process_time() - cpu


def selection_record(loaded, dump, selected, wall, cpu):
    """Everything a reader needs to recompute the selection's numbers by hand."""
    position = {id(plan): index for index, plan in enumerate(loaded['plans'])}
    indices = [position[id(plan)] for plan in selected]
    return {
        'indices': indices,
        'costs': [loaded['costs'][index] for index in indices],
        'behaviours': [dump['plans'][index]['behaviour'] for index in indices],
        'distinct': [dump['plans'][index]['distinct'] for index in indices],
        'actions': [[str(action) for action in loaded['plans'][index].actions]
                    for index in indices],
        'wall_s': wall, 'cpu_s': cpu,
    }


def base_row(loaded, dump, model_record, k=None, kappa=None):
    """The nine mandatory fields, ready to be extended with the row's own."""
    record = loaded['record']
    return {'instance': record['instance'], 'domain': record['domain'], 'q': record['q'],
            'N': record['requested'], 'model': model_record['name'], 'k': k, 'kappa': kappa,
            'pool_size': record['size'], 'b': len(dump['distinct'])}


@contextmanager
def time_limit(seconds):
    """A task that runs away is a recorded failure, not a hung array job."""
    def raise_timeout(signum, frame):
        raise TimeoutError(f'task exceeded the selection time limit of {seconds}s')
    previous = signal.signal(signal.SIGALRM, raise_timeout)
    signal.alarm(int(seconds))
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


# ----------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------

def result_path(cfg, kind, task_id):
    """One file per task. The id's slashes become '__' so the directory stays
    flat and greppable; the id itself is inside the file."""
    tail = task_id.split('/', 1)[1].replace('/', '__')
    return results_root(cfg) / 'results' / kind / f'{tail}.json'


def run_task(cfg, kind, task_id, force=False):
    """Run one task and write its result file. Never raises for a task failure."""
    path = result_path(cfg, kind, task_id)
    if path.is_file() and not force:
        return json.loads(path.read_text())
    result = {'schema': 'result', 'version': SCHEMA_VERSION, 'task_id': task_id,
              'kind': kind, 'config_hash': cfg['meta']['config_hash'],
              'git': git_revision(), 'started': now(), 'ended': None, 'error': None,
              'pool': None, 'model': None, 'rows': [], 'extra': {}}
    try:
        with time_limit(cfg['run']['time_limit_selection_s']):
            result.update(module(kind).run_task(task_id, cfg))
    except SkipTask as nothing:
        result['extra'] = dict(result['extra'], skipped=str(nothing))
    except Exception as failure:
        result['error'] = {'type': type(failure).__name__, 'message': str(failure),
                           'traceback': traceback.format_exc()}
    result['ended'] = now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    return result


def _worker(arguments):
    config_path, results_dir, kind, task_id, force = arguments
    cfg = load(config_path, results_dir=results_dir)
    result = run_task(cfg, kind, task_id, force=force)
    return task_id, result.get('error'), bool(result.get('extra', {}).get('skipped'))


def run(cfg, kind, only=None, force=False, jobs=1, log=print):
    """Phase two over a kind's tasks; resumable, one file each."""
    ids = [t for t in tasks(cfg, kind) if only is None or t == only]
    if only is not None and not ids:
        raise ValueError(f"no task '{only}' in {kind}")
    arguments = [(cfg['meta']['config_path'], cfg['run']['results_dir'], kind, t, force)
                 for t in ids]
    counts = {'ok': 0, 'failed': 0, 'skipped': 0}
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outcomes = list(pool.map(_worker, arguments))
    else:
        outcomes = [_worker(argument) for argument in arguments]
    for task_id, error, skipped in outcomes:
        counts['failed' if error else ('skipped' if skipped else 'ok')] += 1
        log(f"{task_id}: {'FAILED ' + error['type'] if error else ('skipped' if skipped else 'ok')}")
    return counts


def load_results(cfg, kind):
    """Every result file of a task kind, in a stable order."""
    root = results_root(cfg) / 'results' / kind
    return [json.loads(path.read_text()) for path in sorted(root.glob('*.json'))] if root.is_dir() else []


def load_dump(cfg, result):
    """The behaviour dump a result's numbers are indexed into."""
    pool = result['pool']
    path = (results_root(cfg) / 'behaviours' / result['model']['hash'] / pool['domain']
            / pool['instance'].split('/')[-1] / f"{pool['pool_stem']}.json")
    return json.loads(path.read_text()) if path.is_file() else None
