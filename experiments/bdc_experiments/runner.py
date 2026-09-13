"""Task ids, the shared setup every experiment opens with, and running one
task with its traceback captured rather than lost.
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

MODULES = {'e1': 'e1_case_study', 'e2': 'e2_separation', 'e3': 'e3_greedy_vs_optimum',
           'e4': 'e4_fixed_size', 'e5': 'e5_resolution', 'e6': 'e6_cost'}

INDICATORS = ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty')


class SkipTask(Exception):
    """A task with nothing to do. Recorded as skipped with its reason, which is
    not the same thing as a failure and not the same thing as a zero."""


def module(experiment):
    if experiment not in MODULES:
        raise ValueError(f"unknown experiment '{experiment}'; valid: {sorted(MODULES)}")
    return importlib.import_module(f'bdc_experiments.{MODULES[experiment]}')


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

def default_tasks(cfg, experiment):
    """One task per (pool, applicable model): what most experiments want."""
    ids = []
    for path in pools.pool_files(cfg):
        pool = pools.read_pool(path)
        for spec in models.models_for(cfg, pool['domain']):
            ids.append(f"{experiment}/{pool['instance']}/{path.stem}/{spec.name}")
    return ids


def tasks(cfg, experiment):
    """An experiment's own task list, or the default one."""
    own = getattr(module(experiment), 'tasks', None)
    return own(cfg) if own else default_tasks(cfg, experiment)


def context(cfg, task_id):
    """The pieces of a task id: ``<experiment>/<domain>/<ipc>/<stem>/<pool>[/...]``."""
    experiment, *parts = task_id.split('/')
    domain, ipc, stem, pool_stem, *extra = parts
    return {
        'experiment': experiment, 'task_id': task_id,
        'instance': f'{domain}/{ipc}/{stem}', 'domain': domain, 'ipc': ipc, 'stem': stem,
        'pool_stem': pool_stem, 'extra': extra,
        'pool_path': results_root(cfg) / 'pools' / domain / stem / f'{pool_stem}.json',
    }


def instance_info(cfg, pool):
    return {'id': pool['instance'], 'domain': pool['domain'],
            'optimal_cost': pool['optimal_cost'], 'q': pool['q'],
            'resource_dir': results_root(cfg) / 'resources'}


def setup(cfg, ctx, spec, trace_cache=None, loaded=None):
    """Task, counter, cost-sorted pool, model record and behaviour dump."""
    pool = pools.read_pool(ctx['pool_path'])
    if not pool['plans']:
        raise SkipTask(f"the pool holds no plans"
                       f"{' (the planner timed out)' if pool.get('timed_out') else ''}, "
                       f'so there is nothing to select from')
    task = pools.task_of(pool)
    counter = models.build_counter(spec, task, instance_info(cfg, pool), trace_cache=trace_cache)
    if loaded is None:
        loaded = pools.load_pool(ctx['pool_path'], counter=counter, task=task)
    record = models.model_record(spec, task, instance_info(cfg, pool))
    dump = pools.behaviour_dump(cfg, counter, loaded, record)
    return task, counter, loaded, record, dump


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

def result_path(cfg, experiment, task_id):
    """One file per task. The id's slashes become '__' so the directory stays
    flat and greppable; the id itself is inside the file."""
    tail = task_id.split('/', 1)[1].replace('/', '__')
    return results_root(cfg) / 'results' / experiment / f'{tail}.json'


def run_task(cfg, experiment, task_id, force=False):
    """Run one task and write its result file. Never raises for a task failure."""
    path = result_path(cfg, experiment, task_id)
    if path.is_file() and not force:
        return json.loads(path.read_text())
    result = {'schema': 'result', 'version': SCHEMA_VERSION, 'task_id': task_id,
              'experiment': experiment, 'config_hash': cfg['meta']['config_hash'],
              'git': git_revision(), 'started': now(), 'ended': None, 'error': None,
              'pool': None, 'model': None, 'rows': [], 'extra': {}}
    try:
        with time_limit(cfg['run']['time_limit_selection_s']):
            result.update(module(experiment).run_task(task_id, cfg))
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
    config_path, results_dir, experiment, task_id, force = arguments
    cfg = load(config_path, results_dir=results_dir)
    result = run_task(cfg, experiment, task_id, force=force)
    return task_id, result.get('error'), bool(result.get('extra', {}).get('skipped'))


def run(cfg, experiment, only=None, force=False, jobs=1, log=print):
    """Phase two over an experiment's tasks; resumable, one file each."""
    ids = [t for t in tasks(cfg, experiment) if only is None or t == only]
    if only is not None and not ids:
        raise ValueError(f"no task '{only}' in {experiment}")
    arguments = [(cfg['meta']['config_path'], cfg['run']['results_dir'], experiment, t, force)
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


def load_results(cfg, experiment):
    """Every result file of an experiment, in a stable order."""
    root = results_root(cfg) / 'results' / experiment
    return [json.loads(path.read_text()) for path in sorted(root.glob('*.json'))] if root.is_dir() else []


def load_dump(cfg, model_hash, domain, stem, pool_stem):
    path = (results_root(cfg) / 'behaviours' / model_hash / domain / stem / f'{pool_stem}.json')
    return json.loads(path.read_text()) if path.is_file() else None


#: The nine fields every row of every experiment carries, so that the reports
#: can group and pair on them without knowing which experiment wrote them.
BASE_FIELDS = ('instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b')


def base_row(loaded, dump, model_record, k=None, kappa=None):
    """The nine mandatory fields, ready to be extended with the row's own."""
    record = loaded['record']
    return {'instance': record['instance'], 'domain': record['domain'], 'q': record['q'],
            'N': record['requested'], 'model': model_record['name'], 'k': k, 'kappa': kappa,
            'pool_size': record['size'], 'b': len(dump['distinct'])}
