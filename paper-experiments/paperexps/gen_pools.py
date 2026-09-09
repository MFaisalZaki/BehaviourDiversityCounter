"""Generating pools with forbid-iterative, for a quality bound q.

A configuration entry with ``"kind": "generate"`` is a generation sweep: one
task per (domain, year, instance, k) of the benchmark, each running the
planner once and writing one pool file in the layout the evaluation reads,

    {q}-{k}-classical-{year}-{domain}-{inst}-{tag}-results.json

so the evaluation experiments pick the new pools up unchanged.  Three
generators are wired, all through ``python -m forbiditerative.plan`` (the
entry point IBM/forbiditerative's README asks to be run under subprocess):

    fi    ``--planner diverse``        forbid-iterative diverse planning, N plans
    topk  ``--planner topk``           the N cheapest plans
    topq  ``--planner topq_via_topk``  every plan within q times the optimal cost

Only ``topq`` knows the bound itself.  For the other two the pool is filtered
to cost <= q * c* afterwards, with c* the optimal cost -- read off an existing
q = 1.0 pool of the instance when one is on disk, and otherwise from one
extra ``topk`` call for a single (optimal) plan.  Every pool records where
its c* came from, the command it ran, and the planner's last lines of output.

Plans are collected from the plan files the planner leaves in its working
directory (``found_plans/done/sas_plan.*``, Fast Downward's format with its
``; cost = N`` comment), which is the one format that is certain; a
``results.json`` written by ``--plans-as-json`` is read as a fallback.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from utils import RESULTS_SUFFIX, POOL_RE, _domain_index, resolve

PLANNER_OF = {'fi': 'diverse', 'topk': 'topk', 'topq': 'topq_via_topk'}
DEFAULT_TAG = {'fi': 'fi-bc', 'topk': 'topk', 'topq': 'topq'}
COST_RE = re.compile(r'cost\s*=\s*(\d+)')


def pool_file_name(q, k, year, domain, inst, tag, track='classical'):
    return f'{q}-{k}-{track}-{year}-{domain}-{inst}-{tag}{RESULTS_SUFFIX}'


def _pick_directory(directories):
    """One benchmark directory per (domain, year): the optimal track where an
    IPC year ran two, since its instances are the ones optimal planners solve."""
    names = sorted(directories)
    return next((n for n in names if 'opt' in n), names[0])


def list_tasks(params):
    """One task per (domain, year, instance, k) of the configured domains."""
    problems_root = resolve(params['benchmark'])
    if os.path.isdir(os.path.join(problems_root, 'classical')):
        problems_root = os.path.join(problems_root, 'classical')
    if not os.path.isdir(problems_root):
        raise FileNotFoundError(f'no benchmark directory at {problems_root}')
    index = _domain_index(problems_root)
    wanted = set(params.get('domains') or [])
    tag = params.get('tag') or DEFAULT_TAG[params['generator']]
    tasks, missing = [], sorted(wanted - {key[0] for key in index})
    for (domain, year), directories in sorted(index.items()):
        if wanted and domain not in wanted:
            continue
        directory = _pick_directory(directories)
        for inst, (domain_path, problem_path) in enumerate(directories[directory], start=1):
            for k in params['pool-k-values']:
                pool = pool_file_name(params['q'], k, year, domain, inst, tag)
                tasks.append({
                    'task_id': pool[:-len(RESULTS_SUFFIX)], 'dumpfile': pool,
                    'domain': domain, 'year': year, 'inst': inst, 'q': params['q'], 'k': k,
                    'track': 'classical', 'generator': tag, 'generator_name': params['generator'],
                    'domain_dir': directory,
                    'domain_file': os.path.join(problems_root, domain_path),
                    'problem_file': os.path.join(problems_root, problem_path),
                    'domain_rel': domain_path, 'problem_rel': problem_path,
                })
    if missing:
        print(f'gen_pools: no api.py entry for domain(s) {missing}', file=sys.stderr)
    return tasks


# ------------------------------------------------------------- the planner --

def planner_command(params, planner, domain_file, problem_file, number_of_plans=None, quality_bound=None):
    command = list(params.get('fi-command') or [sys.executable, '-m', 'forbiditerative.plan'])
    command += ['--planner', planner, '--domain', os.path.abspath(domain_file),
                '--problem', os.path.abspath(problem_file)]
    if number_of_plans is not None:
        command += ['--number-of-plans', str(number_of_plans)]
    if quality_bound is not None:
        command += ['--quality-bound', str(quality_bound)]
    if params.get('time-limit'):
        command += ['--overall-time-limit', str(params['time-limit'])]
    if params.get('memory-limit'):
        command += ['--overall-memory-limit', str(params['memory-limit'])]
    if planner != 'diverse' and params.get('symmetries', True):
        command.append('--symmetries')
    command += ['--suppress-planners-output', '--plans-as-json', '--results-file', 'results.json']
    command += list(params.get('extra-args') or [])
    return command


def read_plan_file(path):
    """``(text, cost)`` of a Fast Downward plan file; cost None when the file
    carries no ``; cost = N`` comment."""
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    match = COST_RE.search(text)
    return text.strip() + '\n', (int(match.group(1)) if match else None)


def collect_plans(workdir):
    """The plans a planner run left behind, in the order it found them."""
    def numbered(paths):
        def key(path):
            match = re.search(r'(\d+)$', os.path.basename(path))
            return (int(match.group(1)) if match else 0, path)
        return sorted(paths, key=key)
    for pattern in ('found_plans/done/sas_plan*', 'found_plans/**/sas_plan*', 'sas_plan*'):
        paths = numbered(glob.glob(os.path.join(workdir, pattern), recursive=True))
        if paths:
            return [read_plan_file(path) for path in paths], pattern
    results = os.path.join(workdir, 'results.json')
    if os.path.isfile(results):
        with open(results) as handle:
            data = json.load(handle)
        plans = []
        for plan in data.get('plans') or []:
            if isinstance(plan, str):
                plans.append((plan, None))
            elif isinstance(plan, dict):
                actions = plan.get('actions') or plan.get('plan') or []
                text = '\n'.join(a if a.startswith('(') else f'({a})' for a in actions)
                cost = plan.get('cost')
                plans.append((text + (f'\n; cost = {cost} (from results.json)\n' if cost is not None else '\n'), cost))
        return plans, 'results.json'
    return [], None


def run_planner(command, workdir, timeout_s, log_lines=40):
    started = time.time()
    try:
        completed = subprocess.run(command, cwd=workdir, capture_output=True, text=True, timeout=timeout_s)
        status, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        status, stdout, stderr = 'timeout', (error.stdout or ''), (error.stderr or '')
        if isinstance(stdout, bytes):
            stdout, stderr = stdout.decode(errors='replace'), (stderr or b'').decode(errors='replace')
    except FileNotFoundError as error:
        status, stdout, stderr = 'not-found', '', str(error)
    elapsed = time.time() - started
    logs = (stdout.splitlines()[-log_lines:] + ['--- stderr ---'] + stderr.splitlines()[-log_lines:])
    return status, elapsed, logs


def optimal_cost_from_pools(taskdetails, params):
    """c* from a q = 1.0 pool of the same instance already on disk, by replaying
    it; None when there is none."""
    plans_root = resolve(params['plansdir'])
    if not os.path.isdir(plans_root):
        return None, None
    t = taskdetails
    pattern = os.path.join(plans_root, f"1.0-*-{t['track']}-{t['year']}-{t['domain']}-{t['inst']}-*{RESULTS_SUFFIX}")
    for path in sorted(glob.glob(pattern), key=os.path.getsize, reverse=True):
        if os.path.getsize(path) <= 48:
            continue
        match = POOL_RE.match(os.path.basename(path))
        if match is None:
            continue
        from harness import load_pool
        details = {**t, 'pool_file': path, 'q': 1.0, 'k': int(match.group('k')),
                   'generator': match.group('generator'), 'generator_name': match.group('generator'),
                   'resources': None, 'resolved_by': 'generation'}
        pool = load_pool(details)
        if pool.optimal_cost is not None:
            return int(pool.optimal_cost) if float(pool.optimal_cost).is_integer() else float(pool.optimal_cost), os.path.basename(path)
    return None, None


def optimal_cost_from_planner(taskdetails, params, timeout_s):
    """c* from one cost-optimal plan: ``--planner topk --number-of-plans 1``."""
    workdir = tempfile.mkdtemp(prefix='bdc-cstar-')
    try:
        command = planner_command(params, 'topk', taskdetails['domain_file'], taskdetails['problem_file'], number_of_plans=1)
        status, elapsed, logs = run_planner(command, workdir, timeout_s)
        plans, _ = collect_plans(workdir)
        costs = [cost for _, cost in plans if cost is not None]
        return (min(costs) if costs else None), {'status': status, 'seconds': elapsed, 'command': command, 'logs': logs}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_task(taskdetails, params):
    """Generate one pool; returns the JSON written."""
    generator = params['generator']
    planner = PLANNER_OF[generator]
    q, k = float(params['q']), int(taskdetails['k'])
    number = min(int(params.get('pool-factor', 10)) * k, int(params.get('pool-cap', 1000)))
    timeout_s = float(params.get('subprocess-timeout-s', 0)) or None
    info = {'domain': taskdetails['domain_rel'], 'problem': os.path.basename(taskdetails['problem_rel']),
            'planner': generator, 'tag': taskdetails['generator'], 'planning-type': 'classical',
            'k': k, 'q': q, 'requested-plans': number if planner != 'topq_via_topk' else None,
            'time-limit': params.get('time-limit'), 'memory-limit': params.get('memory-limit')}
    logs, total = [], 0.0

    optimal, source = None, None
    if planner != 'topq_via_topk':
        optimal, source = optimal_cost_from_pools(taskdetails, params)
        if optimal is None:
            optimal, run = optimal_cost_from_planner(taskdetails, params, timeout_s)
            total += run['seconds']
            source = 'topk --number-of-plans 1'
            logs += [f'[c*] {run["status"]} in {run["seconds"]:.1f}s: {" ".join(run["command"])}'] + run['logs']
        else:
            source = f'replayed {source}'

    workdir = tempfile.mkdtemp(prefix='bdc-gen-')
    try:
        command = planner_command(params, planner, taskdetails['domain_file'], taskdetails['problem_file'],
                                  number_of_plans=None if planner == 'topq_via_topk' else number,
                                  quality_bound=q if planner == 'topq_via_topk' else None)
        status, elapsed, run_logs = run_planner(command, workdir, timeout_s)
        total += elapsed
        found, where = collect_plans(workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    logs += [f'[pool] {status} in {elapsed:.1f}s: {" ".join(command)}', f'[pool] plans read from {where}'] + run_logs

    if planner == 'topq_via_topk' and found:
        costs = [cost for _, cost in found if cost is not None]
        optimal, source = (min(costs) if costs else None), 'cheapest plan of the top-quality pool'

    plans, filtered, unknown = [], 0, 0
    bound = None if optimal is None else q * optimal
    for text, cost in found:
        if cost is None:
            unknown += 1
            plans.append(text)
        elif bound is None or cost <= bound + 1e-9:
            plans.append(text)
        else:
            filtered += 1
    info.update({'optimal-cost': optimal, 'optimal-cost-source': source, 'cost-bound': bound,
                 'found-plans': len(found), 'filtered-out': filtered, 'plans-without-cost': unknown,
                 'planner-status': status, 'command': command})
    return {'plans': plans, 'diversity-scores': {}, 'info': info, 'logs': logs, 'total-time-seconds': total}


def run(tasks, basedir, params, force=False):
    for taskdetails in tasks:
        target = os.path.join(basedir, taskdetails['dumpfile'])
        if os.path.exists(target) and not force:
            print(f"| skipping {taskdetails['task_id']}: {target} exists")
            continue
        print(f"| Generating {taskdetails['task_id']}")
        try:
            pool = run_task(taskdetails, params)
        except Exception as error:                                          # noqa: BLE001
            import traceback
            pool = {'plans': [], 'diversity-scores': {}, 'info': {
                'domain': taskdetails['domain_rel'], 'problem': os.path.basename(taskdetails['problem_rel']),
                'planner': params['generator'], 'tag': taskdetails['generator'], 'planning-type': 'classical',
                'k': taskdetails['k'], 'q': params['q'], 'error': f'{type(error).__name__}: {error}'},
                'logs': traceback.format_exc().splitlines(), 'total-time-seconds': None}
            print(pool['logs'][-1], file=sys.stderr)
        os.makedirs(basedir, exist_ok=True)
        with open(target, 'w') as handle:
            json.dump(pool, handle, indent=1)
        print(f"|   {len(pool['plans'])} plans -> {target}")
