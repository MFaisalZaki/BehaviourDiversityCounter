"""Phase one of the two-phase scheme: a pool of plans per instance.

The planner is SymK, because the paper names top-k and top-quality planners as
the intended pool generators and SymK provides both. It is driven through its
own ``fast-downward.py``, with an explicit search string, rather than through
the unified-planning engine wrapper: the wrapper fixes the plan selector, and
the experiment needs the search string in the manifest.
"""

import glob
import json
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from bdc_experiments import SCHEMA_VERSION
from bdc_experiments.config import results_root

#: Verified against the README of the pinned SymK (up-symk 1.6.0): the engines
#: are `symq_bd` and `symk_bd` with underscores, and the selector is
#: `top_k(num_plans=..)`. The strings go into the manifest as they are used.
SEARCH = {
    'topq': 'symq_bd(plan_selection=top_k(num_plans={n},dump_plans=false),quality={q})',
    'topk': 'symk_bd(plan_selection=top_k(num_plans={n},dump_plans=false))',
}


def planner():
    """``(driver script, version)`` of the SymK the up-symk wheel ships."""
    import up_symk
    driver = Path(up_symk.__file__).parent / 'symk' / 'fast-downward.py'
    if not driver.is_file():
        raise FileNotFoundError(f'the up-symk wheel ships no driver at {driver}')
    return driver, metadata.version('up-symk')


def search_string(mode, q, n):
    if mode not in SEARCH:
        raise ValueError(f"unknown generation mode '{mode}'; valid: {sorted(SEARCH)}")
    return SEARCH[mode].format(n=n, q=q)


def pool_stem(mode, q, n):
    return f'{mode}-q{q}-N{n}'


def pool_path(cfg, instance, mode, q, n):
    return (results_root(cfg) / 'pools' / instance['domain'] / instance['stem']
            / f'{pool_stem(mode, q, n)}.json')


def read_plan_file(text):
    """``(actions, cost)`` out of one ``sas_plan.i``; cost None when unstated."""
    actions = [line.strip() for line in text.splitlines() if line.strip().startswith('(')]
    stated = re.search(r'^;\s*cost\s*=\s*(\d+)', text, re.MULTILINE)
    return actions, (int(stated.group(1)) if stated else None)


def _children_cpu():
    """CPU seconds this process's reaped children have used, user plus system.

    A child killed after a timeout is reaped by ``subprocess``, so its time is
    counted; one that outlives the kill is not, and the record says it timed
    out.
    """
    used = resource.getrusage(resource.RUSAGE_CHILDREN)
    return used.ru_utime + used.ru_stime


def _limits(cfg):
    """Applied in the child before exec: CPU seconds and address space."""
    def apply():
        seconds = cfg['run']['time_limit_generation_s']
        megabytes = cfg['run']['memory_limit_generation_mb']
        resource.setrlimit(resource.RLIMIT_CPU, (seconds, seconds + 5))
        resource.setrlimit(resource.RLIMIT_AS, (megabytes * 1024 * 1024,) * 2)
    return apply


def generate(cfg, instance, mode, q, n, force=False):
    """Run SymK once and write the pool file; return the pool record.

    Resumable: an existing pool file is returned untouched unless ``force``.
    A timeout is a recorded outcome, not an error, and the plans SymK had
    already written are kept.
    """
    out = pool_path(cfg, instance, mode, q, n)
    if out.is_file() and not force:
        return json.loads(out.read_text())

    driver, version = planner()
    search = search_string(mode, q, n)
    started = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory() as work:
        command = [sys.executable, str(driver), '--plan-file', 'sas_plan',
                   instance['domain_file'], instance['problem_file'], '--search', search]
        # The planner is a subprocess, so its CPU is the *children's* rusage and
        # not this process's. E3 compares the second phase's cost against the
        # first phase's, and time.process_time here would have measured the
        # parent waiting -- close to zero, and the comparison meaningless.
        # Generation is serial, so no other child is reaped inside the delta.
        wall = time.perf_counter()
        cpu = _children_cpu()
        timed_out, code, tail = False, None, ''
        try:
            done = subprocess.run(command, cwd=work, capture_output=True, text=True,
                                  timeout=cfg['run']['time_limit_generation_s'] + 60,
                                  preexec_fn=_limits(cfg), start_new_session=True)
            code, tail = done.returncode, done.stdout[-2000:]
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            tail = (expired.stdout or b'').decode(errors='replace')[-2000:]
            _kill(work)
        wall, cpu = time.perf_counter() - wall, _children_cpu() - cpu
        found = [read_plan_file(Path(f).read_text())
                 for f in sorted(glob.glob(os.path.join(work, 'sas_plan.*')),
                                 key=lambda f: int(f.rsplit('.', 1)[1]))]

    seen, plans, duplicates = set(), [], 0
    for actions, cost in found:
        key = tuple(actions)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        plans.append({'actions': actions, 'cost': cost})
    plans.sort(key=lambda p: (p['cost'] if p['cost'] is not None else 0,))

    optimal = min((p['cost'] for p in plans if p['cost'] is not None), default=None)
    record = {
        'schema': 'pool', 'version': SCHEMA_VERSION,
        'instance': instance['id'], 'domain': instance['domain'], 'ipc': instance['ipc'],
        'domain_file': instance['domain_file'], 'problem_file': instance['problem_file'],
        'planner': {'name': 'symk', 'version': version, 'search': search,
                    'exit_code': code, 'stdout_tail': tail},
        'mode': mode, 'q': q, 'requested': n,
        'optimal_cost': optimal,
        'optimal_cost_source': ('symk first plan of a top-quality pool (optimal)'
                                if mode == 'topq' else
                                'cheapest plan of a top-k pool (optimality not proven)'),
        'started': started, 'wall_s': wall, 'cpu_s': cpu,
        'timed_out': timed_out,
        # SymK stops either at num_plans or because no further plan exists
        # within the bound; a clean exit short of the request is the latter.
        'exhausted': (not timed_out) and code == 0 and len(plans) < n,
        'duplicates_dropped': duplicates,
        'plans': plans,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=1))
    return record


def _kill(work):
    """Best effort after a timeout: the driver leaves the search running."""
    for pattern in ('downward', 'fast-downward'):
        subprocess.run(['pkill', '-f', f'{pattern}.*{Path(work).name}'],
                       capture_output=True)


def combinations(cfg):
    """Every ``(mode, q, n)`` phase one is asked for: the selection sweep's
    pool sizes and the sizes E3 times the second phase against."""
    return [(mode, q, n)
            for mode in cfg['generation']['modes']
            for q in cfg['generation']['q_values']
            for n in sorted(set(cfg['generation']['pool_sizes']) | set(cfg['e3']['pool_sizes']))]


def run(cfg, instances, force=False, only=None, log=print):
    """Phase one over the instances, capped at ``instances_per_domain`` solved.

    An instance counts as solved when its first requested pool holds a plan;
    the remaining combinations still run, so a domain contributes the same
    instances at every ``q``.
    """
    cap = cfg['benchmark']['instances_per_domain']
    solved, records = {}, []
    for instance in instances:
        if only is not None and instance['id'] != only:
            continue
        if only is None and solved.get(instance['domain'], 0) >= cap:
            continue
        first = True
        for mode, q, n in combinations(cfg):
            record = generate(cfg, instance, mode, q, n, force=force)
            records.append(record)
            if first and record['plans']:
                solved[instance['domain']] = solved.get(instance['domain'], 0) + 1
            first = False
            log(f"{instance['id']} {pool_stem(mode, q, n)}: {len(record['plans'])} plans"
                f"{' (timed out)' if record['timed_out'] else ''}"
                f"{' (exhausted)' if record['exhausted'] else ''}")
    return records
