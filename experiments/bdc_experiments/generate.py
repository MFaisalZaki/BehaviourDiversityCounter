"""Phase one of the two-phase scheme: the pools, out of the archive.

The paper's pools were produced once, by forbid-iterative in its top-quality
mode with the bound q * c*, and are shipped as
``experiments/data/fi-generated-plans-dir.zip``: one file per (q, instance)
named ``{q}-{k}-classical-{ipc}-{name}-{inst}-fi-bc-results.json``, where
``name`` and ``ipc`` are an ``api.py`` entry of the classical-domains checkout
and ``inst`` the 1-based position of the problem in that entry with its
problems sorted by path (``pfile1, pfile10, pfile11, ...``). A pool the
planner timed out on is a file holding only its total time.

Phase one matches each file to its PDDL, attaches the instance's
``(:resource ...)`` declarations from ``experiments/data/ru-info-dir`` (the
agents the domain models read; see :func:`resources_of` for how the tree
numbers its instances), and writes the pool record every later stage reads.
Nothing is planned here.
"""

import json
import re
import zipfile
from pathlib import Path

from bdc_experiments import SCHEMA_VERSION, benchmark
from bdc_experiments.config import results_root

MEMBER = re.compile(r'^(?P<q>[\d.]+)-(?P<k>\d+)-classical-(?P<ipc>[A-Za-z0-9]+)-'
                    r'(?P<name>.+)-(?P<inst>\d+)-fi-bc-results\.json$')

PLANNER = {'name': 'forbid-iterative', 'tag': 'fi-bc', 'mode': 'top-quality'}


def _data(cfg, section, key):
    """A data path of the config, relative to the config file; None if empty."""
    value = cfg[section][key]
    return (Path(cfg['meta']['config_path']).parent / value).resolve() if value else None


def archive(cfg):
    return _data(cfg, 'generation', 'archive')


def members(cfg):
    """``(name, ipc, inst) -> {(q, k): member}`` for every pool file of the
    archive at a configured q and pool size."""
    sizes, qs = set(cfg['generation']['pool_sizes']), set(cfg['generation']['q_values'])
    found = {}
    with zipfile.ZipFile(archive(cfg)) as bundle:
        for member in bundle.namelist():
            match = MEMBER.match(Path(member).name)
            if match is None or member.startswith('__MACOSX'):
                continue
            q, k = float(match['q']), int(match['k'])
            if q in qs and k in sizes:
                key = (match['name'], match['ipc'], int(match['inst']))
                found.setdefault(key, {})[(q, k)] = member
    return found


def declarations(cfg):
    """``(name, ipc) -> {inst: declarations}`` out of the ru-info tree, keyed
    by each file's own info block (agricola-opt18.json declares agricola/2018)."""
    root = _data(cfg, 'benchmark', 'resources')
    found = {}
    for path in sorted(root.rglob('*.json')) if root else []:
        data = json.loads(path.read_text())
        info = data.get('info') or {}
        found[(info.get('domain'), str(info.get('year')))] = data.get('instances') or {}
    return found


def _declares(text, problem_file):
    """Whether a declaration names exactly the problem's objects of its kind,
    the kind being the alphabetic prefix of the declared names (``truck1``,
    ``truck2`` against every ``truck<n>`` in the file)."""
    declared = {name.lower() for name in re.findall(r'\(:resource\s+(\S+)', text)}
    tokens = set(re.findall(r'[\w-]+', Path(problem_file).read_text().lower()))
    if not any(re.search(r'\d', name) for name in declared):
        return declared <= tokens
    prefixes = {re.match(r'[a-z_-]+', name).group() for name in declared}
    return declared == {t for t in tokens if any(re.fullmatch(p + r'\d+', t) for p in prefixes)}


def resources_of(instance, entries):
    """``(declarations, key)`` of an instance out of its domain's ru-info
    entries, or ``(None, None)``.

    The tree numbers instances by the number in the problem's file name
    (``pfile3`` is 3) in every domain but zenotravel, where it numbers them by
    sorted position as the archive does. The entry is looked up by the number
    first and the position second, and taken only where it names exactly the
    problem's objects of its kind, so a wrong entry is never attached.
    """
    digits = re.findall(r'\d+', instance['stem'])
    for key in dict.fromkeys(([int(digits[0])] if digits else []) + [instance['inst']]):
        text = entries.get(str(key))
        if text and _declares(text, instance['problem_file']):
            return text, key
    return None, None


def pool_stem(q, n):
    return f'topq-q{q}-N{n}'


def pool_path(cfg, instance, q, n):
    return (results_root(cfg) / 'pools' / instance['domain'] / instance['stem']
            / f'{pool_stem(q, n)}.json')


def read_plan(text):
    """``(actions, cost)`` out of one plan string; cost None when unstated."""
    actions = [line.strip() for line in text.splitlines() if line.strip().startswith('(')]
    stated = re.search(r';\s*(\d+)\s+cost|cost\s*=\s*(\d+)', text)
    return actions, (int(next(g for g in stated.groups() if g)) if stated else None)


def pool_record(instance, q, n, data, member, resources, entry=None, time_limit_s=None):
    """The pool record of one archive file.

    A run the driver killed is a file with no plans; a run the planner cut
    short at its time limit keeps the plans it had found and a total time at
    or over the limit. Both are timed out. A pool short of ``n`` inside the
    limit is exhausted: the planner proved no further plan within the bound.
    """
    seen, plans, duplicates = set(), [], 0
    for text in data.get('plans', []):
        actions, cost = read_plan(text)
        if tuple(actions) in seen:
            duplicates += 1
            continue
        seen.add(tuple(actions))
        plans.append({'actions': actions, 'cost': cost})
    plans.sort(key=lambda p: (p['cost'] if p['cost'] is not None else 0,))
    wall = data.get('total-time-seconds')
    timed_out = 'plans' not in data or (len(plans) < n and time_limit_s is not None
                                        and wall is not None and wall >= time_limit_s)
    return {
        'schema': 'pool', 'version': SCHEMA_VERSION,
        'instance': instance['id'], 'domain': instance['domain'], 'ipc': instance['ipc'],
        'domain_file': instance['domain_file'], 'problem_file': instance['problem_file'],
        'planner': {**PLANNER, 'archive_member': member, 'info': data.get('info')},
        'mode': 'topq', 'q': q, 'requested': n,
        'optimal_cost': min((p['cost'] for p in plans if p['cost'] is not None), default=None),
        'optimal_cost_source': ('cheapest plan of the pool: forbid-iterative plans the optimum '
                                'first and bounds every later plan by q times its cost'),
        'wall_s': wall, 'cpu_s': None,
        'timed_out': timed_out,
        'exhausted': (not timed_out) and len(plans) < n,
        'duplicates_dropped': duplicates,
        'resources': resources, 'resources_entry': entry,
        'plans': plans,
    }


def run(cfg, force=False, only=None, domain=None, log=print):
    """Phase one over the configured domains, capped at ``instances_per_domain``
    solved instances each, in instance order.

    An instance counts as solved when its first pool holds a plan; its other
    pools are still written, so a domain contributes the same instances at
    every q. ``only`` names one instance, ``domain`` one domain: the job
    arrays run one domain per element. Resumable: an existing pool file is
    left alone unless ``force``.
    """
    if archive(cfg) is None:
        raise SystemExit('[generation].archive is empty: this config runs on committed pools')
    found, declared = members(cfg), declarations(cfg)
    cap, solved, records = cfg['benchmark']['instances_per_domain'], {}, []
    with zipfile.ZipFile(archive(cfg)) as bundle:
        for instance in benchmark.instances(cfg):
            if (only is not None and instance['id'] != only) \
                    or (domain is not None and instance['domain'] != domain) \
                    or (only is None and solved.get(instance['domain'], 0) >= cap):
                continue
            files = found.get((instance['name'], instance['ipc'], instance['inst']))
            if not files:
                continue                    # the archive holds no pool of this instance
            resources, entry = resources_of(instance, declared.get((instance['name'], instance['ipc']), {}))
            for position, ((q, n), member) in enumerate(sorted(files.items())):
                out = pool_path(cfg, instance, q, n)
                if out.is_file() and not force:
                    record = json.loads(out.read_text())
                else:
                    record = pool_record(instance, q, n, json.loads(bundle.read(member)), member,
                                         resources, entry, cfg['generation']['time_limit_s'])
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(record, indent=1))
                records.append(record)
                if position == 0 and record['plans']:
                    solved[instance['domain']] = solved.get(instance['domain'], 0) + 1
                log(f"{instance['id']} {pool_stem(q, n)}: {len(record['plans'])} plans"
                    f"{' (timed out)' if record['timed_out'] else ''}"
                    f"{' (exhausted)' if record['exhausted'] else ''}"
                    f"{'' if resources else ' (no resource declarations)'}")
    return records
