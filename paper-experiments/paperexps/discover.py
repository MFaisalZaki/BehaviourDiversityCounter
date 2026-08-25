"""Which FI pools an experiment's configuration selects.

Selection is done from the pool *files* alone -- their names, their sizes and
the behaviour-count ForbidIterative already recorded inside them. Nothing here
imports unified_planning or simulates a plan, so a sweep can be generated on a
laptop and only the compute nodes need carry the planning stack.

The recorded behaviour-count is computed over ForbidIterative's own gpo+cb
dimensions rather than the go+ru space these experiments use, so it is a proxy
and is only ever used as a *pre-filter* -- never as a reported number.
"""

import json
import os

from paperexps import common


class Pool:
    """One pool file and what its name and header say about it."""

    __slots__ = ('path', 'task_id', 'domain', 'year', 'inst', 'q', 'requested_k',
                 'n_plans', 'behaviour_count', 'has_resources')

    def __init__(self, path, fields, n_plans, behaviour_count, has_resources):
        self.path = path
        self.task_id = os.path.basename(path).replace('-fi-bc-results.json', '')
        self.domain = fields['domain']
        self.year = fields['year']
        self.inst = fields['inst']
        self.q = fields['q']
        self.requested_k = fields['k']
        self.n_plans = n_plans
        self.behaviour_count = behaviour_count
        self.has_resources = has_resources

    def __repr__(self):
        return (f'<Pool {self.task_id} n={self.n_plans} '
                f'b~{self.behaviour_count} resources={self.has_resources}>')


def read_header(path):
    """(n_plans, behaviour_count) for a pool file, or (0, None).

    Pool files of 48 bytes are planner timeouts; those are skipped rather than
    reported, which is what the sweep has always done.
    """
    if os.path.getsize(path) < 200:
        return 0, None
    try:
        with open(path) as handle:
            raw = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return 0, None
    plans = raw.get('plans') or []
    count = (raw.get('diversity-scores') or {}).get('behaviour-count')
    return len(plans), count


def discover(plans_dir, ru_info_dir=None, selection=None):
    """Every pool under ``plans_dir`` that the selection keeps.

    ``selection`` is an experiment's ``pools`` block; ``None`` keeps everything
    that holds plans.
    """
    selection = dict(selection or {})
    plans_dir = os.path.abspath(os.path.expanduser(plans_dir))
    if not os.path.isdir(plans_dir):
        raise FileNotFoundError(
            f'no pool directory at {plans_dir} '
            '(unzip data/fi-generated-plans-dir.zip, or pass --plans-dir)')

    requested_k = set(selection.get('requested-k') or [])
    include = {name.lower() for name in (selection.get('include-domains') or [])}
    exclude = {name.lower() for name in (selection.get('exclude-domains') or [])}
    min_b = int(selection.get('min-behaviour-count') or 0)
    min_plans = int(selection.get('min-plans') or 0)
    need_resources = bool(selection.get('require-resources'))

    found = []
    for name in sorted(os.listdir(plans_dir)):
        if not name.endswith('-fi-bc-results.json'):
            continue
        fields = common.parse_pool_filename(name)
        if fields is None:
            continue
        if requested_k and fields['k'] not in requested_k:
            continue
        if include and fields['domain'].lower() not in include:
            continue
        if fields['domain'].lower() in exclude:
            continue
        path = os.path.join(plans_dir, name)
        n_plans, behaviour_count = read_header(path)
        if n_plans < max(1, min_plans):
            continue
        if min_b and (behaviour_count or 0) < min_b:
            continue
        has_resources = bool(common.load_resources(
            ru_info_dir or common.DEFAULT_RU_INFO_DIR,
            fields['domain'], fields['year'], fields['inst'])) if ru_info_dir is not False else False
        if need_resources and not has_resources:
            continue
        found.append(Pool(path, fields, n_plans, behaviour_count, has_resources))

    return _cap(found, selection)


def _cap(pools, selection):
    """Apply max-pools-per-domain and max-pools.

    'even' spreads the cap across domains -- one pool from each, then a second
    from each, and so on -- so a capped sweep is a sample of the benchmark set
    rather than a sample of the alphabetically earliest domains.
    """
    per_domain = int(selection.get('max-pools-per-domain') or 0)
    total = int(selection.get('max-pools') or 0)
    even = (selection.get('selection') or 'even') == 'even'

    by_domain = {}
    for pool in pools:
        by_domain.setdefault(pool.domain, []).append(pool)
    if per_domain:
        for domain in by_domain:
            by_domain[domain] = by_domain[domain][:per_domain]

    if not total:
        return [pool for domain in sorted(by_domain) for pool in by_domain[domain]]

    if not even:
        flat = [pool for domain in sorted(by_domain) for pool in by_domain[domain]]
        return flat[:total]

    kept, index = [], 0
    while len(kept) < total and any(len(group) > index for group in by_domain.values()):
        for domain in sorted(by_domain):
            group = by_domain[domain]
            if len(group) > index and len(kept) < total:
                kept.append(group[index])
        index += 1
    return kept


def summarise(pools):
    """A short human summary of a pool list."""
    if not pools:
        return 'no pools'
    by_domain = {}
    for pool in pools:
        by_domain.setdefault(pool.domain, []).append(pool)
    counts = sorted((count for count in
                     (pool.behaviour_count or 0 for pool in pools)))
    middle = counts[len(counts) // 2] if counts else 0
    with_resources = sum(1 for pool in pools if pool.has_resources)
    return (f'{len(pools)} pools across {len(by_domain)} domains; '
            f'median recorded behaviour-count {middle}; '
            f'{with_resources} declare resources')
