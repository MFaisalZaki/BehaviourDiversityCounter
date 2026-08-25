"""One pool -> one task, its plans, its counter and its behaviours.

The three experiments differ in what they measure, not in how they get to a
pool's behaviour space, so the loading, the seeding, the per-dimension readout
and the CSV writing all live here. Every runner takes ``--pool`` one or more
times and writes the rows for exactly those pools, which is what lets one SLURM
array index handle one pool and ``aggregate.py`` concatenate afterwards.

The single most important thing this module enforces is that a task's pool is
loaded and its behaviours extracted **once**, and that every condition then
selects from that identical list. Nothing else may differ between conditions;
it is the entire reason the comparison is valid.
"""

import argparse
import csv
import os
import resource
import sys
import zlib

if __package__ in (None, ''):  # runnable as a plain script from anywhere
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common

#: The five conditions, in the order every table reports them.
SELECTORS = ['bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty', 'maxsum_stability']

#: The behaviour-space indicators, as scorer names.
INDICATORS = ['b_coverage', 'b_maxsum', 'b_maxmin', 'b_novelty']

#: Every dimension the paper's space can carry, so the CSV shape does not
#: change between a task that declares resources and one that does not.
SPREAD_DIMENSIONS = ['go', 'ru', 'cb']


class Pool:
    """A loaded pool: the task, its plans, its counter and its behaviours."""

    def __init__(self, task_id, fields, info, task, resources, plans, counter,
                 dimensions, behaviours, parse_failures, inapplicable, n_raw_plans):
        self.task_id = task_id
        self.task = task
        self.resources = resources
        self.fields = fields
        self.info = info
        self.plans = plans
        self.counter = counter
        self.dimensions = dimensions
        self.behaviours = behaviours
        self.parse_failures = parse_failures
        self.inapplicable = inapplicable
        self.n_raw_plans = n_raw_plans
        self.distinct = list(dict.fromkeys(behaviours))

    @property
    def domain(self):
        return self.fields['domain'] if self.fields else os.path.dirname(self.info['domain'])

    @property
    def year(self):
        return self.fields['year'] if self.fields else ''

    @property
    def q(self):
        return self.fields['q'] if self.fields else self.info.get('q')

    @property
    def n(self):
        return len(self.plans)

    @property
    def b(self):
        return len(self.distinct)

    def position(self):
        """plan id -> its index in the pool, for reporting selections."""
        return {id(plan): index for index, plan in enumerate(self.plans)}


class PoolSkipped(Exception):
    """A pool that cannot contribute rows, with the reason why."""


def task_id_of(pool_path):
    """The pool's identity in every CSV: its filename without the suffix."""
    return os.path.basename(pool_path).replace('-fi-bc-results.json', '')


def derive_seed(base_seed, task_id, salt):
    """A per-(task, purpose) seed derived from the run's one ``--seed``.

    Salted per purpose so that, for instance, Experiment B's hidden-preference
    draws are independent of its subsampling -- the brief asks for a seed
    "independent of everything else", and deriving it from a different salt is
    what makes that true while keeping the whole run reproducible from one
    number on the command line.
    """
    return (base_seed ^ zlib.crc32(f'{salt}:{task_id}'.encode())) & 0x7FFFFFFF


def load_pool(pool_path, args, workdir):
    """Parse one pool into a :class:`Pool`, or raise :class:`PoolSkipped`.

    Behaviour extraction happens here, once, and its wall-clock is returned
    alongside: Experiment A reports it apart from selection because it is by
    far the dominant cost, happens once per pool and is shared by all five
    conditions, so folding it into the selection time would hide the result.
    """
    import time

    from behaviour_diversity_counter import InapplicablePlanError

    task_id = task_id_of(pool_path)
    fields = common.parse_pool_filename(pool_path)
    raw = common.load_pool(pool_path)

    plan_strings = raw.get('plans') or []
    if not plan_strings or 'info' not in raw:
        raise PoolSkipped('pool holds no plans (planner timeout or failure)')

    info = raw['info']
    domain_file, problem_file = common.task_files(info, args.classical_domains)
    for path in (domain_file, problem_file):
        if not os.path.exists(path):
            raise PoolSkipped(f'task file not found: {path}')

    resources = common.load_resources(args.ru_info_dir, fields['domain'],
                                      fields['year'], fields['inst']) if fields else None

    task = common.parse_task(domain_file, problem_file)
    parsed, parse_failures = common.parse_plans(task, plan_strings)
    counter, dimensions = common.build_counter(task, resources, workdir)
    if args.count:
        counter.enable_counters()

    plans, behaviours, inapplicable = [], [], []
    start = time.perf_counter()
    for index, plan in enumerate(parsed):
        try:
            behaviours.append(counter._behaviours_of([plan])[0])
            plans.append(plan)
        except InapplicablePlanError as error:
            inapplicable.append({'index': index, 'error': str(error)})
    extract_seconds = time.perf_counter() - start

    if not plans:
        raise PoolSkipped('no plan survived parsing and simulation')

    pool = Pool(task_id, fields, info, task, resources, plans, counter, dimensions,
                behaviours, parse_failures, inapplicable, len(plan_strings))
    pool.extract_seconds = extract_seconds
    return pool


# ----------------------------------------------------------------------
# Reading a behaviour back out, per dimension
# ----------------------------------------------------------------------

def dimension_value(behaviour, name):
    """One dimension's value out of a behaviour string, by token prefix.

    Prefix, never substring: predicate and object names inside another
    dimension's token routinely contain a dimension's name (``truck1``
    contains ``ru``), which is the bug the dimensions themselves guard
    against.
    """
    for part in behaviour.split(' $$ '):
        part = part.strip()
        if part.startswith(name + ':'):
            return part[len(name) + 1:]
    return None


def behaviour_profile(behaviour, names):
    """{dimension name -> value} for one behaviour."""
    return {name: dimension_value(behaviour, name) for name in names}


def dimension_spread(counter, behaviours, name):
    """Mean pairwise distance along one dimension over distinct behaviours.

    Per dimension and never aggregated, and *raw* -- the dimension's own
    distance, not its weighted contribution -- so that the number means the
    same thing at every point of Experiment C's weight sweep.
    """
    distinct = list(dict.fromkeys(behaviours))
    dimension = counter.dimensions.get(name)
    if dimension is None or len(distinct) < 2:
        return ''
    total = pairs = 0.0
    for i in range(len(distinct)):
        for j in range(i + 1, len(distinct)):
            total += dimension.distance(distinct[i], distinct[j])
            pairs += 1
    return total / pairs


def jaccard(a, b):
    """|a & b| / |a | b|, two empty sets counting as identical."""
    union = len(a | b)
    return 1.0 if union == 0 else len(a & b) / union


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------

def peak_rss_mb():
    """Peak resident set size of this process, in MB.

    ru_maxrss is kilobytes on Linux. It is a high-water mark for the whole
    process, so it is reported per row as what the row's run had reached, not
    as the cost of that row alone.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def base_parser(description):
    """The arguments every runner shares."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--pool', action='append', default=[],
                        help='A *-fi-bc-results.json pool file. Repeatable.')
    parser.add_argument('--pools-file',
                        help='A file of pool paths, one per line.')
    parser.add_argument('--out', required=True, help='CSV to write.')
    parser.add_argument('--classical-domains', default=common.DEFAULT_CLASSICAL_DOMAINS,
                        help='classical-domains checkout (repo root or its classical/ dir).')
    parser.add_argument('--ru-info-dir', default=common.DEFAULT_RU_INFO_DIR,
                        help='Directory of per-domain (:resource ...) declarations.')
    parser.add_argument('--seed', type=int, default=2026,
                        help='The one seed every derived seed comes from.')
    parser.add_argument('--k-nn', type=int, default=3,
                        help="B-Novelty's neighbourhood size.")
    parser.add_argument('--dry-run', action='store_true',
                        help='Cut the sweep down to a quick end-to-end check.')
    parser.add_argument('--count', action='store_true',
                        help='Enable the distance/simulator counters.')
    return parser


def pools_from(args):
    paths = list(args.pool)
    if args.pools_file:
        with open(args.pools_file) as handle:
            paths += [line.strip() for line in handle if line.strip()]
    if not paths:
        raise SystemExit('error: no pools given; pass --pool or --pools-file')
    return paths
