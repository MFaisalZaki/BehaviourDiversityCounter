"""Experiment A: selection cost.

How does selection time grow with the pool size, for each rule? The k = 1000
pools are subsampled at 25, 50, 100, 200, 500 and 1000 -- seeded, and nested so
a smaller sample is a prefix of a larger one -- and all five selectors run at
k = 20 with five repeats at each size, followed by a sweep of k at the largest
size.

Behaviour extraction is reported apart from selection, and that separation is
the point of the experiment as much as the timings are. Extraction replays
every plan through the SequentialSimulator, is by far the dominant cost,
happens once per pool and is shared by all five conditions; folding it into the
selection time would hide the result and invite the objection that the
dimensions were chosen to be cheap. So ``extract_seconds`` and
``n_simulator_calls`` describe the extraction of *this row's* subsample, and
``select_seconds`` with the two distance counters describe the selection alone.

Each timed selection starts on a cold distance cache. The cache is exactly what
the claim is about -- there are only b behaviour strings to compare however many
plans exhibit them -- so a repeat that inherited a warm cache from the repeat
before would be timing nothing.
"""

import os
import random
import statistics
import sys
import tempfile
import time

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common, harness
from paperexps.baseline import Stability, greedy_maxsum_stability

FIELDS = [
    'task_id', 'domain', 'q', 'pool_size', 'k', 'dimensions', 'n_dims', 'selector',
    'repeat', 'seed', 'extract_seconds', 'select_seconds', 'n_distance_evals',
    'n_distance_misses', 'n_simulator_calls', 'b_achieved', 'peak_rss_mb',
    # Beyond the specified shape:
    'n_selected',            # B-MaxMin and B-Novelty return fewer than k
    'b_subsample',           # distinct behaviours in this row's subsample
    'extract_seconds_pool',  # the whole pool's extraction, for scale
    'n_pool', 'year',
]

DEFAULT_POOL_SIZES = [25, 50, 100, 200, 500, 1000]
DEFAULT_K_SWEEP = [5, 10, 20, 50]


def time_selection(plans, selector, k, k_nn, counter, stability_factory):
    """One cold-cache selection, timed, with the counters read off after."""
    # Reset before every selector, the baseline included: its row must report
    # the zero distance evaluations it actually makes -- it never touches the
    # behaviour space -- rather than inheriting the previous selector's count.
    counter._distance_cache.clear()
    counter.reset_counters()
    if selector == 'maxsum_stability':
        # A fresh Stability per repeat: its feature cache is the analogue of the
        # counter's distance cache, and carrying it over would time nothing.
        stability = stability_factory()
        start = time.perf_counter()
        selected = greedy_maxsum_stability(plans, k, stability)
        return time.perf_counter() - start, selected
    start = time.perf_counter()
    selected = counter.extract(plans, k=k, indicator=selector, k_nn=k_nn)
    return time.perf_counter() - start, selected


def run_pool(pool_path, args):
    """The rows for one pool: the size sweep, then the k sweep at the largest size."""
    rows = []
    with tempfile.TemporaryDirectory(prefix='paperexps-') as workdir:
        pool = harness.load_pool(pool_path, args, workdir)
        seed = harness.derive_seed(args.seed, pool.task_id, 'subsample')
        # One shuffle, so every subsample is a prefix of the next one up.
        order = list(range(pool.n))
        random.Random(seed).shuffle(order)

        sizes = [size for size in args.pool_sizes if size <= pool.n] or [pool.n]
        largest = max(sizes)

        for size in sizes:
            subsample = [pool.plans[index] for index in order[:size]]
            # A fresh counter, so this size's extraction is actually measured
            # rather than answered out of the pool-wide cache.
            counter, dimensions = common.build_counter(
                pool.task, pool.resources, workdir)
            counter.enable_counters()
            start = time.perf_counter()
            behaviours = counter._behaviours_of(subsample)
            extract_seconds = time.perf_counter() - start
            simulator_calls = counter.n_simulator_calls
            b_subsample = len(set(behaviours))

            ks = sorted(set(args.k_sweep + [args.k])) if size == largest else [args.k]
            for k in ks:
                for selector in harness.SELECTORS:
                    for repeat in range(args.repeats):
                        elapsed, selected = time_selection(
                            subsample, selector, k, args.k_nn, counter,
                            lambda: Stability(subsample, multiset=not args.set_stability))
                        rows.append({
                            'task_id': pool.task_id, 'domain': pool.domain, 'q': pool.q,
                            'pool_size': size, 'k': k,
                            'dimensions': '+'.join(dimensions), 'n_dims': len(dimensions),
                            'selector': selector, 'repeat': repeat, 'seed': seed,
                            'extract_seconds': extract_seconds,
                            'select_seconds': elapsed,
                            'n_distance_evals': counter.n_distance_evals,
                            'n_distance_misses': counter.n_distance_misses,
                            'n_simulator_calls': simulator_calls,
                            'b_achieved': counter.b_coverage(selected),
                            'peak_rss_mb': harness.peak_rss_mb(),
                            'n_selected': len(selected),
                            'b_subsample': b_subsample,
                            'extract_seconds_pool': pool.extract_seconds,
                            'n_pool': pool.n, 'year': pool.year,
                        })
    return rows


def summarise(rows):
    """Median select_seconds per (pool_size, k, selector), for the console."""
    groups = {}
    for row in rows:
        key = (row['pool_size'], row['k'], row['selector'])
        groups.setdefault(key, []).append(row['select_seconds'])
    return {key: statistics.median(values) for key, values in sorted(groups.items())}


def main(argv=None):
    parser = harness.base_parser(__doc__.splitlines()[0])
    parser.add_argument('--k', type=int, default=20,
                        help='Selection size for the pool-size sweep.')
    parser.add_argument('--pool-sizes', type=lambda t: [int(p) for p in t.split(',') if p],
                        default=DEFAULT_POOL_SIZES)
    parser.add_argument('--k-sweep', type=lambda t: [int(p) for p in t.split(',') if p],
                        default=DEFAULT_K_SWEEP)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--set-stability', action='store_true',
                        help='Baseline with A(p) as a set rather than a multiset.')
    args = parser.parse_args(argv)
    args.count = True  # the counters are columns of this experiment
    if args.dry_run:
        args.pool_sizes = [size for size in args.pool_sizes if size <= 100] or [25]
        args.k_sweep = [5, 20]
        args.repeats = 2

    rows, skipped = [], []
    for path in harness.pools_from(args):
        try:
            pool_rows = run_pool(path, args)
        except harness.PoolSkipped as reason:
            skipped.append((harness.task_id_of(path), str(reason)))
            print(f'{harness.task_id_of(path)}: skipped -- {reason}')
            continue
        rows += pool_rows
        biggest = max(row['pool_size'] for row in pool_rows)
        medians = summarise([r for r in pool_rows
                             if r['pool_size'] == biggest and r['k'] == args.k])
        parts = ' '.join(f'{selector}={seconds * 1000:.1f}ms'
                         for (_, _, selector), seconds in medians.items())
        print(f'{harness.task_id_of(path)}: pool_size={biggest} {parts}')

    harness.write_csv(args.out, FIELDS, rows)
    print(f'wrote {len(rows)} rows to {args.out}')
    if skipped:
        print(f'skipped {len(skipped)} pool(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
