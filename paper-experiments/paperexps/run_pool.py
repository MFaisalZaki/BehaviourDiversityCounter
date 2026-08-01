"""One FI plan pool -> one result JSON covering experiments 1-5.

Everything expensive about a pool -- parsing its task, replaying every plan
through the simulator -- happens exactly once here, and all five experiments
read off the same behaviour extraction:

- **Exp 1 (behaviour collapse)**: pool size, distinct-behaviour count b, and
  the behaviour multiplicity profile.
- **Exp 2 (discrimination among equal-BDC sets)**: B-MaxSum spread within
  groups of equal BDC. Because B-MaxSum depends only on the *distinct*
  behaviours a subset exhibits, a size-m plan subset with BDC = v is
  equivalent to a v-subset of the b distinct behaviours that some m plans can
  realise (total multiplicity >= m). Sampling behaviour subsets directly gives
  the exact same min/max/spread as sampling plan subsets, at a fraction of the
  cost; means/stds are over behaviour combinations, not plan subsets.
- **Exp 3 (cross-evaluation)**: select k plans by metric-based greedy MaxSum
  (Stability -- the ForbidIterative baseline -- plus optional States and
  Uniqueness), by Extract_BDC, Extract_BMaxSum, random and first-k; score
  every selection under both families.
- **Exp 4 (greedy gap for B-MaxSum)**: greedy Extract_BMaxSum against the
  exact optimum by enumeration over behaviour combinations (feasible whenever
  C(b, m) is within --exact-cap), plus the BDC greedy-optimality check of
  Theorem "bdc-greedy" (value must equal min(k, b)).
- **Exp 5 (runtime)**: behaviour extraction time, warm BDC time, cold/warm
  B-MaxSum time, and cold plan-level MaxSum-Stability time on the full pool.
"""

import argparse
import collections
import itertools
import json
import math
import os
import random
import statistics
import sys
import tempfile
import time
import zlib

if __package__ in (None, ''):  # runnable as a plain script from anywhere
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def summary_stats(values):
    """Compact distribution summary for a list of floats."""
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return {'n': 0}
    quantile = lambda q: ordered[min(n - 1, int(q * (n - 1) + 0.5))]
    return {
        'n': n,
        'min': ordered[0],
        'q25': quantile(0.25),
        'median': quantile(0.5),
        'q75': quantile(0.75),
        'max': ordered[-1],
        'mean': statistics.fmean(ordered),
        'std': statistics.pstdev(ordered) if n > 1 else 0.0,
    }


def pair_sum(distance, subset):
    return sum(distance(a, b) for a, b in itertools.combinations(subset, 2))


def time_call(fn, repeats=1):
    """(best wall-clock seconds over `repeats` runs, last result)."""
    best, result = None, None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return best, result


# ----------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------

def exp2_discrimination(distinct, multiplicities, distance, subset_sizes, samples, rng):
    """B-MaxSum spread within equal-BDC groups, per subset size m.

    For each m and each attainable BDC value v, gather (up to) `samples`
    feasible v-subsets of the distinct behaviours -- exhaustively when there
    are few enough combinations, by rejection sampling otherwise -- and
    summarise their B-MaxSum values.
    """
    b = len(distinct)
    total_plans = sum(multiplicities)
    records = []
    for m in subset_sizes:
        if total_plans < m:
            continue
        for v in range(1, min(m, b) + 1):
            combos = math.comb(b, v)
            feasible = lambda idx: sum(multiplicities[i] for i in idx) >= m
            values = []
            if combos <= samples:
                exhaustive = True
                for idx in itertools.combinations(range(b), v):
                    if feasible(idx):
                        values.append(pair_sum(distance, [distinct[i] for i in idx]))
            else:
                exhaustive = False
                attempts = 0
                while len(values) < samples and attempts < samples * 20:
                    attempts += 1
                    idx = rng.sample(range(b), v)
                    if feasible(idx):
                        values.append(pair_sum(distance, [distinct[i] for i in idx]))
            if not values:
                continue
            stats = summary_stats(values)
            # The paper's headline number: how far apart can two subsets with
            # the same BDC be? Guard the ratio against a zero minimum.
            stats['spread_ratio'] = (stats['max'] / stats['min']) if stats['min'] > 0 else None
            stats['spread_range'] = stats['max'] - stats['min']
            records.append({'m': m, 'v': v, 'exhaustive': exhaustive, **stats})
    return records


def exp3_cross_evaluation(task, plans, counter, select_ks, rng, with_states, with_uniqueness):
    """Select k plans per selector, score every selection under both families."""
    from plandiversity.shortcuts import GreedySolver, MaxSum, Stability, States, Uniqueness

    stability = Stability(task, plans)
    scorers = {'maxsum_stability': MaxSum(stability)}
    metric_selectors = {'metric-stability': stability}
    if with_uniqueness:
        uniqueness = Uniqueness(task, plans)
        scorers['maxsum_uniqueness'] = MaxSum(uniqueness)
        metric_selectors['metric-uniqueness'] = uniqueness
    if with_states:
        states = States(task, plans)
        scorers['maxsum_states'] = MaxSum(states)
        metric_selectors['metric-states'] = states

    position = {id(plan): index for index, plan in enumerate(plans)}
    records = []
    for k in select_ks:
        if len(plans) < k:
            continue
        selections = {}
        for name, metric in metric_selectors.items():
            selections[name] = list(GreedySolver(MaxSum(metric)).select(plans, k).plans)
        selections['extract-bdc'] = counter.extract(plans, k, indicator='bdc')
        selections['extract-bmaxsum'] = counter.extract(plans, k, indicator='bmaxsum')
        selections['random'] = rng.sample(plans, k)
        selections['first-k'] = plans[:k]  # the pool in FI generation order

        for selector, chosen in selections.items():
            record = {
                'k': k,
                'selector': selector,
                'indices': [position[id(plan)] for plan in chosen],
                'bdc': counter.bdc(chosen),
                'bmaxsum': counter.b_maxsum(chosen),
            }
            for scorer_name, model in scorers.items():
                record[scorer_name] = float(model(chosen))
            records.append(record)
    return records


def exp4_greedy_gap(plans, counter, distinct, distance, exact_sizes, exact_cap):
    """Greedy Extract_BMaxSum against the exact optimum, plus the BDC check.

    Exact optimum by enumeration: a size-m plan set maximising B-MaxSum
    exhibits v = min(m, b) distinct behaviours (B-MaxSum is monotone in
    behaviours and any v distinct behaviours are realisable with one plan
    each), so the optimum is the best pair-sum over C(b, v) combinations.
    """
    b = len(distinct)
    records = []
    for m in exact_sizes:
        if len(plans) < m:
            continue
        record = {'m': m, 'v': min(m, b)}

        greedy_set = counter.extract(plans, m, indicator='bmaxsum')
        record['greedy'] = counter.b_maxsum(greedy_set)

        bdc_set = counter.extract(plans, m, indicator='bdc')
        record['bdc_greedy_value'] = counter.bdc(bdc_set)
        record['bdc_greedy_expected'] = min(m, b)

        v = record['v']
        combos = math.comb(b, v)
        record['n_combos'] = combos
        if combos > exact_cap:
            record['skipped'] = f'C({b},{v}) = {combos} exceeds --exact-cap {exact_cap}'
        else:
            best = 0.0
            for idx in itertools.combinations(range(b), v):
                total = pair_sum(distance, [distinct[i] for i in idx])
                best = max(best, total)
            record['exact'] = best
            record['ratio'] = (record['greedy'] / best) if best > 0 else 1.0
        records.append(record)
    return records


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def run(args):
    pool_name = os.path.basename(args.pool).replace('.json', '')
    fields = common.parse_pool_filename(args.pool)
    pool = common.load_pool(args.pool)
    result = {'pool': pool_name, 'filename_fields': fields}

    plan_strings = pool.get('plans') or []
    if not plan_strings or 'info' not in pool:
        result['skipped'] = 'pool holds no plans (planner timeout or failure)'
        result['total_time_seconds'] = pool.get('total-time-seconds')
        return result

    info = pool['info']
    domain_file, problem_file = common.task_files(info, args.classical_domains)
    for path in (domain_file, problem_file):
        if not os.path.exists(path):
            result['skipped'] = f'task file not found: {path}'
            return result

    resources = common.load_resources(args.ru_info_dir, fields['domain'],
                                      fields['year'], fields['inst']) if fields else None

    timings = {}
    timings['parse_task'], task = time_call(lambda: common.parse_task(domain_file, problem_file))
    timings['parse_plans'], (plans, parse_failures) = time_call(
        lambda: common.parse_plans(task, plan_strings))

    with tempfile.TemporaryDirectory(prefix='paperexps-') as workdir:
        counter, dimensions = common.build_counter(task, resources, workdir)

        # --- Behaviour extraction: the timed pass that everything reuses. ---
        from behaviour_diversity_counter.behaviour_diversity_counter import InapplicablePlanError
        valid_plans, behaviours, inapplicable = [], [], []
        start = time.perf_counter()
        for index, plan in enumerate(plans):
            try:
                behaviours.append(counter._behaviours_of([plan])[0])
                valid_plans.append(plan)
            except InapplicablePlanError as error:
                inapplicable.append({'index': index, 'error': str(error)})
        timings['behaviour_extraction'] = time.perf_counter() - start

        result['meta'] = {
            'domain': fields['domain'] if fields else os.path.dirname(info['domain']),
            'year': fields['year'] if fields else None,
            'k_requested': fields['k'] if fields else pool['info'].get('k'),
            'q': fields['q'] if fields else pool['info'].get('q'),
            'domain_file': info['domain'],
            'problem_file': info['problem'],
            'dimensions': dimensions,
            'resources_declared': bool(resources),
            'n_raw_plans': len(plan_strings),
            'n_parse_failures': len(parse_failures),
            'n_inapplicable': len(inapplicable),
            'n_valid_plans': len(valid_plans),
            'fi_stored_behaviour_count': (pool.get('diversity-scores') or {}).get('behaviour-count'),
            'fi_total_time_seconds': pool.get('total-time-seconds'),
        }
        if parse_failures:
            result['parse_failures'] = parse_failures[:20]
        if inapplicable:
            result['inapplicable_plans'] = inapplicable[:20]
        if not valid_plans:
            result['skipped'] = 'no plan survived parsing and simulation'
            return result

        distinct = list(dict.fromkeys(behaviours))
        b = len(distinct)
        counts = collections.Counter(behaviours)
        multiplicities = [counts[behaviour] for behaviour in distinct]
        n = len(valid_plans)

        # --- Exp 5: indicator timings on the full pool. -------------------
        # Order matters: bdc on warm behaviour cache; the first b_maxsum call
        # pays the distance computations (cold), later calls are warm.
        timings['bdc_warm'], bdc_pool = time_call(lambda: counter.bdc(valid_plans), repeats=3)
        bmaxsum_pool = None
        if b <= args.max_b:
            timings['bmaxsum_cold'], bmaxsum_pool = time_call(lambda: counter.b_maxsum(valid_plans))
            timings['bmaxsum_warm'], _ = time_call(lambda: counter.b_maxsum(valid_plans), repeats=3)
        else:
            timings['bmaxsum_skipped'] = f'b = {b} exceeds --max-b {args.max_b}'

        from plandiversity.shortcuts import MaxSum, Stability
        timings['maxsum_stability_cold'], maxsum_stability_pool = time_call(
            lambda: float(MaxSum(Stability(task))(valid_plans)))

        result['exp5'] = {
            **{f't_{name}': value for name, value in timings.items()},
            'n_plans': n,
            'b': b,
            'mean_plan_length': statistics.fmean(len(plan.actions) for plan in valid_plans),
        }

        # --- Exp 1: behaviour collapse. -----------------------------------
        result['exp1'] = {
            'n_plans': n,
            'b': b,
            'b_over_n': b / n,
            'multiplicities': sorted(multiplicities, reverse=True),
            'bdc': bdc_pool,
            'bmaxsum': bmaxsum_pool,
            'maxsum_stability': maxsum_stability_pool,
        }
        if b <= args.max_stored_behaviours:
            result['exp1']['behaviours'] = distinct

        seed = args.seed ^ zlib.crc32(pool_name.encode())
        distance = counter._pair_distance

        # --- Exp 2: discrimination among equal-BDC subsets. ---------------
        result['exp2'] = exp2_discrimination(
            distinct, multiplicities, distance,
            subset_sizes=args.subset_sizes, samples=args.samples,
            rng=random.Random(seed))

        # --- Exp 3: cross-evaluation of selectors. ------------------------
        result['exp3'] = exp3_cross_evaluation(
            task, valid_plans, counter, select_ks=args.select_k,
            rng=random.Random(seed + 1),
            with_states=args.with_states, with_uniqueness=not args.no_uniqueness)

        # --- Exp 4: greedy gap for B-MaxSum. ------------------------------
        result['exp4'] = exp4_greedy_gap(
            valid_plans, counter, distinct, distance,
            exact_sizes=args.exact_sizes, exact_cap=args.exact_cap)

    return result


def parse_int_list(text):
    return [int(part) for part in text.split(',') if part.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--pool', required=True, help='Path to one *-fi-bc-results.json pool file.')
    parser.add_argument('--out', required=True, help='Path of the result JSON to write.')
    parser.add_argument('--classical-domains', default=common.DEFAULT_CLASSICAL_DOMAINS,
                        help='classical-domains checkout (repo root or its classical/ dir).')
    parser.add_argument('--ru-info-dir', default=common.DEFAULT_RU_INFO_DIR,
                        help='Directory of per-domain (:resource ...) declarations.')
    parser.add_argument('--subset-sizes', type=parse_int_list, default=[5, 10, 20],
                        help='Exp 2: subset sizes m (comma-separated).')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Exp 2: samples per (m, BDC value) group.')
    parser.add_argument('--select-k', type=parse_int_list, default=[5, 10],
                        help='Exp 3: selection sizes k (comma-separated).')
    parser.add_argument('--with-states', action='store_true',
                        help='Exp 3: add the States metric (costs a second simulation pass).')
    parser.add_argument('--no-uniqueness', action='store_true',
                        help='Exp 3: drop the Uniqueness metric.')
    parser.add_argument('--exact-sizes', type=parse_int_list, default=[5, 10],
                        help='Exp 4: subset sizes for the exact/greedy comparison.')
    parser.add_argument('--exact-cap', type=int, default=200_000,
                        help='Exp 4: enumeration budget on C(b, m) combinations.')
    parser.add_argument('--max-b', type=int, default=4000,
                        help='Skip full-pool B-MaxSum when b exceeds this (quadratic cost).')
    parser.add_argument('--max-stored-behaviours', type=int, default=200,
                        help='Store the behaviour strings themselves only up to this many.')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args(argv)

    result = run(args)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as handle:
        json.dump(result, handle, indent=1)
    status = result.get('skipped', 'ok')
    print(f"{result['pool']}: {status}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
