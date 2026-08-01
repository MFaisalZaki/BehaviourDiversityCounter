"""Fold the per-pool result JSONs into one CSV per experiment.

Reads every JSON in the sandbox results directory and writes:

    exp1_collapse.csv        one row per pool: n, b, b/n, top multiplicity
    exp2_discrimination.csv  one row per (pool, m, BDC value) group
    exp3_cross.csv           one row per (pool, k, selector)
    exp3_summary.csv         mean scores per (k, selector) across pools
    exp4_greedy_gap.csv      one row per (pool, m)
    exp5_runtime.csv         one row per pool: sizes and timings
    skipped.csv              pools without a result and why

plus a plain-text summary of the headline numbers on stdout. Everything is
stdlib csv -- no pandas on the cluster.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common  # noqa: F401  (keeps the path bootstrap in one place)


def load_results(results_dir):
    results = []
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith('.json'):
            continue
        path = os.path.join(results_dir, name)
        try:
            with open(path) as handle:
                results.append(json.load(handle))
        except json.JSONDecodeError:
            print(f'note: unreadable result {name}', file=sys.stderr)
    return results


def write_csv(path, rows, fieldnames):
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {path} ({len(rows)} rows)')


def pool_key(result):
    meta = result.get('meta') or {}
    fields = result.get('filename_fields') or {}
    return {
        'pool': result['pool'],
        'domain': fields.get('domain') or meta.get('domain'),
        'year': fields.get('year') or meta.get('year'),
        'k_requested': fields.get('k') or meta.get('k_requested'),
    }


def aggregate(results, out_dir):
    exp1, exp2, exp3, exp4, exp5, skipped = [], [], [], [], [], []

    for result in results:
        key = pool_key(result)
        if 'skipped' in result:
            skipped.append({**key, 'reason': result['skipped']})
            continue

        meta = result['meta']
        if 'exp1' in result:
            data = result['exp1']
            multiplicities = data.get('multiplicities') or []
            exp1.append({
                **key,
                'n_plans': data['n_plans'],
                'b': data['b'],
                'b_over_n': round(data['b_over_n'], 6),
                'max_multiplicity': max(multiplicities, default=0),
                'bmaxsum': data.get('bmaxsum'),
                'maxsum_stability': data.get('maxsum_stability'),
                'has_resources': meta.get('resources_declared'),
                'fi_stored_behaviour_count': meta.get('fi_stored_behaviour_count'),
            })
        for record in result.get('exp2', []):
            exp2.append({**key, **{field: record.get(field) for field in (
                'm', 'v', 'exhaustive', 'n', 'min', 'q25', 'median', 'q75', 'max',
                'mean', 'std', 'spread_ratio', 'spread_range')}})
        for record in result.get('exp3', []):
            exp3.append({**key, **{field: record.get(field) for field in (
                'k', 'selector', 'bdc', 'bmaxsum', 'maxsum_stability',
                'maxsum_uniqueness', 'maxsum_states')}})
        for record in result.get('exp4', []):
            exp4.append({**key, **{field: record.get(field) for field in (
                'm', 'v', 'n_combos', 'exact', 'greedy', 'ratio',
                'bdc_greedy_value', 'bdc_greedy_expected', 'skipped')}})
        if 'exp5' in result:
            data = result['exp5']
            exp5.append({**key, **{field: data.get(field) for field in (
                'n_plans', 'b', 'mean_plan_length', 't_parse_task', 't_parse_plans',
                't_behaviour_extraction', 't_bdc_warm', 't_bmaxsum_cold',
                't_bmaxsum_warm', 't_maxsum_stability_cold')}})

    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, 'exp1_collapse.csv'), exp1,
              ['pool', 'domain', 'year', 'k_requested', 'n_plans', 'b', 'b_over_n',
               'max_multiplicity', 'bmaxsum', 'maxsum_stability', 'has_resources',
               'fi_stored_behaviour_count'])
    write_csv(os.path.join(out_dir, 'exp2_discrimination.csv'), exp2,
              ['pool', 'domain', 'year', 'k_requested', 'm', 'v', 'exhaustive', 'n',
               'min', 'q25', 'median', 'q75', 'max', 'mean', 'std',
               'spread_ratio', 'spread_range'])
    write_csv(os.path.join(out_dir, 'exp3_cross.csv'), exp3,
              ['pool', 'domain', 'year', 'k_requested', 'k', 'selector', 'bdc',
               'bmaxsum', 'maxsum_stability', 'maxsum_uniqueness', 'maxsum_states'])
    write_csv(os.path.join(out_dir, 'exp4_greedy_gap.csv'), exp4,
              ['pool', 'domain', 'year', 'k_requested', 'm', 'v', 'n_combos',
               'exact', 'greedy', 'ratio', 'bdc_greedy_value',
               'bdc_greedy_expected', 'skipped'])
    write_csv(os.path.join(out_dir, 'exp5_runtime.csv'), exp5,
              ['pool', 'domain', 'year', 'k_requested', 'n_plans', 'b',
               'mean_plan_length', 't_parse_task', 't_parse_plans',
               't_behaviour_extraction', 't_bdc_warm', 't_bmaxsum_cold',
               't_bmaxsum_warm', 't_maxsum_stability_cold'])
    write_csv(os.path.join(out_dir, 'skipped.csv'), skipped,
              ['pool', 'domain', 'year', 'k_requested', 'reason'])

    # Exp 3 summary: mean of every score per (k, selector), across pools that
    # have all selectors -- so the comparison is on the same pools.
    summary_rows = summarise_exp3(exp3)
    write_csv(os.path.join(out_dir, 'exp3_summary.csv'), summary_rows,
              ['k', 'selector', 'n_pools', 'mean_bdc', 'mean_bmaxsum',
               'mean_maxsum_stability', 'mean_maxsum_uniqueness', 'mean_maxsum_states'])

    print_headlines(exp1, exp2, exp3, exp4, skipped)


def summarise_exp3(exp3_rows):
    grouped = defaultdict(list)
    for row in exp3_rows:
        grouped[(row['k'], row['selector'])].append(row)
    summary = []
    for (k, selector), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        entry = {'k': k, 'selector': selector, 'n_pools': len(rows)}
        for score in ('bdc', 'bmaxsum', 'maxsum_stability', 'maxsum_uniqueness', 'maxsum_states'):
            values = [row[score] for row in rows if row.get(score) is not None]
            entry[f'mean_{score}'] = round(statistics.fmean(values), 4) if values else None
        summary.append(entry)
    return summary


def print_headlines(exp1, exp2, exp3, exp4, skipped):
    print()
    print('=== Headlines ===')
    if exp1:
        ratios = [row['b_over_n'] for row in exp1]
        collapsed = sum(1 for row in exp1 if row['b'] == 1)
        print(f'Exp 1: {len(exp1)} pools; median b/n = {statistics.median(ratios):.3f}; '
              f'{collapsed} pools ({collapsed / len(exp1):.0%}) collapse to a single behaviour.')
    if exp2:
        ratios = [row['spread_ratio'] for row in exp2 if row.get('spread_ratio')]
        if ratios:
            print(f'Exp 2: {len(ratios)} (pool, m, v) groups with a positive minimum; '
                  f'median within-group B-MaxSum max/min = {statistics.median(ratios):.2f}, '
                  f'90th percentile = {sorted(ratios)[int(0.9 * (len(ratios) - 1))]:.2f}.')
    if exp3:
        by_selector = defaultdict(list)
        for row in exp3:
            if row.get('bdc') is not None:
                by_selector[row['selector']].append(row['bdc'])
        parts = [f"{selector}: {statistics.fmean(values):.2f}"
                 for selector, values in sorted(by_selector.items())]
        print('Exp 3: mean BDC of selected sets by selector -> ' + '; '.join(parts))
    if exp4:
        ratios = [row['ratio'] for row in exp4 if row.get('ratio') is not None]
        checks = [row for row in exp4 if row.get('bdc_greedy_value') is not None]
        failed = sum(1 for row in checks
                     if row['bdc_greedy_value'] != row['bdc_greedy_expected'])
        if ratios:
            print(f'Exp 4: {len(ratios)} exact comparisons; median greedy/optimal = '
                  f'{statistics.median(ratios):.4f}, min = {min(ratios):.4f}.')
        print(f'Exp 4: BDC greedy-optimality check failed on {failed}/{len(checks)} cases '
              f'(Theorem bdc-greedy predicts 0).')
    if skipped:
        print(f'Skipped: {len(skipped)} pools without results.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_sandbox = os.path.join(common.PAPER_EXPERIMENTS_DIR, 'sandbox')
    parser.add_argument('--results-dir', default=os.path.join(default_sandbox, 'results'))
    parser.add_argument('--out-dir', default=os.path.join(default_sandbox, 'analysis'))
    args = parser.parse_args(argv)

    results = load_results(args.results_dir)
    if not results:
        print(f'No result JSONs in {args.results_dir}.', file=sys.stderr)
        return 1
    aggregate(results, args.out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
