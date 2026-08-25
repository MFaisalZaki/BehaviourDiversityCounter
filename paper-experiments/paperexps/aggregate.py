"""The three experiments' CSVs -> summary tables and the headline numbers.

Reads whatever ``--exp-a``, ``--exp-b-scores``, ``--exp-b-overlap`` and
``--exp-c`` point at -- a file, several files, or a directory of the per-pool
fragments a SLURM array leaves behind -- concatenates them and writes one
summary CSV per table into ``--analysis-dir``.

Every Experiment B and C comparison is a Wilcoxon signed-rank test, paired per
task, at p < 0.05, Holm-corrected across the selectors compared against the
same baseline.

Experiment B is reported twice: over every task, and over the tasks whose pool
holds at least k distinct behaviours. That split is not a filter for a nicer
number, it is the result. A pool with b < k forces k - b of the redundancy on
every rule alive, so on such a task no selection rule can differ from any other
and the medians over all tasks measure the pools rather than the rules. Both
tables are written, and the headline reports both.
"""

import argparse
import collections
import csv
import glob
import os
import statistics
import sys

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common, harness
from paperexps.stats import holm_bonferroni, median, wilcoxon

BASELINE = 'maxsum_stability'


def read_rows(patterns):
    """Every row of every CSV the patterns name, concatenated."""
    rows = []
    for pattern in patterns or []:
        paths = []
        if os.path.isdir(pattern):
            paths = sorted(glob.glob(os.path.join(pattern, '*.csv')))
        else:
            paths = sorted(glob.glob(pattern)) or ([pattern] if os.path.exists(pattern) else [])
        for path in paths:
            with open(path, newline='') as handle:
                rows.extend(csv.DictReader(handle))
    return rows


def number(row, column):
    value = row.get(column, '')
    if value in ('', None):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def column(rows, name):
    return [value for value in (number(row, name) for row in rows) if value is not None]


def per_task(rows, selector, name):
    """{task -> value} for one selector and one column."""
    return {row['task_id']: number(row, name)
            for row in rows if row['selector'] == selector and number(row, name) is not None}


def paired_tests(rows, columns, selectors, baseline=BASELINE):
    """Wilcoxon against the baseline, per column, Holm-corrected per column."""
    results = []
    for name in columns:
        reference = per_task(rows, baseline, name)
        block, p_values = [], []
        for selector in selectors:
            if selector == baseline:
                continue
            ours = per_task(rows, selector, name)
            shared = sorted(set(ours) & set(reference))
            differences = [ours[task] - reference[task] for task in shared]
            statistic, p, used = wilcoxon(differences)
            block.append({
                'column': name, 'selector': selector, 'baseline': baseline,
                'n_tasks': len(shared), 'n_nonzero_pairs': used,
                'median_ours': median([ours[t] for t in shared]) if shared else '',
                'median_baseline': median([reference[t] for t in shared]) if shared else '',
                'median_difference': median(differences) if differences else '',
                'wins': sum(1 for d in differences if d > 0),
                'ties': sum(1 for d in differences if d == 0),
                'losses': sum(1 for d in differences if d < 0),
                'statistic': statistic, 'p_value': p,
            })
            p_values.append(p)
        adjusted = holm_bonferroni(p_values) if p_values else []
        for entry, value in zip(block, adjusted):
            entry['p_holm'] = value
            entry['significant'] = value < 0.05
        results.extend(block)
    return results


# ----------------------------------------------------------------------
# Experiment A
# ----------------------------------------------------------------------

EXP_A_SUMMARY_FIELDS = [
    'pool_size', 'k', 'selector', 'n_tasks', 'n_rows',
    'median_select_seconds', 'mean_select_seconds', 'max_select_seconds',
    'median_extract_seconds', 'extract_over_select',
    'median_distance_evals', 'median_distance_misses', 'median_simulator_calls',
    'median_b_achieved', 'median_n_selected', 'median_peak_rss_mb',
]


def summarise_exp_a(rows):
    """Medians over repeats, per (pool_size, k, selector). No significance test
    on wall-clock: these are timings on one machine, not a sample from a
    population, and a p-value would dress them up as something they are not."""
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(number(row, 'pool_size'), number(row, 'k'), row['selector'])].append(row)
    out = []
    for (pool_size, k, selector), group in sorted(groups.items()):
        # Median over repeats within a task first, then over tasks, so a task
        # with more repeats does not weigh more than one with fewer.
        by_task = collections.defaultdict(list)
        for row in group:
            by_task[row['task_id']].append(row)
        task_medians = {task: median(column(rows_, 'select_seconds'))
                        for task, rows_ in by_task.items()}
        extract = [median(column(rows_, 'extract_seconds')) for rows_ in by_task.values()]
        select = list(task_medians.values())
        out.append({
            'pool_size': int(pool_size), 'k': int(k), 'selector': selector,
            'n_tasks': len(by_task), 'n_rows': len(group),
            'median_select_seconds': median(select),
            'mean_select_seconds': statistics.fmean(select) if select else '',
            'max_select_seconds': max(select) if select else '',
            'median_extract_seconds': median(extract),
            'extract_over_select': (median(extract) / median(select)) if median(select) else '',
            'median_distance_evals': median(column(group, 'n_distance_evals')),
            'median_distance_misses': median(column(group, 'n_distance_misses')),
            'median_simulator_calls': median(column(group, 'n_simulator_calls')),
            'median_b_achieved': median(column(group, 'b_achieved')),
            'median_n_selected': median(column(group, 'n_selected')),
            'median_peak_rss_mb': median(column(group, 'peak_rss_mb')),
        })
    return out


# ----------------------------------------------------------------------
# Experiment B
# ----------------------------------------------------------------------

EXP_B_SUMMARY_FIELDS = [
    'stratum', 'selector', 'n_tasks',
    'median_n_selected', 'median_b_achieved', 'median_redundancy',
    'median_redundancy_floor', 'median_redundancy_excess',
    'median_b_coverage', 'median_b_maxsum', 'median_b_maxmin', 'median_b_novelty',
    'median_maxsum_stability', 'median_spread_go', 'median_spread_ru',
    'median_hidden_pref_rate', 'mean_hidden_pref_rate',
    'total_hidden_hits', 'total_hidden_draws', 'pooled_hidden_rate',
]

EXP_B_TESTED = ['hidden_pref_rate', 'b_coverage', 'b_maxsum', 'b_maxmin',
                'b_novelty', 'maxsum_stability', 'redundancy_excess']


def strata(rows):
    """(name, rows) for every task, and for the tasks a rule can act on."""
    yield 'all', rows
    discriminating = [row for row in rows
                      if (number(row, 'b_pool') or 0) >= (number(row, 'k') or 0)]
    yield 'b_pool>=k', discriminating


def summarise_exp_b(rows):
    out = []
    for name, subset in strata(rows):
        for selector in harness.SELECTORS:
            group = [row for row in subset if row['selector'] == selector]
            if not group:
                continue
            hits = sum(column(group, 'hidden_pref_hits'))
            draws = sum(column(group, 'hidden_pref_draws'))
            out.append({
                'stratum': name, 'selector': selector, 'n_tasks': len(group),
                'median_n_selected': median(column(group, 'n_selected')),
                'median_b_achieved': median(column(group, 'b_achieved')),
                'median_redundancy': median(column(group, 'redundancy')),
                'median_redundancy_floor': median(column(group, 'redundancy_floor')),
                'median_redundancy_excess': median(column(group, 'redundancy_excess')),
                'median_b_coverage': median(column(group, 'b_coverage')),
                'median_b_maxsum': median(column(group, 'b_maxsum')),
                'median_b_maxmin': median(column(group, 'b_maxmin')),
                'median_b_novelty': median(column(group, 'b_novelty')),
                'median_maxsum_stability': median(column(group, 'maxsum_stability')),
                'median_spread_go': median(column(group, 'spread_go')),
                'median_spread_ru': median(column(group, 'spread_ru')),
                'median_hidden_pref_rate': median(column(group, 'hidden_pref_rate')),
                'mean_hidden_pref_rate': statistics.fmean(
                    column(group, 'hidden_pref_rate')) if group else '',
                'total_hidden_hits': hits, 'total_hidden_draws': draws,
                'pooled_hidden_rate': (hits / draws) if draws else '',
            })
    return out


def summarise_exp_b_tests(rows):
    out = []
    for name, subset in strata(rows):
        for entry in paired_tests(subset, EXP_B_TESTED, harness.SELECTORS):
            out.append({'stratum': name, **entry})
    return out


EXP_B_OVERLAP_FIELDS = ['selector_i', 'selector_j', 'n_tasks',
                        'median_jaccard', 'mean_jaccard',
                        'median_jaccard_plans', 'mean_jaccard_plans',
                        'behaviour_minus_plan']


def summarise_exp_b_overlap(rows):
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row['selector_i'], row['selector_j'])].append(row)
    out = []
    for (a, b), group in sorted(groups.items()):
        behaviour = column(group, 'jaccard')
        plans = column(group, 'overlap_jaccard_plans')
        out.append({
            'selector_i': a, 'selector_j': b, 'n_tasks': len(group),
            'median_jaccard': median(behaviour),
            'mean_jaccard': statistics.fmean(behaviour) if behaviour else '',
            'median_jaccard_plans': median(plans),
            'mean_jaccard_plans': statistics.fmean(plans) if plans else '',
            'behaviour_minus_plan': (median(behaviour) - median(plans))
                                    if behaviour and plans else '',
        })
    return out


# ----------------------------------------------------------------------
# Experiment C
# ----------------------------------------------------------------------

EXP_C_SUMMARY_FIELDS = [
    'task_id', 'domain', 'n_pool', 'b_pool', 'ru_values_in_pool', 'go_values_in_pool',
    'n_steps', 'first_change_w_go', 'first_change_w_go_plans',
    'n_distinct_behaviour_sets', 'n_distinct_plan_sets',
    'spread_go_at_0', 'spread_go_at_1', 'spread_go_delta',
    'spread_ru_at_0', 'spread_ru_at_1', 'spread_ru_delta',
    'go_rises', 'ru_falls', 'paper_prediction_holds',
]


def summarise_exp_c(rows):
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row['task_id']].append(row)
    out = []
    for task, group in sorted(by_task.items()):
        group.sort(key=lambda row: number(row, 'w_go'))
        changed = [row for row in group if row['changed_from_baseline'] in ('True', 'true', '1')]
        changed_plans = [row for row in group
                         if row['changed_from_baseline_plans'] in ('True', 'true', '1')]
        go_0, go_1 = number(group[0], 'spread_go'), number(group[-1], 'spread_go')
        ru_0, ru_1 = number(group[0], 'spread_ru'), number(group[-1], 'spread_ru')
        go_delta = (go_1 - go_0) if None not in (go_0, go_1) else None
        ru_delta = (ru_1 - ru_0) if None not in (ru_0, ru_1) else None
        go_rises = go_delta is not None and go_delta > 0
        ru_falls = ru_delta is not None and ru_delta < 0
        out.append({
            'task_id': task, 'domain': group[0]['domain'],
            'n_pool': group[0].get('n_pool', ''), 'b_pool': group[0].get('b_pool', ''),
            'ru_values_in_pool': group[0].get('ru_values_in_pool', ''),
            'go_values_in_pool': group[0].get('go_values_in_pool', ''),
            'n_steps': len(group),
            'first_change_w_go': number(changed[0], 'w_go') if changed else '',
            'first_change_w_go_plans': number(changed_plans[0], 'w_go') if changed_plans else '',
            'n_distinct_behaviour_sets': number(group[-1], 'n_distinct_behaviour_sets_so_far'),
            'n_distinct_plan_sets': number(group[-1], 'n_distinct_plan_sets_so_far'),
            'spread_go_at_0': go_0, 'spread_go_at_1': go_1,
            'spread_go_delta': go_delta if go_delta is not None else '',
            'spread_ru_at_0': ru_0, 'spread_ru_at_1': ru_1,
            'spread_ru_delta': ru_delta if ru_delta is not None else '',
            'go_rises': go_rises, 'ru_falls': ru_falls,
            'paper_prediction_holds': go_rises and ru_falls,
        })
    return out


EXP_C_TEST_FIELDS = ['comparison', 'n_tasks', 'n_nonzero_pairs', 'median_at_0',
                     'median_at_1', 'median_difference', 'rises', 'unchanged',
                     'falls', 'statistic', 'p_value', 'p_holm', 'significant']


def summarise_exp_c_tests(summary):
    """The paper's section 5.1 prediction, tested rather than asserted."""
    tests, p_values = [], []
    for label, at_0, at_1 in (('spread_go: w_go=1 vs w_go=0', 'spread_go_at_0', 'spread_go_at_1'),
                              ('spread_ru: w_go=1 vs w_go=0', 'spread_ru_at_0', 'spread_ru_at_1')):
        pairs = [(row[at_0], row[at_1]) for row in summary
                 if isinstance(row[at_0], float) and isinstance(row[at_1], float)]
        differences = [b - a for a, b in pairs]
        statistic, p, used = wilcoxon(differences)
        tests.append({
            'comparison': label, 'n_tasks': len(pairs), 'n_nonzero_pairs': used,
            'median_at_0': median([a for a, _ in pairs]),
            'median_at_1': median([b for _, b in pairs]),
            'median_difference': median(differences) if differences else '',
            'rises': sum(1 for d in differences if d > 0),
            'unchanged': sum(1 for d in differences if d == 0),
            'falls': sum(1 for d in differences if d < 0),
            'statistic': statistic, 'p_value': p,
        })
        p_values.append(p)
    for entry, value in zip(tests, holm_bonferroni(p_values)):
        entry['p_holm'] = value
        entry['significant'] = value < 0.05
    return tests


# ----------------------------------------------------------------------
# Headline
# ----------------------------------------------------------------------

def print_headline(exp_b_summary, exp_b_tests, exp_c_summary, exp_c_tests, exp_a_summary):
    def rule(title):
        print()
        print('=' * 92)
        print(title)
        print('=' * 92)

    if exp_a_summary:
        rule('EXPERIMENT A -- selection cost (median seconds over tasks; no test on wall-clock)')
        sizes = sorted({row['pool_size'] for row in exp_a_summary})
        k = median([row['k'] for row in exp_a_summary])
        print(f'  {"selector":<20}' + ''.join(f'{size:>12}' for size in sizes))
        for selector in harness.SELECTORS:
            line = f'  {selector:<20}'
            for size in sizes:
                match = [row for row in exp_a_summary
                         if row['selector'] == selector and row['pool_size'] == size
                         and row['k'] == int(k)]
                line += f'{match[0]["median_select_seconds"]:>12.5f}' if match else f'{"":>12}'
            print(line)
        extraction = [row for row in exp_a_summary if row['pool_size'] == max(sizes)]
        if extraction:
            print(f'  behaviour extraction at pool_size={max(sizes)}: '
                  f'{extraction[0]["median_extract_seconds"]:.3f}s '
                  f'(shared by all five conditions, measured apart from selection)')

    if exp_b_summary:
        for stratum in ('all', 'b_pool>=k'):
            block = [row for row in exp_b_summary if row['stratum'] == stratum]
            if not block:
                continue
            rule(f'EXPERIMENT B -- what each rule selects [{stratum}, '
                 f'{block[0]["n_tasks"]} tasks]')
            print(f'  {"selector":<20}{"n_sel":>7}{"b_ach":>7}{"redund":>8}{"floor":>7}'
                  f'{"excess":>8}{"hidden":>9}{"pooled":>9}')
            for row in block:
                print(f'  {row["selector"]:<20}{row["median_n_selected"]:>7.0f}'
                      f'{row["median_b_achieved"]:>7.0f}{row["median_redundancy"]:>8.1f}'
                      f'{row["median_redundancy_floor"]:>7.1f}'
                      f'{row["median_redundancy_excess"]:>8.1f}'
                      f'{row["median_hidden_pref_rate"]:>9.3f}'
                      f'{row["pooled_hidden_rate"]:>9.3f}')

        rule('EXPERIMENT B -- hidden preference, Wilcoxon paired per task vs the baseline')
        print(f'  {"stratum":<12}{"selector":<20}{"n":>4}{"win":>5}{"tie":>5}{"loss":>6}'
              f'{"median diff":>13}{"p":>12}{"p(Holm)":>10}{"":>4}')
        for row in exp_b_tests:
            if row['column'] != 'hidden_pref_rate':
                continue
            print(f'  {row["stratum"]:<12}{row["selector"]:<20}{row["n_tasks"]:>4}'
                  f'{row["wins"]:>5}{row["ties"]:>5}{row["losses"]:>6}'
                  f'{row["median_difference"]:>13.4f}{row["p_value"]:>12.5f}'
                  f'{row["p_holm"]:>10.5f}{"  *" if row["significant"] else "":>4}')

    if exp_c_summary:
        rule(f'EXPERIMENT C -- weights [{len(exp_c_summary)} tasks]')
        acting = [row for row in exp_c_summary if (row['ru_values_in_pool'] or 0) != '']
        varying = [row for row in exp_c_summary
                   if str(row['ru_values_in_pool']).isdigit() and int(row['ru_values_in_pool']) > 1]
        never = [row for row in exp_c_summary if row['first_change_w_go'] == '']
        print(f'  tasks whose ru dimension varies at all : {len(varying)}/{len(exp_c_summary)}')
        print(f'  selections that never change over the sweep: {len(never)}/{len(exp_c_summary)}'
              f'  (the two-phase ceiling: the pool cannot act on the preference)')
        firsts = [row['first_change_w_go'] for row in exp_c_summary
                  if isinstance(row['first_change_w_go'], float)]
        if firsts:
            print(f'  first change in the behaviour set, median w_go = {median(firsts):.2f}')
        counts = [row['n_distinct_behaviour_sets'] for row in exp_c_summary
                  if isinstance(row['n_distinct_behaviour_sets'], float)]
        if counts:
            print(f'  distinct behaviour sets over 21 weightings, median = {median(counts):.0f}'
                  f' (max {max(counts):.0f})')
        holds = sum(1 for row in exp_c_summary if row['paper_prediction_holds'])
        print(f'  tasks where spread_go rises AND spread_ru falls: '
              f'{holds}/{len(exp_c_summary)}')
        for row in exp_c_tests:
            print(f'  {row["comparison"]:<30} median {row["median_at_0"]:.4f} -> '
                  f'{row["median_at_1"]:.4f}  rises/unchanged/falls = '
                  f'{row["rises"]}/{row["unchanged"]}/{row["falls"]}  '
                  f'p = {row["p_value"]:.5f}{"  *" if row["significant"] else ""}')
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--exp-a', action='append', default=[])
    parser.add_argument('--exp-b-scores', action='append', default=[])
    parser.add_argument('--exp-b-overlap', action='append', default=[])
    parser.add_argument('--exp-c', action='append', default=[])
    parser.add_argument('--analysis-dir',
                        default=os.path.join(common.PAPER_EXPERIMENTS_DIR, 'out', 'analysis'))
    args = parser.parse_args(argv)

    exp_a = read_rows(args.exp_a)
    exp_b = read_rows(args.exp_b_scores)
    exp_b_overlap = read_rows(args.exp_b_overlap)
    exp_c = read_rows(args.exp_c)

    a_summary = summarise_exp_a(exp_a) if exp_a else []
    b_summary = summarise_exp_b(exp_b) if exp_b else []
    b_tests = summarise_exp_b_tests(exp_b) if exp_b else []
    overlap = summarise_exp_b_overlap(exp_b_overlap) if exp_b_overlap else []
    c_summary = summarise_exp_c(exp_c) if exp_c else []
    c_tests = summarise_exp_c_tests(c_summary) if c_summary else []

    written = []
    for name, fields, rows in (
            ('exp_a_summary.csv', EXP_A_SUMMARY_FIELDS, a_summary),
            ('exp_b_summary.csv', EXP_B_SUMMARY_FIELDS, b_summary),
            ('exp_b_tests.csv',
             ['stratum', 'column', 'selector', 'baseline', 'n_tasks', 'n_nonzero_pairs',
              'median_ours', 'median_baseline', 'median_difference', 'wins', 'ties',
              'losses', 'statistic', 'p_value', 'p_holm', 'significant'], b_tests),
            ('exp_b_overlap_summary.csv', EXP_B_OVERLAP_FIELDS, overlap),
            ('exp_c_summary.csv', EXP_C_SUMMARY_FIELDS, c_summary),
            ('exp_c_tests.csv', EXP_C_TEST_FIELDS, c_tests)):
        if rows:
            written.append(harness.write_csv(os.path.join(args.analysis_dir, name),
                                             fields, rows))

    print_headline(b_summary, b_tests, c_summary, c_tests, a_summary)
    for path in written:
        print(f'wrote {path}')
    if not written:
        print('nothing to aggregate; pass --exp-a / --exp-b-scores / --exp-c')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
