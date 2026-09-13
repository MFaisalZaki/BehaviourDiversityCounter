"""E3 -- the greedy selections against the exhaustive optimum.

Claim C3 has two halves. The exact half -- greedy selection is exact for
B-Coverage -- is checked here on every case, and a violation is blocking. The
other half -- B-MaxSum, B-MaxMin and B-Novelty are selected greedily and the
paper states no approximation bound -- is measured: the ratio of the greedy
value to the enumerated optimum over the subsets of exactly k behaviours. The
ratios describe the pools that were small enough to enumerate. They are
observations, not a bound, and neither prove nor assume one.
"""

import json
import math
import statistics
import time

from bdc_experiments import models, reference, runner
from bdc_experiments import report as rp

BASE = ['instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b']

#: A ratio counts as "at the optimum" within this of 1. The greedy and the
#: enumeration add the same rationals in a different order, so bit equality is
#: too strict a reading of "the greedy found an optimal set".
TOLERANCE = 1e-12

COLUMNS = BASE + [
    'pool_stem', 'indicator', 'greedy', 'optimal', 'ratio', 'optimum_exists',
    'greedy_distinct', 'greedy_subset', 'optimal_subset', 'optimal_at_most',
    'at_most_size', 'at_most_subset', 'at_most_exceeds', 'ratio_at_most',
    'enum_wall_s', 'enum_cached', 'enum_at_most_wall_s', 'select_wall_s', 'select_cpu_s']


def _ratio(value, optimal):
    """greedy / optimal -- missing, never zero, when the optimum is 0 or absent."""
    if optimal is None or optimal == 0:
        return None
    return value / optimal


def _optimum(cache, behaviours, d, k, indicator, kappa, at_most=False):
    """One enumerated optimum, with its wall time, computed once per case.

    B-Coverage, B-MaxSum and B-MaxMin do not read kappa, so their enumeration is
    shared across the kappa loop; the row says whether its time came from the
    cache rather than from a fresh enumeration.
    """
    key = (indicator, k, at_most, kappa if indicator == 'bnovelty' else None)
    if key in cache:
        return {**cache[key], 'cached': True}
    enumerator = reference.ref_optimum_at_most if at_most else reference.ref_optimum
    clock = time.perf_counter()
    value, subset = enumerator(behaviours, d, k, indicator, kappa)
    cache[key] = {'value': value, 'subset': None if subset is None else list(subset),
                  'wall_s': time.perf_counter() - clock, 'cached': False}
    return cache[key]


def _subset(entry):
    return None if entry['subset'] is None else ' '.join(str(i) for i in entry['subset'])


def run_task(task_id, cfg):
    """Every (k, kappa, indicator): the greedy set, the enumerated optimum, and
    for B-MaxMin the at-most-k optimum as well."""
    ctx = runner.context(cfg, task_id)
    spec = models.registry(cfg)[ctx['extra'][0]]
    task, counter, loaded, model_record, dump = runner.setup(cfg, ctx, spec)
    plans, b = loaded['plans'], len(dump['distinct'])
    low, high = cfg['e3']['behaviour_range']
    if not low <= b <= high:
        return {'pool': loaded['record'], 'model': model_record, 'rows': [],
                'extra': {'skipped': f'b = {b} lies outside the enumerated behaviour range '
                                     f'[{low}, {high}], so no optimum is enumerated here'}}

    # A behaviour is its position in dump['distinct']; d reads the dump's matrix.
    behaviours = list(range(b))
    d = lambda i, j: dump['matrix'][i][j]
    cache, rows, cases = {}, [], []
    for k in cfg['e3']['k_values']:
        for kappa in cfg['selection']['kappa_values']:
            base = runner.base_row(loaded, dump, model_record, k=k, kappa=kappa)
            for indicator in runner.INDICATORS:
                selected, wall, cpu = runner.select(counter, plans, k, indicator, kappa)
                selection = runner.selection_record(loaded, dump, selected, wall, cpu)
                greedy = runner.indicators(counter, selected, kappa)[indicator]
                held = sorted(set(selection['distinct']))
                best = _optimum(cache, behaviours, d, k, indicator, kappa)
                row = {**base, 'pool_stem': loaded['record']['pool_stem'],
                       'indicator': indicator, 'greedy': greedy, 'optimal': best['value'],
                       'ratio': _ratio(greedy, best['value']),
                       'optimum_exists': best['value'] is not None,
                       'greedy_distinct': len(held),
                       'greedy_subset': ' '.join(str(i) for i in held),
                       'optimal_subset': _subset(best),
                       'enum_wall_s': best['wall_s'], 'enum_cached': best['cached'],
                       'select_wall_s': wall, 'select_cpu_s': cpu}
                loose = None
                if indicator == 'bmaxmin':
                    loose = _optimum(cache, behaviours, d, k, indicator, kappa, at_most=True)
                    row.update({
                        'optimal_at_most': loose['value'], 'at_most_subset': _subset(loose),
                        'at_most_size': None if loose['subset'] is None else len(loose['subset']),
                        'at_most_exceeds': (None if None in (loose['value'], best['value'])
                                            else loose['value'] > best['value'] + TOLERANCE),
                        'ratio_at_most': _ratio(greedy, loose['value']),
                        'enum_at_most_wall_s': loose['wall_s']})
                rows.append(row)
                cases.append({'k': k, 'kappa': kappa, 'indicator': indicator,
                              'greedy_value': greedy, 'greedy_behaviours': held,
                              'selection': selection, 'optimum': best, 'optimum_at_most': loose})

    return {'pool': loaded['record'], 'model': model_record, 'rows': rows,
            'extra': {'cases': cases, 'behaviour_range': [low, high],
                      'behaviour_indexing': "a behaviour is its position in the dump's "
                                            "'distinct' list; d(i, j) is dump['matrix'][i][j]",
                      'k_values': cfg['e3']['k_values'],
                      'kappa_values': cfg['selection']['kappa_values']}}


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _percentile(values, p):
    """The p-th percentile by linear interpolation; missing for an empty sample."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _summary(rows):
    """The ratio's distribution per (indicator, k, kappa)."""
    out = []
    for indicator in runner.INDICATORS:
        for k, kappa in sorted({(row['k'], row['kappa']) for row in rows}):
            group = [row for row in rows
                     if (row['indicator'], row['k'], row['kappa']) == (indicator, k, kappa)]
            values = [row['ratio'] for row in group if row['ratio'] is not None]
            out.append({'indicator': indicator, 'k': k, 'kappa': kappa, 'cases': len(group),
                        'rated': len(values), 'min': min(values) if values else None,
                        'p5': _percentile(values, 5),
                        'median': statistics.median(values) if values else None,
                        'at_optimum': (sum(v >= 1 - TOLERANCE for v in values) / len(values)
                                       if values else None),
                        'pooled_mean': rp.pooled(group, 'ratio'),
                        'macro_mean': rp.macro(group, 'ratio')})
    return out


def _worst(rows):
    """The worst-rated case per (indicator, k, kappa), with its pool named and
    the behaviours the greedy missed."""
    out = []
    for indicator in runner.INDICATORS:
        for k, kappa in sorted({(row['k'], row['kappa']) for row in rows}):
            group = [row for row in rows
                     if (row['indicator'], row['k'], row['kappa']) == (indicator, k, kappa)
                     and row['ratio'] is not None]
            if not group:
                continue
            row = min(group, key=lambda r: r['ratio'])
            missed = (set((row['optimal_subset'] or '').split())
                      - set((row['greedy_subset'] or '').split()))
            out.append({**row, 'missed': ' '.join(sorted(missed, key=int))})
    return out


def _checks(rows):
    """The blocking half of C3: every B-Coverage ratio is exactly 1."""
    cases = [row for row in rows if row['indicator'] == 'bcoverage']
    named = lambda row: (f"{row['instance']} {row['pool_stem']} {row['model']} "
                         f"k={row['k']} kappa={row['kappa']}")
    return {
        'check': 'the greedy B-Coverage value equals the enumerated B-Coverage optimum '
                 'over the subsets of exactly k behaviours, on every case',
        'passed': all(row['ratio'] == 1.0 for row in cases if row['optimum_exists']),
        'cases': len(cases),
        'violations': [{'case': named(row), 'greedy': row['greedy'], 'optimal': row['optimal'],
                        'ratio': row['ratio']}
                       for row in cases if row['optimum_exists'] and row['ratio'] != 1.0],
        'not_enumerated': [{'case': named(row), 'reason': 'k exceeds b, so no subset of exactly '
                            'k behaviours exists and there is no ratio to check'}
                           for row in cases if not row['optimum_exists']],
        'tolerance': 'none: this ratio is a quotient of two integer-valued counts',
    }


def _at_most(rows):
    """B-MaxMin only: how often the at-most-k optimum beats the exactly-k one,
    and where the greedy set of k stands against it."""
    group = [row for row in rows
             if row['indicator'] == 'bmaxmin' and row.get('optimal_at_most') is not None]
    exceeds = [row for row in group if row.get('at_most_exceeds')]
    return {'cases': len(group), 'exceeds_exactly_k': len(exceeds),
            'exceeds_fraction': len(exceeds) / len(group) if group else None,
            'greedy_over_at_most': rp.summarise([row.get('ratio_at_most') for row in group]),
            'note': 'B-MaxMin is a minimum over pairs, so it can only fall as a set grows; the '
                    'at-most-k optimum is therefore an upper bound on the exactly-k optimum and '
                    'a fixed-size selection is measured against it here for reference only'}


def _figure(rows, path):
    """The ratio per indicator, one box per k, kappas pooled."""
    fig, ax = rp.figure(size=(6.0, 3.4))
    ks = sorted({row['k'] for row in rows})
    width = 0.8 / max(len(ks), 1)
    for offset, k in enumerate(ks):
        colour = rp.PALETTE[offset % len(rp.PALETTE)]
        samples = {index: [row['ratio'] for row in rows if row['indicator'] == indicator
                           and row['k'] == k and row['ratio'] is not None]
                   for index, indicator in enumerate(runner.INDICATORS)}
        samples = {index: values for index, values in samples.items() if values}
        if samples:
            drawn = ax.boxplot(list(samples.values()), patch_artist=True, manage_ticks=False,
                               widths=width * 0.8,
                               positions=[index + (offset - (len(ks) - 1) / 2) * width
                                          for index in samples])
            for box in drawn['boxes']:
                box.set_facecolor(colour)
            for median in drawn['medians']:
                median.set_color('black')
        ax.plot([], [], color=colour, linewidth=6, label=f'k = {k}')
    ax.axhline(1.0, color='grey', linewidth=0.8)
    ax.set_xticks(range(len(runner.INDICATORS)))
    ax.set_xticklabels(list(runner.INDICATORS))
    ax.set_ylabel('greedy / optimum')
    ax.legend(loc='lower right', fontsize='small')
    return rp.save(fig, path)


def report(cfg, results):
    """The ratio CSV, the worst cases, the B-Coverage check, the table, the
    figure and the manifest."""
    out = rp.report_dir(cfg, 'e3')
    rows = rp.all_rows(results)
    summary, checks = _summary(rows), _checks(rows)
    if not checks['passed']:
        print(f"E3 CHECK FAILED: {len(checks['violations'])} of {checks['cases']} B-Coverage "
              'cases do not reach the enumerated optimum; see e3_checks.json')

    written = [
        rp.write_csv(out / 'e3_ratios.csv', rows, COLUMNS),
        rp.write_csv(out / 'e3_worst_cases.csv', _worst(rows), COLUMNS + ['missed']),
        rp.write_csv(out / 'e3_summary.csv', summary,
                     ['indicator', 'k', 'kappa', 'cases', 'rated', 'min', 'p5', 'median',
                      'at_optimum', 'pooled_mean', 'macro_mean']),
        rp.table(out / 'tables' / 'e3_ratios.tex', 'tab:e3-ratios',
                 'The value each greedy selection reaches as a fraction of the exhaustive optimum '
                 'over the subsets of exactly $k$ behaviours, on the pools small enough to '
                 'enumerate. B-Coverage is exact by Thm.~bcov-greedy and is shown as the check it '
                 'is; the other three rules are heuristics for which the paper states no '
                 'approximation bound, so the values here are measurements on this benchmark '
                 'and not a bound. "at opt." is the fraction of cases reaching the optimum. '
                 + rp.TIE_RULE,
                 ['indicator', 'k', 'kappa', 'cases', 'min', 'p5', 'median', 'at opt.',
                  'macro mean'],
                 [[row['indicator'], row['k'], row['kappa'], row['cases'], row['min'], row['p5'],
                   row['median'], row['at_optimum'], row['macro_mean']] for row in summary]),
        _figure(rows, out / 'figures' / 'e3_ratios.pdf'),
    ]

    path = out / 'e3_checks.json'
    path.write_text(json.dumps(checks, indent=1))
    written.append(path)
    written.append(rp.manifest(cfg, 'e3', written, results, extra={
        'bcoverage_check': (('PASS' if checks['passed'] else 'FAIL') if checks['cases']
                            else 'no B-Coverage case was enumerated'),
        'bcoverage_violations': checks['violations'],
        'bcoverage_cases': checks['cases'],
        'behaviour_range': cfg['e3']['behaviour_range'],
        'k_values': cfg['e3']['k_values'],
        'bmaxmin_at_most': _at_most(rows),
        'at_optimum_tolerance': TOLERANCE,
        'note': 'the ratios report how the three heuristic selections behaved on the enumerated '
                'pools; the paper states no approximation bound and nothing here establishes one. '
                'A ratio short of 1 by a few units in the last place on a greedy set equal to the '
                'optimal one is floating point, not a loss: the library sums the pairwise '
                'distances in pool order and the reference uses math.fsum, so the tolerance above '
                'decides which cases count as reaching the optimum.',
    }))
    return written
