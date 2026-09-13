"""E4 -- the fixed-size reading of B-MaxMin and B-Novelty.

Claim C4 says the two are not monotone in the plans selected, so a value of
either is only meaningful at a set size fixed in advance. One greedy run per
(kappa, indicator) at the largest size the pool admits shows it: the greedy
procedures are prefix-consistent for k >= 2, so step k of that run is the
selection at k and the value of every prefix is read off the one run. Every
prefix value is computed twice, through the library counter and from the
behaviour dump's dissimilarity matrix, and the two are compared.
"""

import statistics

from bdc_experiments import models, reference, runner
from bdc_experiments import report as rp

BASE = ['instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b']

#: One prefix series: these fields fix the pool, the model and the kappa.
CELL = ('instance', 'q', 'N', 'model', 'kappa')

#: The two indicators that cannot fall as the selected set grows.
MONOTONE = ('bcoverage', 'bmaxsum')

#: A step smaller than this is the order the same rationals were added in, not
#: a fall; and the library and the matrix are read as agreeing within 1e-9.
TOLERANCE = 1e-12
AGREEMENT = 1e-9

COLUMNS = BASE + ['pool_stem', 'indicator', 'k_max', 'value', 'matrix_value', 'gap', 'agrees',
                  'previous', 'delta', 'fell', 'relative_fall']


def run_task(task_id, cfg):
    """Every prefix of one greedy run per (kappa, indicator), valued twice."""
    ctx = runner.context(cfg, task_id)
    spec = models.registry(cfg)[ctx['extra'][0]]
    task, counter, loaded, model_record, dump = runner.setup(cfg, ctx, spec)
    plans, b = loaded['plans'], len(dump['distinct'])
    k_min, k_cap = cfg['e4']['k_range']
    k_max = min(k_cap, b, len(plans))
    if k_max < k_min:
        return {'pool': loaded['record'], 'model': model_record, 'rows': [],
                'extra': {'skipped': f'k_max = min({k_cap}, b = {b}, pool size = {len(plans)}) '
                                     f'= {k_max} is below k_min = {k_min}, so this pool has no '
                                     'prefix to read'}}

    # A behaviour is its position in the dump's 'distinct' list.
    d = lambda i, j: dump['matrix'][i][j]
    rows, runs = [], []
    for kappa in cfg['selection']['kappa_values']:
        for indicator in runner.INDICATORS:
            selected, wall, cpu = runner.select(counter, plans, k_max, indicator, kappa)
            selection = runner.selection_record(loaded, dump, selected, wall, cpu)
            previous, values = None, []
            for k in range(k_min, k_max + 1):
                value = runner.indicators(counter, selected[:k], kappa)[indicator]
                matrix = reference.ref_indicator(indicator, selection['distinct'][:k], d, kappa)
                delta = None if previous is None else value - previous
                fell = None if delta is None else delta < -TOLERANCE
                rows.append({**runner.base_row(loaded, dump, model_record, k=k, kappa=kappa),
                             'pool_stem': loaded['record']['pool_stem'], 'indicator': indicator,
                             'k_max': k_max, 'value': value, 'matrix_value': matrix,
                             'gap': abs(value - matrix),
                             'agrees': abs(value - matrix) <= AGREEMENT,
                             'previous': previous, 'delta': delta, 'fell': fell,
                             'relative_fall': ((previous - value) / previous
                                               if fell and previous > 0 else None)})
                values.append(value)
                previous = value
            runs.append({'indicator': indicator, 'kappa': kappa, 'k_min': k_min, 'k_max': k_max,
                         'prefix_values': values, 'selection': selection})

    return {'pool': loaded['record'], 'model': model_record, 'rows': rows,
            'extra': {'runs': runs, 'k_min': k_min, 'k_max': k_max,
                      'k_range': cfg['e4']['k_range'],
                      'prefix_consistency': 'the greedy procedures extend the selection at k to '
                                            'the one at k + 1 for k >= 2, so each selection was '
                                            'run once at k_max and its prefixes are the '
                                            'selections at the smaller sizes',
                      'behaviour_indexing': "a behaviour is its position in the dump's 'distinct' "
                                            "list; d(i, j) is dump['matrix'][i][j]"}}


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _series(rows, indicator):
    """``cell -> [(k, value)]`` for one indicator, in increasing k."""
    return {cell: sorted((row['k'], row['value']) for row in members)
            for cell, members in rp.group([r for r in rows if r['indicator'] == indicator],
                                          CELL).items()}


def _first_falls(rows):
    """Per B-MaxMin series: the first k whose value is below the opening one."""
    out = []
    for cell, members in rp.group([r for r in rows if r['indicator'] == 'bmaxmin'], CELL).items():
        ordered = sorted(members, key=lambda row: row['k'])
        opening = ordered[0]
        fall = next((row for row in ordered[1:]
                     if row['value'] < opening['value'] - TOLERANCE), None)
        out.append({**dict(zip(CELL, cell)), 'domain': opening['domain'], 'b': opening['b'],
                    'opening_k': opening['k'], 'opening_value': opening['value'],
                    'first_fall_k': fall['k'] if fall else None,
                    'first_fall_value': fall['value'] if fall else None,
                    'first_fall_over_b': (fall['k'] / opening['b']
                                          if fall and opening['b'] else None)})
    return out


def _summary(rows):
    """Per indicator: how often a step falls, by how much, and the check."""
    falls = _first_falls(rows)
    out = []
    for indicator in runner.INDICATORS:
        group = [row for row in rows if row['indicator'] == indicator]
        steps = [row for row in group if row['fell'] is not None]
        fallen = [row for row in steps if row['fell']]
        entry = {'indicator': indicator, 'series': len(rp.group(group, CELL)),
                 'steps': len(steps), 'falls': len(fallen),
                 'fall_fraction': rp.pooled(steps, 'fell'),
                 'fall_fraction_macro': rp.macro(steps, 'fell'),
                 'median_relative_fall': (statistics.median([row['relative_fall']
                                                             for row in fallen])
                                          if fallen else None),
                 'monotone_check': (None if indicator not in MONOTONE or not steps
                                    else ('ok' if not fallen else f'VIOLATED ({len(fallen)})')),
                 'disagreements': sum(1 for row in group if not row['agrees'])}
        if indicator == 'bmaxmin':
            first = [row for row in falls if row['first_fall_k'] is not None]
            entry.update({
                'first_fall_series': len(first), 'never_falls_series': len(falls) - len(first),
                'first_fall_k_median': (statistics.median([row['first_fall_k'] for row in first])
                                        if first else None),
                'first_fall_over_b_median': (statistics.median([row['first_fall_over_b']
                                                                for row in first])
                                             if first else None)})
        out.append(entry)
    return out


def _figure(rows, path):
    """Value against k over the opening value: three pools each, and the mean."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6), sharex=True)
    for ax, indicator in zip(axes.flat, runner.INDICATORS):
        ratios = {cell: [(k, value / points[0][1]) for k, value in points]
                  for cell, points in _series(rows, indicator).items()
                  if points and points[0][1]}
        chosen = sorted(ratios, key=lambda cell: (-len(ratios[cell]), cell))[:3]
        for offset, cell in enumerate(chosen):
            ax.plot([k for k, _ in ratios[cell]], [v for _, v in ratios[cell]],
                    color=rp.PALETTE[offset], linewidth=0.9, marker='.', markersize=4,
                    label='one pool' if offset == 0 else None)
        mean = {}
        for points in ratios.values():
            for k, value in points:
                mean.setdefault(k, []).append(value)
        if mean:
            ax.plot(sorted(mean), [statistics.fmean(mean[k]) for k in sorted(mean)],
                    color='black', linewidth=2.0, label='mean of all')
        ax.axhline(1.0, color='grey', linewidth=0.8)
        ax.set_ylabel(f'{indicator} / opening', fontsize='small')
    if axes.flat[0].get_legend_handles_labels()[0]:
        axes.flat[0].legend(loc='upper left', fontsize='x-small')
    for ax in axes[1]:
        ax.set_xlabel('k')
    return rp.save(fig, path)


def report(cfg, results):
    """The prefix values, the fall summary, the monotonicity check and the figure."""
    out = rp.report_dir(cfg, 'e4')
    rows = rp.all_rows(results)
    summary = _summary(rows)
    violations = [row for row in rows
                  if row['indicator'] in MONOTONE and row['fell']]
    disagreements = [row for row in rows if not row['agrees']]
    if violations:
        print(f'E4 CHECK FAILED: {len(violations)} step(s) at which B-Coverage or B-MaxSum fell; '
              'see the monotone column of e4_summary.csv')
    if disagreements:
        print(f'E4: {len(disagreements)} prefix value(s) where the library and the behaviour '
              f'matrix differ by more than {AGREEMENT}; see the gap column of e4_prefix_values.csv')

    columns = ['indicator', 'series', 'steps', 'falls', 'fall_fraction', 'fall_fraction_macro',
               'median_relative_fall', 'first_fall_series', 'never_falls_series',
               'first_fall_k_median', 'first_fall_over_b_median', 'disagreements',
               'monotone_check']
    written = [
        rp.write_csv(out / 'e4_prefix_values.csv', rows, COLUMNS),
        rp.write_csv(out / 'e4_summary.csv', summary, columns),
        rp.table(out / 'tables' / 'e4_summary.tex', 'tab:e4-summary',
                 'How often the value of a greedy selection falls when the set size grows by one, '
                 'over the prefixes of one run per pool, model and $\\kappa$. B-Coverage and '
                 'B-MaxSum cannot fall and the monotone column is the check that they never did; '
                 'anything but "ok" there is a violation. The last two columns are B-MaxMin only: '
                 'the median size at which it first drops below its value at the opening pair, '
                 'absolutely and as a fraction of $b$. ' + rp.TIE_RULE,
                 ['indicator', 'series', 'steps', 'falls', 'fall frac.', 'macro',
                  'median rel. fall', 'first fall k', 'first fall k/b', 'monotone'],
                 [[row['indicator'], row['series'], row['steps'], row['falls'],
                   row['fall_fraction'], row['fall_fraction_macro'], row['median_relative_fall'],
                   row.get('first_fall_k_median'), row.get('first_fall_over_b_median'),
                   row['monotone_check']] for row in summary]),
        _figure(rows, out / 'figures' / 'e4_prefixes.pdf'),
    ]
    written.append(rp.manifest(cfg, 'e4', written, results, extra={
        'monotonicity_check': 'FAIL' if violations else 'PASS' if rows else 'no prefix values',
        'monotonicity_note': 'B-Coverage and B-MaxSum are non-decreasing as the selected set '
                             'grows; a fall of either is a violation, not a measurement',
        'violations': [f"{row['instance']} {row['pool_stem']} {row['model']} "
                       f"kappa={row['kappa']} {row['indicator']} k={row['k']} "
                       f"delta={row['delta']}" for row in violations],
        'library_matrix_disagreements': len(disagreements),
        'max_gap': max((row['gap'] for row in rows), default=None),
        'agreement_tolerance': AGREEMENT, 'fall_tolerance': TOLERANCE,
        'k_range': cfg['e4']['k_range'], 'kappa_values': cfg['selection']['kappa_values'],
        'bmaxmin_first_falls': _first_falls(rows),
    }))
    return written
