"""E3 -- what the second phase costs, from the timing tasks' samples.

Every sample is a row; the medians are per (N, k, model, feature count,
phase, indicator). The figures plot the samples against the pool size and
against b on log axes with the two terms of the cost of App. cost drawn as
reference slopes through the median sample. No curve is fitted, and no
selection is ranked against another: none of the selection functions is ours.
"""

import statistics

from bdc_experiments import runner, timing
from bdc_experiments import report as rp

COLUMNS = list(runner.BASE_FIELDS) + ['features', 'phase', 'indicator', 'repeat', 'wall_s', 'cpu_s',
                                      'generation_wall_s', 'generation_cpu_s',
                                      'wall_over_generation', 'cpu_over_generation', 'exhausted']

MARKERS = ('o', 's', '^', 'D', 'v')


def _median(members, key):
    return rp.summarise([m.get(key) for m in members])['median']


def _medians(rows):
    keys = ('N', 'k', 'model', 'features', 'phase', 'indicator')
    out = []
    for values, members in rp.group(rows, keys).items():
        wall, cpu = rp.summarise([m['wall_s'] for m in members]), rp.summarise([m['cpu_s'] for m in members])
        out.append({**dict(zip(keys, values)), 'samples': len(members),
                    'wall_median': wall['median'], 'wall_q1': wall['q1'], 'wall_q3': wall['q3'],
                    'cpu_median': cpu['median'], 'cpu_q1': cpu['q1'], 'cpu_q3': cpu['q3'],
                    'cpu_pooled_mean': rp.pooled(members, 'cpu_s'),
                    'cpu_macro_mean': rp.macro(members, 'cpu_s'),
                    'pool_size_median': _median(members, 'pool_size'), 'b_median': _median(members, 'b'),
                    'generation_wall_median': _median(members, 'generation_wall_s'),
                    'wall_over_generation_median': _median(members, 'wall_over_generation'),
                    'wall_over_generation_macro': rp.macro(members, 'wall_over_generation')})
    return sorted(out, key=lambda r: (r['N'], r['phase'], r['features'], r['model'], r['k'] or 0,
                                      r['indicator'] or ''))


def _slope(ax, points, exponent, label):
    """A reference slope through the median sample: positioned, never fitted."""
    if len({x for x, _ in points}) < 2:
        return False
    x0, y0 = statistics.median([x for x, _ in points]), statistics.median([y for _, y in points])
    grid = sorted({x for x, _ in points})
    ax.plot(grid, [y0 * (x / x0) ** exponent for x in grid], color='black', linewidth=0.8,
            linestyle='--', label=label)
    return True


def _scatter(ax, points, offset, label, marker=None, area=14):
    if points:
        ax.scatter([x for x, _ in points], [y for _, y in points], s=area, alpha=0.7,
                   marker=marker or MARKERS[offset % len(MARKERS)], facecolors='none',
                   linewidths=0.8, edgecolors=rp.PALETTE[offset % len(rp.PALETTE)], label=label)
    return points


def _mapping_figure(rows, path):
    """Mapping and planning against the pool size, both on the wall clock."""
    fig, ax = rp.figure()
    mapping = [r for r in rows if r['phase'] == 'mapping' and r['wall_s'] > 0]
    points = []
    for offset, n in enumerate(sorted({r['features'] for r in mapping})):
        points += _scatter(ax, [(r['pool_size'], r['wall_s']) for r in mapping if r['features'] == n],
                           offset, f'mapping, n = {n}')
    _scatter(ax, sorted({(r['pool_size'], r['generation_wall_s']) for r in mapping if r['generation_wall_s']}),
             len(rp.PALETTE) - 1, 'pool generation', marker='P')
    drawn = _slope(ax, points, 1, 'reference slope 1 in $|P|$ (no fit)')
    if points:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize='small')
    ax.set_xlabel('pool size $|P|$')
    ax.set_ylabel('wall-clock time (s)')
    return rp.save(fig, path), drawn


def _selection_figure(rows, path):
    """Selection CPU time against b, one colour per indicator, marker area per
    pool size; the slope is positioned within the largest pool size alone."""
    fig, ax = rp.figure()
    selection = [r for r in rows if r['phase'] == 'selection' and r['b'] and r['cpu_s'] > 0]
    sizes = sorted({r['pool_size'] for r in selection})
    for offset, indicator in enumerate(runner.INDICATORS):
        for rank, size in enumerate(sizes):
            _scatter(ax, [(r['b'], r['cpu_s']) for r in selection
                          if r['indicator'] == indicator and r['pool_size'] == size],
                     offset, indicator if rank == 0 else None, area=10 + 12 * rank)
    biggest = sizes[-1] if sizes else None
    drawn = _slope(ax, [(r['b'], r['cpu_s']) for r in selection if r['pool_size'] == biggest], 2,
                   f'reference slope 2 in $b$ at pool size {biggest} (no fit)')
    if selection:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize='small')
    ax.set_xlabel('distinct behaviours $b$')
    ax.set_ylabel('selection CPU time (s)')
    return rp.save(fig, path), drawn


def report(cfg):
    results = rp.usable(runner.load_results(cfg, 'time'))
    rows = [row for result in results for row in result['rows']]
    medians = _medians(rows)
    out = rp.report_dir(cfg, 'e3')
    mapping_path, mapping_slope = _mapping_figure(rows, out / 'figures' / 'e3_mapping_vs_generation.pdf')
    selection_path, selection_slope = _selection_figure(rows, out / 'figures' / 'e3_selection_vs_b.pdf')
    written = [
        rp.write_csv(out / 'e3_timing.csv', rows, COLUMNS),
        rp.write_csv(out / 'e3_medians.csv', medians,
                     ['N', 'k', 'model', 'features', 'phase', 'indicator', 'samples', 'wall_median',
                      'wall_q1', 'wall_q3', 'cpu_median', 'cpu_q1', 'cpu_q3', 'cpu_pooled_mean',
                      'cpu_macro_mean', 'pool_size_median', 'b_median', 'generation_wall_median',
                      'wall_over_generation_median', 'wall_over_generation_macro']),
        rp.table(out / 'tables' / 'e3_medians.tex', 'tab:e3-medians',
                 'Median cost of the second phase per requested pool size $N$, selection size $k$ '
                 'and model: mapping a whole pool into the behaviour space, and one selection. '
                 '$n$ is the number of features of the model timed. The last two columns are the '
                 'median over samples, and the mean of the per-domain means, of one sample\'s '
                 'wall-clock time divided by the wall clock the planner spent producing that same '
                 'pool. ' + timing.PROTOCOL.replace('_', '\\_') + ' '
                 + timing.GENERATION_CLOCK.replace('_', '\\_') + ' ' + rp.TIE_RULE,
                 ['N', 'k', 'model', 'n features', 'phase', 'indicator', 'samples',
                  'median wall (s)', 'median CPU (s)', 'pooled CPU (s)', 'macro CPU (s)',
                  'median wall / generation', 'macro wall / generation'],
                 [[r['N'], r['k'], r['model'], r['features'], r['phase'], r['indicator'],
                   r['samples'], r['wall_median'], r['cpu_median'], r['cpu_pooled_mean'],
                   r['cpu_macro_mean'], r['wall_over_generation_median'],
                   r['wall_over_generation_macro']] for r in medians], digits=6),
        mapping_path, selection_path,
    ]
    mapping = [r for r in rows if r['phase'] == 'mapping']
    written.append(rp.manifest(cfg, 'e3', written, runner.load_results(cfg, 'time'), extra={
        'protocol': timing.PROTOCOL, 'generation_clock': timing.GENERATION_CLOCK,
        'no_fit': 'the lines in the figures are reference slopes through the median sample',
        'repeats': cfg['e3']['repeats'], 'largest_pools': cfg['e3']['largest_pools'],
        'samples': {'mapping': len(mapping), 'selection': len(rows) - len(mapping)},
        'reference_slope_pool_size': 'drawn' if mapping_slope else 'not drawn: a single pool size',
        'reference_slope_b': 'drawn, within the largest pool size' if selection_slope
                             else 'not drawn: the largest pool size covers a single b',
        'mapping_wall_over_generation_median': _median(mapping, 'wall_over_generation'),
        'mapping_wall_over_generation_macro': rp.macro(mapping, 'wall_over_generation')}))
    return written
