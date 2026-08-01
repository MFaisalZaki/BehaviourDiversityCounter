"""Paper figures from the aggregated CSVs (run aggregate.py first).

One figure per experiment, sized for an ECAI two-column page (3.3in column,
6.9in full width), written as PDF + PNG into <analysis>/figures/.

Colour discipline: categorical hues only where series identity is the point
(the runtime series, the requested-k groups), in a fixed order; a single hue
for magnitude comparisons (bars, boxes, histograms); no dual axes; recessive
grid and axis chrome. Palette slots and chrome follow the validated reference
palette (light mode -- these figures render on paper).
"""

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common  # noqa: F401

# Validated reference palette, light mode: categorical slots in fixed order.
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'

COLUMN_IN = 3.3
FULL_IN = 6.9


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'sans-serif',
        'text.color': INK,
        'axes.edgecolor': BASELINE,
        'axes.labelcolor': INK_SECONDARY,
        'axes.titlesize': 8,
        'axes.titlecolor': INK,
        'axes.linewidth': 0.6,
        'axes.grid': True,
        'axes.axisbelow': True,
        'grid.color': GRID,
        'grid.linewidth': 0.5,
        'xtick.color': MUTED,
        'ytick.color': MUTED,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.frameon': False,
        'legend.fontsize': 7,
        'figure.dpi': 200,
        'savefig.bbox': 'tight',
    })
    return plt


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as handle:
        return list(csv.DictReader(handle))


def as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def save(figure, out_dir, name):
    for extension in ('pdf', 'png'):
        figure.savefig(os.path.join(out_dir, f'{name}.{extension}'))
    print(f'wrote {os.path.join(out_dir, name)}.pdf/.png')


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def figure_exp1(plt, rows, out_dir):
    """ECDF of b/n per requested pool size: how much do pools collapse?"""
    groups = defaultdict(list)
    for row in rows:
        ratio = as_float(row['b_over_n'])
        if ratio is not None and row['k_requested']:
            groups[int(row['k_requested'])].append(ratio)
    if not groups:
        return
    figure, axis = plt.subplots(figsize=(COLUMN_IN, 2.2))
    for slot, k in enumerate(sorted(groups)[:len(SERIES)]):
        values = sorted(groups[k])
        steps = [i / len(values) for i in range(1, len(values) + 1)]
        axis.step(values, steps, where='post', linewidth=1.2,
                  color=SERIES[slot], label=f'k = {k} ({len(values)} pools)')
    axis.set_xlabel('distinct behaviours / pool size  (b/n)')
    axis.set_ylabel('fraction of pools')
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc='lower right')
    save(figure, out_dir, 'exp1_collapse_ecdf')
    plt.close(figure)


def figure_exp2(plt, rows, out_dir):
    """Within-group B-MaxSum spread against the group's BDC value, per m."""
    subset_sizes = sorted({int(row['m']) for row in rows if row['m']})
    if not subset_sizes:
        return
    figure, axes = plt.subplots(1, len(subset_sizes),
                                figsize=(FULL_IN, 2.0), sharey=True)
    if len(subset_sizes) == 1:
        axes = [axes]
    for axis, m in zip(axes, subset_sizes):
        grouped = defaultdict(list)
        for row in rows:
            if row['m'] and int(row['m']) == m:
                ratio = as_float(row['spread_ratio'])
                if ratio is not None and row['v']:
                    grouped[int(row['v'])].append(ratio)
        values = sorted(grouped)
        if not values:
            axis.set_visible(False)
            continue
        boxes = axis.boxplot([grouped[v] for v in values], positions=values,
                             widths=0.6, showfliers=False, patch_artist=True)
        for patch in boxes['boxes']:
            patch.set(facecolor=SERIES[0], alpha=0.35, edgecolor=SERIES[0], linewidth=0.8)
        for part in ('whiskers', 'caps'):
            for line in boxes[part]:
                line.set(color=SERIES[0], linewidth=0.8)
        for line in boxes['medians']:
            line.set(color=INK, linewidth=1.0)
        axis.axhline(1.0, color=BASELINE, linewidth=0.6, zorder=0)
        axis.set_yscale('log')
        axis.set_title(f'm = {m}')
        axis.set_xlabel('BDC value of the group (v)')
        if len(values) > 8:  # keep the tick labels from colliding
            shown = [v for v in values if v % max(1, round(len(values) / 8)) == 0]
            axis.set_xticks(shown)
            axis.set_xticklabels([str(v) for v in shown])
    axes[0].set_ylabel('B-MaxSum max/min in group')
    save(figure, out_dir, 'exp2_spread_by_bdc')
    plt.close(figure)


def figure_exp3(plt, rows, out_dir):
    """Cross-evaluation: mean score of each selector under both families."""
    ks = sorted({int(row['k']) for row in rows if row['k']})
    if not ks:
        return
    k = ks[0] if len(ks) == 1 else ks[-1]  # headline panel: the largest k
    scorers = [('bdc', 'BDC'), ('bmaxsum', 'B-MaxSum'),
               ('maxsum_stability', 'MaxSum-Stability')]
    selectors, means = [], defaultdict(dict)
    for row in rows:
        if int(row['k']) != k:
            continue
        selector = row['selector']
        if selector not in selectors:
            selectors.append(selector)
        for field, _ in scorers:
            value = as_float(row.get(field))
            if value is not None:
                means[field].setdefault(selector, []).append(value)

    figure, axes = plt.subplots(1, len(scorers), figsize=(FULL_IN, 1.9))
    positions = range(len(selectors))
    for axis, (field, label) in zip(axes, scorers):
        values = [statistics.fmean(means[field].get(s, [0.0])) for s in selectors]
        axis.barh(list(positions), values, height=0.62, color=SERIES[0])
        axis.set_yticks(list(positions))
        axis.set_yticklabels(selectors if axis is axes[0] else [''] * len(selectors))
        axis.invert_yaxis()
        axis.set_title(f'mean {label} (k = {k})')
        axis.grid(axis='y', visible=False)
    save(figure, out_dir, 'exp3_cross_evaluation')
    plt.close(figure)


def figure_exp4(plt, rows, out_dir):
    """Histogram of the greedy/optimal B-MaxSum ratio."""
    ratios = [as_float(row['ratio']) for row in rows]
    ratios = [r for r in ratios if r is not None]
    if not ratios:
        return
    figure, axis = plt.subplots(figsize=(COLUMN_IN, 1.9))
    low = min(0.9, min(ratios))
    axis.hist(ratios, bins=[low + i * (1.001 - low) / 40 for i in range(41)],
              color=SERIES[0], edgecolor='white', linewidth=0.3)
    axis.axvline(1.0, color=INK, linewidth=0.8, linestyle='--')
    axis.set_xlabel('greedy / optimal B-MaxSum')
    axis.set_ylabel('cases')
    optimal = sum(1 for r in ratios if r >= 0.9999)
    axis.set_title(f'{optimal}/{len(ratios)} cases greedy-optimal '
                   f'(median {statistics.median(ratios):.4f})')
    save(figure, out_dir, 'exp4_greedy_gap')
    plt.close(figure)


def figure_exp5(plt, rows, out_dir):
    """Runtime against pool size (log-log): BDC linear, the sums quadratic."""
    series = [('t_bdc_warm', 'BDC'),
              ('t_bmaxsum_cold', 'B-MaxSum'),
              ('t_maxsum_stability_cold', 'plan-level MaxSum (Stability)')]
    figure, (left, right) = plt.subplots(1, 2, figsize=(FULL_IN, 2.3))

    for slot, (field, label) in enumerate(series):
        points = [(as_float(row['n_plans']), as_float(row[field]))
                  for row in rows if as_float(row.get(field))]
        points = [(n, t) for n, t in points if n and t and t > 0]
        if not points:
            continue
        left.scatter([n for n, _ in points], [t for _, t in points],
                     s=4, alpha=0.35, linewidths=0, color=SERIES[slot], label=label)
    left.set_xscale('log')
    left.set_yscale('log')
    left.set_xlabel('pool size n')
    left.set_ylabel('seconds')
    left.legend(loc='upper left', markerscale=2.5)

    points = [(as_float(row['b']), as_float(row['t_bmaxsum_cold']))
              for row in rows if as_float(row.get('t_bmaxsum_cold'))]
    points = [(b, t) for b, t in points if b and t and t > 0]
    if points:
        right.scatter([b for b, _ in points], [t for _, t in points],
                      s=4, alpha=0.35, linewidths=0, color=SERIES[1])
        # A b^2 guide anchored at the median point, so the slope is readable.
        med_b = statistics.median([b for b, _ in points])
        med_t = statistics.median([t for _, t in points])
        guide_b = [max(1, min(b for b, _ in points)), max(b for b, _ in points)]
        right.plot(guide_b, [med_t * (g / med_b) ** 2 for g in guide_b],
                   color=MUTED, linewidth=0.8, linestyle='--')
        right.annotate('slope 2', xy=(guide_b[1], med_t * (guide_b[1] / med_b) ** 2),
                       fontsize=7, color=INK_SECONDARY,
                       textcoords='offset points', xytext=(-18, 2))
    right.set_xscale('log')
    right.set_yscale('log')
    right.set_xlabel('distinct behaviours b')
    right.set_ylabel('B-MaxSum seconds')

    save(figure, out_dir, 'exp5_runtime')
    plt.close(figure)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default_analysis = os.path.join(common.PAPER_EXPERIMENTS_DIR, 'sandbox', 'analysis')
    parser.add_argument('--analysis-dir', default=default_analysis,
                        help='Directory holding the aggregate.py CSVs.')
    args = parser.parse_args(argv)

    plt = style()
    out_dir = os.path.join(args.analysis_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    tables = {
        name: read_csv(os.path.join(args.analysis_dir, f'{name}.csv'))
        for name in ('exp1_collapse', 'exp2_discrimination', 'exp3_cross',
                     'exp4_greedy_gap', 'exp5_runtime')
    }
    for name, rows in tables.items():
        if not rows:
            print(f'note: no rows for {name}, skipping its figure', file=sys.stderr)

    if tables['exp1_collapse']:
        figure_exp1(plt, tables['exp1_collapse'], out_dir)
    if tables['exp2_discrimination']:
        figure_exp2(plt, tables['exp2_discrimination'], out_dir)
    if tables['exp3_cross']:
        figure_exp3(plt, tables['exp3_cross'], out_dir)
    if tables['exp4_greedy_gap']:
        figure_exp4(plt, tables['exp4_greedy_gap'], out_dir)
    if tables['exp5_runtime']:
        figure_exp5(plt, tables['exp5_runtime'], out_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
