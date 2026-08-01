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
        # --- LaTeX inclusion ------------------------------------------------
        # Type 42 (TrueType) rather than matplotlib's default Type 3: IEEE,
        # ACM and arXiv all flag or reject Type 3 fonts in submitted PDFs.
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        # Crop to the drawn content, with only a hairline of padding, so
        # \includegraphics needs no trimming and no \vspace correction.
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
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
    """Write the figure as PDF (for LaTeX) and PNG (for quick viewing).

    bbox/pad are passed explicitly as well as set in rcParams so the crop
    holds even if a caller restyles.
    """
    for extension in ('pdf', 'png'):
        figure.savefig(os.path.join(out_dir, f'{name}.{extension}'),
                       bbox_inches='tight', pad_inches=0.01)
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


# Pool-size buckets for the runtime figure. Pool sizes cluster hard on the
# requested-k values (5, 10, 100, 1000), so these edges keep every bucket
# populated while spanning three orders of magnitude.
SIZE_BUCKETS = [(1, 1, '1'), (2, 4, '2-4'), (5, 9, '5-9'), (10, 49, '10-49'),
                (50, 99, '50-99'), (100, 499, '100-499'), (500, 10 ** 9, '500+')]

# Each indicator with the complexity its analysis predicts, as an exponent of
# the pool size n. B-MaxSum is O(b^2) in the number of *distinct behaviours*;
# since b <= n that is an upper bound of n^2 here, which is why its overlay is
# labelled as a bound rather than a prediction -- and why it also carries a fit
# against b, the variable its bound is actually stated in.
RUNTIME_SERIES = [
    {'field': 't_bdc_warm', 'label': 'BDC (warm)',
     'exponent': 1.0, 'theory': 'O(n)'},
    {'field': 't_bmaxsum_cold', 'label': 'B-MaxSum (cold)',
     'exponent': 2.0, 'theory': 'O(b$^2$) $\\leq$ O(n$^2$)', 'also_fit': ('b', 'b')},
    {'field': 't_maxsum_stability_cold', 'label': 'plan-level MaxSum-Stability (cold)',
     'exponent': 2.0, 'theory': 'O(n$^2$)'},
]


def theory_label_plain(label):
    """The mathtext complexity label as plain text, for stdout."""
    return (label.replace('$^2$', '^2').replace('$\\leq$', '<=')
                 .replace('$', ''))


def _log_fit_exponent(points):
    """Least-squares slope of log t against log n, and its R^2."""
    xs = [math.log(n) for n, _ in points]
    ys = [math.log(t) for _, t in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None, None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
    residual = sum((y - (mean_y + slope * (x - mean_x))) ** 2 for x, y in zip(xs, ys))
    total = sum((y - mean_y) ** 2 for y in ys)
    return slope, (1 - residual / total if total else None)


def figure_exp5(plt, rows, out_dir):
    """Runtime per indicator by pool size, against the predicted complexity.

    One box per (indicator, pool-size bucket) -- whiskers span the full
    min-max range, the notch-free box is the quartiles, the diamond is the
    mean -- with each indicator's theoretical curve overlaid. The curve has
    its exponent *fixed* at the predicted value and only its constant fitted
    (least squares in log space), so the question the figure answers is
    whether the measured slope matches the predicted one, not whether some
    curve can be made to pass through the data.
    """
    from matplotlib.lines import Line2D

    figure, axis = plt.subplots(figsize=(FULL_IN, 2.8))
    offsets = [-0.26, 0.0, 0.26]
    legend_handles = []

    for slot, series in enumerate(RUNTIME_SERIES):
        field, label = series['field'], series['label']
        exponent, theory_label = series['exponent'], series['theory']
        points = [(as_float(row['n_plans']), as_float(row[field])) for row in rows]
        points = [(n, t) for n, t in points if n and t and n > 0 and t > 0]
        if not points:
            continue

        grouped, positions = [], []
        for index, (low, high, _) in enumerate(SIZE_BUCKETS):
            values = [t for n, t in points if low <= n <= high]
            if values:
                grouped.append(values)
                positions.append(index + offsets[slot])
        if not grouped:
            continue

        boxes = axis.boxplot(grouped, positions=positions, widths=0.22,
                             whis=(0, 100), showmeans=True, patch_artist=True,
                             manage_ticks=False)
        for patch in boxes['boxes']:
            patch.set(facecolor=SERIES[slot], alpha=0.30,
                      edgecolor=SERIES[slot], linewidth=0.7)
        for part in ('whiskers', 'caps'):
            for line in boxes[part]:
                line.set(color=SERIES[slot], linewidth=0.7)
        for line in boxes['medians']:
            line.set(color=INK, linewidth=0.9)
        for marker in boxes['means']:
            marker.set(marker='D', markersize=2.4, markerfacecolor='white',
                       markeredgecolor=SERIES[slot], markeredgewidth=0.7)

        # Theory curve: exponent fixed, constant fitted in log space.
        constant = statistics.fmean(math.log(t) - exponent * math.log(n)
                                    for n, t in points)
        midpoints = [(index, statistics.geometric_mean([n for n, _ in points
                                                        if low <= n <= high]))
                     for index, (low, high, _) in enumerate(SIZE_BUCKETS)
                     if any(low <= n <= high for n, _ in points)]
        axis.plot([index for index, _ in midpoints],
                  [math.exp(constant) * mid ** exponent for _, mid in midpoints],
                  color=SERIES[slot], linewidth=1.0, linestyle='--', zorder=5)

        # The measured exponents stay out of the figure -- they go to stdout, so
        # the plot reads cleanly and the numbers behind the dashed curves are
        # still recorded every time it is regenerated.
        measured, r_squared = _log_fit_exponent(points)
        fits = [f'n^{measured:.2f} (R2 {r_squared:.2f})']
        # Where the stated bound is in another variable, fit that one too --
        # otherwise dropping its panel would drop the only test of the bound.
        if series.get('also_fit'):
            column, symbol = series['also_fit']
            other = [(as_float(row[column]), as_float(row[field])) for row in rows]
            other = [(x, t) for x, t in other if x and t and x > 0 and t > 0]
            if other:
                slope, other_r2 = _log_fit_exponent(other)
                fits.append(f'{symbol}^{slope:.2f} (R2 {other_r2:.2f})')
        print(f'  exp5 {label}: predicted {theory_label_plain(theory_label)}, '
              f'measured {"; ".join(fits)}')

        legend_handles.append(Line2D([], [], color=SERIES[slot], linewidth=6,
                                     alpha=0.45, label=label))

    axis.set_yscale('log')
    axis.set_xticks(range(len(SIZE_BUCKETS)))
    axis.set_xticklabels([label for _, _, label in SIZE_BUCKETS])
    axis.set_xlim(-0.6, len(SIZE_BUCKETS) - 0.4)
    axis.set_xlabel('pool size n')
    axis.set_ylabel('seconds')
    axis.grid(axis='x', visible=False)
    # Two decades of headroom so the legend never sits over a box or a curve.
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top * 10 ** 2.4)
    legend_handles.append(Line2D([], [], color=MUTED, linewidth=1.0, linestyle='--',
                                 label='predicted complexity, constant fitted'))
    axis.legend(handles=legend_handles, loc='upper left', ncol=1,
                handlelength=1.6, labelspacing=0.35, borderpad=0.2)
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
