"""Figures for the three experiments, from the CSVs the runners write.

matplotlib is the only thing here that the library itself does not need, so it
is imported lazily and its absence is reported as an instruction rather than a
traceback.

Each figure is written as both PDF and PNG -- the PDF for the paper, the PNG
for looking at.
"""

import argparse
import collections
import os
import sys

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import common, harness
from paperexps.aggregate import column, number, read_rows
from paperexps.stats import median

#: One colour and marker per condition, held fixed across every figure so the
#: same rule is the same line wherever it appears.
STYLE = {
    'bcoverage':        ('#1b6ca8', 'o', '-'),
    'bmaxsum':          ('#158467', 's', '-'),
    'bmaxmin':          ('#b8860b', '^', '--'),
    'bnovelty':         ('#8b3a62', 'D', '--'),
    'maxsum_stability': ('#a83232', 'v', ':'),
}


def pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            'plots.py needs matplotlib, which the library itself does not.\n'
            "  pip install matplotlib      (or: poetry install --with analysis)")
    plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3,
                         'figure.dpi': 130, 'savefig.bbox': 'tight'})
    return plt


def save(figure, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for suffix in ('pdf', 'png'):
        path = os.path.join(out_dir, f'{name}.{suffix}')
        figure.savefig(path)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------

def dominant_k(rows):
    """The k the pool-size sweep ran at: the one appearing at the most sizes.

    Not the median of the k column -- the k sweep adds 5, 10 and 50 at the
    largest size only, which can drag a median off the value every size shares.
    """
    spread = collections.defaultdict(set)
    for row in rows:
        spread[number(row, 'k')].add(number(row, 'pool_size'))
    if not spread:
        return 20
    return int(max(spread, key=lambda k: (len(spread[k]), -k)))


def figure_exp_a_scaling(rows, out_dir, plt):
    """Selection time against pool size, per rule, with extraction for scale."""
    k = dominant_k(rows)
    at_k = [row for row in rows if number(row, 'k') == k]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    sizes = sorted({int(number(row, 'pool_size')) for row in at_k})
    for selector in harness.SELECTORS:
        colour, marker, line = STYLE[selector]
        ys = []
        for size in sizes:
            values = column([row for row in at_k
                             if row['selector'] == selector
                             and number(row, 'pool_size') == size], 'select_seconds')
            ys.append(median(values) if values else float('nan'))
        axes[0].plot(sizes, ys, marker=marker, linestyle=line, color=colour,
                     label=selector, markersize=4)
    extraction = [median(column([row for row in at_k
                                 if number(row, 'pool_size') == size], 'extract_seconds'))
                  for size in sizes]
    axes[0].plot(sizes, extraction, color='0.45', linestyle='-.', marker='.',
                 label='behaviour extraction')
    axes[0].set(xscale='log', yscale='log', xlabel='pool size',
                ylabel='seconds (median over tasks)',
                title=f'Selection cost against pool size (k = {k})')
    axes[0].legend(fontsize=7)

    largest = max(sizes) if sizes else 0
    at_size = [row for row in rows if number(row, 'pool_size') == largest]
    ks = sorted({int(number(row, 'k')) for row in at_size})
    for selector in harness.SELECTORS:
        colour, marker, line = STYLE[selector]
        ys = []
        for value in ks:
            values = column([row for row in at_size
                             if row['selector'] == selector
                             and number(row, 'k') == value], 'select_seconds')
            ys.append(median(values) if values else float('nan'))
        axes[1].plot(ks, ys, marker=marker, linestyle=line, color=colour,
                     label=selector, markersize=4)
    axes[1].set(xscale='log', yscale='log', xlabel='k',
                ylabel='seconds (median over tasks)',
                title=f'Selection cost against k (pool size = {largest})')
    figure.tight_layout()
    return save(figure, out_dir, 'exp_a_selection_cost')


def figure_exp_a_distances(rows, out_dir, plt):
    """Distance evaluations against cache misses: what the cache is worth."""
    k = dominant_k(rows)
    at_k = [row for row in rows if number(row, 'k') == k]
    sizes = sorted({int(number(row, 'pool_size')) for row in at_k})
    figure, axis = plt.subplots(figsize=(5, 3.4))
    for selector in harness.SELECTORS:
        if selector == 'maxsum_stability':
            continue    # never touches the behaviour distance
        colour, marker, line = STYLE[selector]
        evals, misses = [], []
        for size in sizes:
            group = [row for row in at_k if row['selector'] == selector
                     and number(row, 'pool_size') == size]
            evals.append(median(column(group, 'n_distance_evals')))
            misses.append(median(column(group, 'n_distance_misses')))
        axis.plot(sizes, evals, marker=marker, linestyle='-', color=colour,
                  label=f'{selector} (calls)', markersize=4)
        axis.plot(sizes, misses, marker=marker, linestyle=':', color=colour,
                  alpha=0.6, label=f'{selector} (cache misses)', markersize=4)
    axis.set(xscale='log', yscale='log', xlabel='pool size',
             ylabel='distance evaluations (median)',
             title=f'Behaviour distances: calls vs misses (k = {k})')
    axis.legend(fontsize=6)
    figure.tight_layout()
    return save(figure, out_dir, 'exp_a_distance_calls')


def figure_exp_b_hidden(rows, out_dir, plt):
    """The hidden preference, per rule, over both strata."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    strata = [('every task', lambda r: True),
              ('b_pool >= k', lambda r: (number(r, 'b_pool') or 0) >= (number(r, 'k') or 0))]
    for axis, (title, keep) in zip(axes, strata):
        data, labels, colours = [], [], []
        for selector in harness.SELECTORS:
            values = column([row for row in rows
                             if row['selector'] == selector and keep(row)],
                            'hidden_pref_rate')
            if values:
                data.append(values)
                labels.append(selector)
                colours.append(STYLE[selector][0])
        if not data:
            continue
        box = axis.boxplot(data, patch_artist=True, widths=0.6,
                           medianprops={'color': 'black'})
        for patch, colour in zip(box['boxes'], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.45)
        axis.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        axis.set_title(f'{title}  (n = {len(data[0])})')
    axes[0].set_ylabel('hidden-preference hit rate')
    figure.suptitle('Does the selection contain a plan the user secretly wanted?', y=1.02)
    figure.tight_layout()
    return save(figure, out_dir, 'exp_b_hidden_preference')


def figure_exp_b_redundancy(rows, out_dir, plt):
    """Redundancy split into what the pool forces and what the rule adds."""
    figure, axis = plt.subplots(figsize=(5.5, 3.4))
    labels, floors, excesses = [], [], []
    for selector in harness.SELECTORS:
        group = [row for row in rows if row['selector'] == selector]
        if not group:
            continue
        labels.append(selector)
        floors.append(median(column(group, 'redundancy_floor')))
        excesses.append(median(column(group, 'redundancy_excess')))
    positions = range(len(labels))
    axis.bar(positions, floors, color='0.7', label='forced by the pool (k - b)')
    axis.bar(positions, excesses, bottom=floors, color='#a83232',
             label="the rule's own share")
    axis.set_xticks(list(positions))
    axis.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
    axis.set(ylabel='redundancy = k - b_coverage (median)',
             title='Where the redundancy comes from')
    axis.legend(fontsize=7)
    figure.tight_layout()
    return save(figure, out_dir, 'exp_b_redundancy')


def figure_exp_b_overlap(rows, out_dir, plt):
    """Behaviour-level against plan-level agreement between the rules."""
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, (name, title) in zip(axes, [('jaccard', 'behaviour sets B(S)'),
                                          ('overlap_jaccard_plans', 'plan sets S')]):
        grid = []
        for a in harness.SELECTORS:
            row = []
            for b in harness.SELECTORS:
                values = column([r for r in rows if r['selector_i'] == a
                                 and r['selector_j'] == b], name)
                row.append(median(values) if values else float('nan'))
            grid.append(row)
        image = axis.imshow(grid, vmin=0, vmax=1, cmap='viridis')
        axis.set_xticks(range(len(harness.SELECTORS)))
        axis.set_yticks(range(len(harness.SELECTORS)))
        axis.set_xticklabels(harness.SELECTORS, rotation=40, ha='right', fontsize=6)
        axis.set_yticklabels(harness.SELECTORS, fontsize=6)
        axis.set_title(f'Jaccard over {title}', fontsize=8)
        axis.grid(False)
        for i, line in enumerate(grid):
            for j, value in enumerate(line):
                axis.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6,
                          color='white' if value < 0.6 else 'black')
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle('How much the rules agree -- at the behaviour level, and at the plan level',
                    y=1.02, fontsize=9)
    figure.tight_layout()
    return save(figure, out_dir, 'exp_b_overlap')


def figure_exp_c(rows, out_dir, plt):
    """Responsiveness and sensitivity over the weight sweep."""
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row['task_id']].append(row)
    for group in by_task.values():
        group.sort(key=lambda row: number(row, 'w_go'))

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for axis, name, title in ((axes[0], 'spread_go', 'spread along go'),
                              (axes[1], 'spread_ru', 'spread along ru')):
        for group in by_task.values():
            xs = [number(row, 'w_go') for row in group]
            ys = [number(row, name) for row in group]
            if any(y is None for y in ys):
                continue
            axis.plot(xs, ys, color='0.7', linewidth=0.6, alpha=0.6)
        steps = sorted({number(row, 'w_go') for row in rows})
        medians = [median([number(row, name) for row in rows
                           if number(row, 'w_go') == step and number(row, name) is not None])
                   for step in steps]
        axis.plot(steps, medians, color='#a83232', linewidth=2, marker='o',
                  markersize=3, label='median over tasks')
        axis.set(xlabel='$w_{go}$', ylabel=name, title=title)
        axis.legend(fontsize=7)

    counts = [number(group[-1], 'n_distinct_behaviour_sets_so_far')
              for group in by_task.values()]
    counts = [c for c in counts if c is not None]
    if counts:
        axes[2].hist(counts, bins=range(1, int(max(counts)) + 2), color='#1b6ca8',
                     alpha=0.75, align='left')
    axes[2].set(xlabel='distinct behaviour sets over the sweep',
                ylabel='tasks', title='Sensitivity to the weights')
    figure.suptitle('Experiment C: does raising $w_{go}$ move the selection?',
                    y=1.03, fontsize=9)
    figure.tight_layout()
    return save(figure, out_dir, 'exp_c_weights')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--exp-a', action='append', default=[])
    parser.add_argument('--exp-b-scores', action='append', default=[])
    parser.add_argument('--exp-b-overlap', action='append', default=[])
    parser.add_argument('--exp-c', action='append', default=[])
    parser.add_argument('--figures-dir',
                        default=os.path.join(common.PAPER_EXPERIMENTS_DIR, 'out', 'figures'))
    args = parser.parse_args(argv)

    plt = pyplot()
    written = []
    exp_a = read_rows(args.exp_a)
    if exp_a:
        written += figure_exp_a_scaling(exp_a, args.figures_dir, plt)
        written += figure_exp_a_distances(exp_a, args.figures_dir, plt)
    exp_b = read_rows(args.exp_b_scores)
    if exp_b:
        written += figure_exp_b_hidden(exp_b, args.figures_dir, plt)
        written += figure_exp_b_redundancy(exp_b, args.figures_dir, plt)
    overlap = read_rows(args.exp_b_overlap)
    if overlap:
        written += figure_exp_b_overlap(overlap, args.figures_dir, plt)
    exp_c = read_rows(args.exp_c)
    if exp_c:
        written += figure_exp_c(exp_c, args.figures_dir, plt)

    for path in written:
        print(f'wrote {path}')
    if not written:
        print('nothing to plot; pass --exp-a / --exp-b-scores / --exp-b-overlap / --exp-c')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
