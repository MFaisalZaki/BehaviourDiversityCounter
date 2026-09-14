"""E5 -- feature resolution.

Claim C5 says the resolution of the features decides what the behaviour space
distinguishes, and that the cell count grows as the product of the dimension
sizes. The generic model is rebuilt at every goal cap and every cost bin width
the config lists, over one pool at a time and through one shared trace, and
each variant is measured by the |BS| it declares, the b the pool exposes in it,
and what the four selections then score and cost.

One task carries the whole grid: at the default config eighteen dense b x b
behaviour dumps for one pool under one selection time limit. It is walked
cheapest cap first, so a timeout loses the expensive tail rather than the whole
pool; capping the dumped matrix would have to come from pools and runner.
"""

import statistics

from bdc_experiments import models, pools, runner
from bdc_experiments import report as rp

BASE = ['instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b']

COLUMNS = BASE + ['pool_stem', 'goal_cap', 'cost_bin_width', 'goal_atoms', 'goal_orders',
                  'cost_bins', 'space_size', 'b_over_pool', 'b_over_space', 'saturated',
                  'cost_bin_degenerate', 'indicator', 'bcoverage', 'bmaxsum', 'bmaxmin',
                  'bnovelty', 'wall_s', 'cpu_s']

#: One behaviour space: the pool and the two resolution knobs fix it.
CELL = ('instance', 'pool_stem', 'goal_cap', 'cost_bin_width')

#: Open markers, so two domains whose medians coincide stay countable.
MARKERS = ('o', 's', '^', 'v', 'D', 'P', 'X', '*', '<')

#: Said in the report rather than dropped: at q = 1 the pool holds only optimal
#: plans, the cost grid over [1, q] is one bin wide and the feature is constant.
DEGENERATE = ('At q = 1 the cost bin dimension has a single bin, so the feature is '
              'constant over the pool and the bin width changes nothing; those variants '
              'are kept and flagged rather than dropped.')


def tasks(cfg):
    """One task per pool for the generic model alone: the resolution variants
    are looped inside the task, so a pool is replayed once and not once each."""
    ids = [task_id for task_id in runner.default_tasks(cfg, 'e5')
           if task_id.rsplit('/', 1)[1] == 'generic']
    # An empty grid over a non-empty pool set reports as a clean zero-task
    # sweep, which is a broken build that says nothing about why.
    if not ids and pools.pool_files(cfg):
        raise ValueError('E5 varies the generic model, and [models].enabled does not list it; '
                         'add "generic" or E5 has no behaviour space to vary')
    return ids


def run_task(task_id, cfg):
    """Every (goal cap, cost bin width) variant of the generic model on one pool."""
    ctx = runner.context(cfg, task_id)
    base_spec = models.generic_spec(cfg)
    # Ascending, so a task that runs out of time has done the cheap caps first.
    grid = sorted((cap, width) for cap in cfg['e5']['goal_caps']
                  for width in cfg['e5']['cost_bin_widths'])
    trace, loaded, task, rows, variants = {}, None, None, [], []

    for goal_cap, width in grid:
        spec = models.generic_spec(cfg, goal_cap=goal_cap, cost_bin_width=width,
                                   name=f'generic-g{goal_cap}-w{width}')
        task, counter, loaded, record, dump = runner.setup(cfg, ctx, spec,
                                                           trace_cache=trace, loaded=loaded)
        plans, b = loaded['plans'], len(dump['distinct'])
        if not plans:
            return {'pool': loaded['record'], 'model': record, 'rows': [],
                    'extra': {'skipped': 'the pool loaded no plans, so it exposes no '
                                         'behaviour space at any resolution'}}

        sizes = {key: models.dimension_size(counter, key) for key in counter.dimensions}
        space = models.space_size(counter)
        goal = counter.dimensions.get('go')
        k = min(cfg['e5']['k'], len(plans))
        shape = {'pool_stem': loaded['record']['pool_stem'], 'goal_cap': goal_cap,
                 'cost_bin_width': width,
                 'goal_atoms': len(goal.vars) if goal is not None else None,
                 'goal_orders': sizes.get('go'), 'cost_bins': sizes.get('cbin'),
                 'space_size': space, 'b_over_pool': b / len(plans),
                 'b_over_space': (b / space if space else None),
                 'saturated': b == len(plans),
                 # Not applicable, rather than False, without a cost dimension.
                 'cost_bin_degenerate': (sizes['cbin'] == 1
                                         if 'cbin' in counter.dimensions else None)}

        selections = []
        for kappa in cfg['selection']['kappa_values']:
            for indicator in runner.INDICATORS:
                selected, wall, cpu = runner.select(counter, plans, k, indicator, kappa)
                values = runner.indicators(counter, selected, kappa)
                selections.append({'indicator': indicator, 'kappa': kappa, 'k': k,
                                   **runner.selection_record(loaded, dump, selected, wall, cpu)})
                rows.append({**runner.base_row(loaded, dump, record, k=k, kappa=kappa),
                             **shape, 'indicator': indicator, **values,
                             'wall_s': wall, 'cpu_s': cpu})
        variants.append({'name': spec.name, 'goal_cap': goal_cap, 'cost_bin_width': width,
                         'model': record, 'b': b, 'k': k, 'dimension_sizes': sizes,
                         'space_size': space, 'selections': selections})

    generic = models.model_record(base_spec, task, runner.instance_info(cfg, loaded['pool']))
    return {'pool': loaded['record'], 'model': generic, 'rows': rows,
            'extra': {'variants': variants, 'grid': [list(pair) for pair in grid],
                      'k_requested': cfg['e5']['k'],
                      'space_size_rule': 'the product of the dimension sizes: m! goal orders '
                                         'over the m capped goal atoms, times the cost bin count',
                      'degenerate_cost_bin': DEGENERATE}}


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _variants(rows):
    """One row per behaviour space: the selection rows repeat b and |BS|."""
    seen = {}
    for row in rows:
        seen.setdefault(tuple(row[key] for key in CELL), row)
    return list(seen.values())


def _median(members, key):
    values = [m[key] for m in members if m.get(key) is not None]
    return statistics.median(values) if values else None


def _timings(rows, keys):
    """Median selection CPU time per group, over every selection made.

    It has to come from the rows and not from the variants: b and |BS| are
    constant over a behaviour space, the time each selection took is not.
    """
    return {values: rp.summarise([row['cpu_s'] for row in members])['median']
            for values, members in rp.group(rows, keys).items()}


def _entry(members, domain, goal_cap, width, cpu_median):
    """One summary row: how b, b/|BS| and b/n move at this resolution."""
    stats = rp.summarise([m['b'] for m in members])
    return {'domain': domain, 'goal_cap': goal_cap, 'cost_bin_width': width,
            'pools': len(members), 'b_median': stats['median'], 'b_q1': stats['q1'],
            'b_q3': stats['q3'], 'b_min': stats['min'], 'b_max': stats['max'],
            'b_pooled_mean': rp.pooled(members, 'b'), 'b_macro_mean': rp.macro(members, 'b'),
            'space_size_median': _median(members, 'space_size'),
            'b_over_space_median': _median(members, 'b_over_space'),
            'b_over_space_pooled': rp.pooled(members, 'b_over_space'),
            'b_over_space_macro': rp.macro(members, 'b_over_space'),
            'b_over_pool_median': _median(members, 'b_over_pool'),
            'cpu_s_median': cpu_median,
            'saturated_pools': sum(1 for m in members if m['saturated']),
            'degenerate_cost_bin_pools': sum(1 for m in members if m['cost_bin_degenerate'])}


def _summary(variants, rows):
    """Per (domain, cap, width), then the same over every domain at once, where
    the macro mean stops agreeing with the pooled one."""
    keys = ('domain', 'goal_cap', 'cost_bin_width')
    per_domain, over_all = _timings(rows, keys), _timings(rows, keys[1:])
    out = [_entry(members, *values, per_domain.get(values))
           for values, members in rp.group(variants, keys).items()]
    out += [_entry(members, 'all', cap, width, over_all.get((cap, width)))
            for (cap, width), members in rp.group(variants, keys[1:]).items()]
    return sorted(out, key=lambda row: (row['domain'], row['goal_cap'], row['cost_bin_width']))


def _saturation(variants):
    """The smallest goal cap at which every plan of a pool is its own behaviour."""
    out = []
    for values, members in rp.group(variants, ('instance', 'pool_stem', 'cost_bin_width')).items():
        reached = sorted(m['goal_cap'] for m in members if m['saturated'])
        out.append({'instance': values[0], 'pool_stem': values[1], 'cost_bin_width': values[2],
                    'pool_size': members[0]['pool_size'],
                    'max_b': max(m['b'] for m in members),
                    'first_saturating_goal_cap': reached[0] if reached else None})
    return sorted(out, key=lambda row: (row['instance'], row['pool_stem'], row['cost_bin_width']))


def _legend(ax):
    """A legend only where something was drawn: an empty report still builds."""
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize='small')


def _cap_figure(variants, path):
    """b against the goal cap, one line per domain, median with the IQR."""
    fig, ax = rp.figure()
    for offset, domain in enumerate(sorted({v['domain'] for v in variants})):
        points = []
        for cap in sorted({v['goal_cap'] for v in variants if v['domain'] == domain}):
            stats = rp.summarise([v['b'] for v in variants
                                  if v['domain'] == domain and v['goal_cap'] == cap])
            if stats['n']:
                points.append((cap, stats))
        if points:
            ax.errorbar([cap for cap, _ in points], [s['median'] for _, s in points],
                        yerr=[[s['median'] - s['q1'] for _, s in points],
                              [s['q3'] - s['median'] for _, s in points]],
                        color=rp.PALETTE[offset % len(rp.PALETTE)], capsize=2, linewidth=1.0,
                        marker=MARKERS[offset % len(MARKERS)], fillstyle='none', label=domain)
    ax.set_xlabel('goal cap')
    ax.set_ylabel('b')
    _legend(ax)
    return rp.save(fig, path)


def _time_figure(rows, path):
    """Selection CPU time against b, log-log, with the b^2 term of C6 drawn in."""
    fig, ax = rp.figure()
    for offset, indicator in enumerate(runner.INDICATORS):
        points = [(r['b'], r['cpu_s']) for r in rows
                  if r['indicator'] == indicator and r['b'] > 0 and r['cpu_s'] > 0]
        if points:
            ax.scatter([b for b, _ in points], [t for _, t in points], s=9,
                       color=rp.PALETTE[offset], label=indicator)
    positive = sorted((r['b'], r['cpu_s']) for r in rows if r['b'] > 1 and r['cpu_s'] > 0)
    if positive:
        # The guide is anchored at the smallest (b, time) point and drawn as
        # b^2; it is a slope to read the scatter against, not a fit.
        (b0, t0), xs = positive[0], sorted({b for b, _ in positive})
        ax.plot(xs, [t0 * (x / b0) ** 2 for x in xs], color='black', linewidth=0.8,
                linestyle='--', label='$b^2$ from the cheapest point')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('b')
    ax.set_ylabel('selection CPU time (s)')
    _legend(ax)
    return rp.save(fig, path)


def report(cfg, results):
    """The per-selection rows, the per-resolution summary, and the two figures."""
    out = rp.report_dir(cfg, 'e5')
    rows = rp.all_rows(results)
    variants = _variants(rows)
    summary = _summary(variants, rows)
    saturation = _saturation(variants)
    degenerate = [v for v in variants if v['cost_bin_degenerate']]

    columns = ['domain', 'goal_cap', 'cost_bin_width', 'pools', 'b_median', 'b_q1', 'b_q3',
               'b_min', 'b_max', 'b_pooled_mean', 'b_macro_mean', 'space_size_median',
               'b_over_space_median', 'b_over_space_pooled', 'b_over_space_macro',
               'b_over_pool_median', 'cpu_s_median', 'saturated_pools',
               'degenerate_cost_bin_pools']
    shown = [row for row in summary if row['domain'] == 'all'] or summary
    written = [
        rp.write_csv(out / 'e5_resolution.csv', rows, COLUMNS),
        rp.write_csv(out / 'e5_summary.csv', summary, columns),
        rp.table(out / 'tables' / 'e5_resolution.tex', 'tab:e5-resolution',
                 'Feature resolution over every pool: the generic model at each goal cap and '
                 'cost bin width, its cell count $|BS|$ as the product of the dimension sizes, '
                 'and the $b$ the pool then exposes. Rows are over all domains at once, so the '
                 'macro mean (the mean of the per-domain means) can differ from the pooled one; '
                 'the per-domain rows are in e5\\_summary.csv. ' + DEGENERATE + ' ' + rp.TIE_RULE,
                 # Plain words: report.table escapes the header cells, so maths
                 # here would print as literal dollar signs; the caption is raw.
                 ['cap', 'width', 'pools', 'median b', 'b IQR', 'pooled b', 'macro b',
                  'median |BS|', 'median b/|BS|', 'median b/n', 'saturated'],
                 [[row['goal_cap'], row['cost_bin_width'], row['pools'], row['b_median'],
                   (None if row['b_q1'] is None else f"{row['b_q1']:g}--{row['b_q3']:g}"),
                   row['b_pooled_mean'], row['b_macro_mean'], row['space_size_median'],
                   row['b_over_space_median'], row['b_over_pool_median'],
                   row['saturated_pools']] for row in shown]),
        _cap_figure(variants, out / 'figures' / 'e5_cap_vs_b.pdf'),
        _time_figure(rows, out / 'figures' / 'e5_time_vs_b.pdf'),
    ]
    written.append(rp.manifest(cfg, 'e5', written, results, extra={
        'goal_caps': cfg['e5']['goal_caps'], 'cost_bin_widths': cfg['e5']['cost_bin_widths'],
        'k': cfg['e5']['k'], 'kappa_values': cfg['selection']['kappa_values'],
        'behaviour_spaces': len(variants), 'selections': len(rows),
        'degenerate_cost_bin': DEGENERATE,
        'degenerate_variants': len(degenerate),
        'degenerate_q_values': sorted({v['q'] for v in degenerate}),
        'saturation': saturation,
        'saturated_spaces': sum(1 for v in variants if v['saturated']),
        'saturation_note': 'b = n is the point at which every plan of the pool is its own '
                           'behaviour; first_saturating_goal_cap is None where no cap reached it',
        'zero_cpu_selections': sum(1 for row in rows if not row['cpu_s']),
    }))
    return written
