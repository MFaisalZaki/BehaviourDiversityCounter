"""E6 -- what the second phase costs.

Claim C6 says the representation's cost is knowledge engineering and not
computation, and that an indicator runs in O(k n c_ext + b^2 n c_dist). Each
pool of the configured sizes is timed repeatedly: a cold replay of the whole
pool into the behaviour space, then each of the four selections at each k, both
clocks on every sample and every sample kept. The planner's own time is in the
pool record, so the comparison is against the pool that was measured. The
generic model is rebuilt at the configured feature counts so that the n of the
cost expression is varied rather than assumed.
"""

import statistics
import time
from itertools import product

from bdc_experiments import models, pools, runner
from bdc_experiments import report as rp

BASE = ['instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b']

COLUMNS = BASE + ['k_requested', 'pool_stem', 'features', 'phase', 'indicator', 'repeat',
                  'wall_s', 'cpu_s', 'generation_wall_s', 'generation_cpu_s',
                  'wall_over_generation', 'cpu_over_generation', 'exhausted']

COST = ('O(k n c_ext + b^2 n c_dist): k n behaviour extractions over n features, and the '
        'b^2 distances between the distinct behaviours')

#: How the two phases are isolated from one another, said wherever a number is.
PROTOCOL = ('A mapping sample builds a fresh counter -- construction included in the sample -- '
            'with no trace cache, and maps the whole pool, so it pays the replay and the feature '
            'extraction. The selection samples of one model share a single counter, built over '
            'the task-wide trace cache and mapped once, untimed; only its behaviour-distance '
            'cache is cleared before each sample, so a sample pays the distances and the greedy '
            'cold and the replay not at all. Samples are taken at every k the config lists. '
            'kappa reaches B-Novelty alone, so the other three indicators are timed at the first '
            'configured kappa rather than repeatedly at each of them.')

#: Why the planner is compared on the wall clock and not on the CPU one.
GENERATION_CLOCK = ('The planner runs as a subprocess, so the pool record measures its CPU with '
                    "getrusage(RUSAGE_CHILDREN) around the call rather than the parent's "
                    'process_time, and generation_cpu_s is the planner\'s own CPU. Both ratios '
                    'are in e6_timing.csv; the table and the figure use the wall clock, which is '
                    'the clock a user waits on. A pool generated before this was corrected '
                    "carries a generation_cpu_s of nearly zero -- the parent's wait -- and the "
                    'cpu_over_generation column is then meaningless for it.')

NO_FIT = ('The lines in the figures are reference slopes, positioned to pass through the median '
          'sample; no curve was fitted to these data.')

#: Samples of different series land on the same point, so the markers are open
#: and shaped apart rather than filled discs one on top of another.
MARKERS = ('o', 's', '^', 'D', 'v')


def tasks(cfg):
    """The default grid, restricted to the pool sizes E6 times."""
    sizes = set(cfg['e6']['pool_sizes'])
    timed = {str(path) for path in pools.pool_files(cfg)
             if pools.read_pool(path)['requested'] in sizes}
    return [task_id for task_id in runner.default_tasks(cfg, 'e6')
            if str(runner.context(cfg, task_id)['pool_path']) in timed]


def _variant(cfg, n, domain):
    """The n-feature model to time, or ``(None, why it cannot be built)``."""
    configured = cfg['models']['generic']['features']
    if n == 1:
        return models.generic_spec(cfg, features=['goal_order'], name='generic-n1'), None
    if n == 3:
        per_domain = [s for s in models.models_for(cfg, domain) if s.domains is not None]
        if not per_domain:
            return None, (f'no per-domain model applies to {domain}, so the three-feature '
                          'variant (its features plus the cost bin) is not defined')
        spec = per_domain[0]
        features = tuple(models.FeatureSpec(f.key, f.params) for f in spec.features)
        width = cfg['models']['generic']['cost_bin_width']
        return (models.ModelSpec(f'{spec.name}-cbin', spec.domains,
                                 features + (models.FeatureSpec('cbin', {'width': width}),)),
                None)
    if n == len(configured):
        return models.generic_spec(cfg), None
    return None, (f'no {n}-feature model is defined: one feature is goal_order alone, two is '
                  f"the config's pair of {len(configured)}, three is a per-domain model plus "
                  'the cost bin')


def _specs(cfg, spec, domain):
    """The models this task times, and the feature counts it cannot build."""
    if spec.name != 'generic':
        return [(len(spec.features), spec)], []
    chosen, missing = [], []
    for n in cfg['e6']['feature_counts']:
        variant, reason = _variant(cfg, n, domain)
        if variant is None:
            missing.append({'features': n, 'reason': reason})
        else:
            chosen.append((n, variant))
    return chosen, missing


def _ratio(value, reference):
    """value / reference, or None where the reference is missing or zero."""
    return None if not reference else value / reference


def _row(base, sample, phase, **fields):
    """One timing row, against the planner's own time where the pool records it."""
    return {**base, **fields, 'phase': phase, **sample,
            'wall_over_generation': _ratio(sample['wall_s'], base['generation_wall_s']),
            'cpu_over_generation': _ratio(sample['cpu_s'], base['generation_cpu_s'])}


def run_task(task_id, cfg):
    """Both clocks on the mapping and on every selection, repeats times over."""
    ctx = runner.context(cfg, task_id)
    spec = models.registry(cfg)[ctx['extra'][0]]
    task, _, loaded, model_record, _ = runner.setup(cfg, ctx, spec)
    if not loaded['plans']:
        return {'pool': loaded['record'], 'model': model_record, 'rows': [],
                'extra': {'skipped': 'the pool loaded no plans, so there is nothing to map '
                                     'or to select from'}}

    plans, record = loaded['plans'], loaded['record']
    info = runner.instance_info(cfg, loaded['pool'])
    chosen, missing = _specs(cfg, spec, ctx['domain'])
    repeats = range(cfg['e6']['repeats'])
    trace, rows, timed = {}, [], []

    for n, variant in chosen:
        try:
            task, built, loaded, variant_record, variant_dump = runner.setup(
                cfg, ctx, variant, trace_cache=trace, loaded=loaded)
        except ValueError as failure:
            missing.append({'features': n, 'reason': f'{type(failure).__name__}: {failure}'})
            continue
        base = {**runner.base_row(loaded, variant_dump, variant_record),
                'pool_stem': record['pool_stem'], 'features': n, 'indicator': None,
                'k_requested': None, 'generation_wall_s': record['generation_wall_s'],
                'generation_cpu_s': record['generation_cpu_s'], 'exhausted': record['exhausted']}

        mapping = []
        for repeat in repeats:
            wall, cpu = time.perf_counter(), time.process_time()
            fresh = models.build_counter(variant, task, info)     # no trace cache: a cold replay
            fresh.b_coverage(plans)
            mapping.append({'repeat': repeat, 'wall_s': time.perf_counter() - wall,
                            'cpu_s': time.process_time() - cpu})
            rows.append(_row(base, mapping[-1], 'mapping'))

        selections = []
        kappas = cfg['selection']['kappa_values']
        # One counter for every selection sample of this model, mapped once and
        # untimed: a remap per sample outweighs the sample itself on a big pool.
        warm = models.build_counter(variant, task, info, trace_cache=trace)
        warm.b_coverage(plans)                         # timed above, so not timed again here
        for k, indicator, repeat in product(cfg['selection']['k_values'],
                                            runner.INDICATORS, repeats):
            # k_nn is read by the B-Novelty rule and by no other, so timing the
            # other three at every kappa would repeat one computation.
            for kappa in (kappas if indicator == 'bnovelty' else kappas[:1]):
                warm._behaviour_distance_cache.clear()   # no public reset; every sample pays b^2
                taken = min(k, len(plans))
                selected, wall, cpu = runner.select(warm, plans, taken, indicator, kappa)
                selections.append({'k': taken, 'k_requested': k, 'kappa': kappa,
                                   'indicator': indicator, 'repeat': repeat,
                                   **runner.selection_record(loaded, variant_dump, selected,
                                                             wall, cpu)})
                rows.append(_row(base, {'repeat': repeat, 'wall_s': wall, 'cpu_s': cpu},
                                 'selection', indicator=indicator, k=taken, k_requested=k,
                                 kappa=kappa))

        timed.append({'name': variant.name, 'features': n, 'model': variant_record,
                      'b': len(variant_dump['distinct']), 'space_size': models.space_size(built),
                      'mapping': mapping, 'selections': selections})

    return {'pool': record, 'model': model_record, 'rows': rows,
            'extra': {'timed': timed, 'missing': missing, 'cost_model': COST,
                      'timing_protocol': PROTOCOL, 'generation_clock': GENERATION_CLOCK,
                      'repeats': cfg['e6']['repeats'],
                      'feature_counts': cfg['e6']['feature_counts'],
                      'k_values': cfg['selection']['k_values'],
                      'kappa_values': cfg['selection']['kappa_values'],
                      'generation_wall_s': record['generation_wall_s'],
                      'generation_cpu_s': record['generation_cpu_s']}}


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _median(members, key):
    values = [m[key] for m in members if m.get(key) is not None]
    return statistics.median(values) if values else None


def _tex(text):
    """Prose written for the JSON dumps, made safe to read as LaTeX text."""
    return text.replace('_', r'\_')


def _medians(rows, keys):
    """Per group: both clocks as median and IQR, and the CPU time two ways.

    The model is one of the keys: grouped on the feature count alone, the
    generic variants would pool with the per-domain models, whose dimensions
    cost differently to extract, and n would be read off a model mix.
    """
    out = []
    for values, members in rp.group(rows, keys).items():
        wall, cpu = (rp.summarise([m['wall_s'] for m in members]),
                     rp.summarise([m['cpu_s'] for m in members]))
        out.append({**dict(zip(keys, values)), 'samples': len(members),
                    'wall_median': wall['median'], 'wall_q1': wall['q1'], 'wall_q3': wall['q3'],
                    'cpu_median': cpu['median'], 'cpu_q1': cpu['q1'], 'cpu_q3': cpu['q3'],
                    'cpu_pooled_mean': rp.pooled(members, 'cpu_s'),
                    'cpu_macro_mean': rp.macro(members, 'cpu_s'),
                    'pool_size_median': _median(members, 'pool_size'),
                    'b_median': _median(members, 'b'),
                    'generation_wall_median': _median(members, 'generation_wall_s'),
                    'generation_cpu_median': _median(members, 'generation_cpu_s'),
                    'wall_over_generation_median': _median(members, 'wall_over_generation'),
                    'cpu_over_generation_median': _median(members, 'cpu_over_generation'),
                    'wall_over_generation_macro': rp.macro(members, 'wall_over_generation'),
                    'cpu_over_generation_macro': rp.macro(members, 'cpu_over_generation')})
    return sorted(out, key=lambda row: (row['N'], row['phase'], row['features'], row['model'],
                                        row['k'] or 0, row['indicator'] or ''))


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
    """One series of samples, as open markers."""
    if points:
        ax.scatter([x for x, _ in points], [y for _, y in points], s=area, alpha=0.7,
                   marker=marker or MARKERS[offset % len(MARKERS)], facecolors='none',
                   linewidths=0.8, edgecolors=rp.PALETTE[offset % len(rp.PALETTE)], label=label)
    return points


def _mapping_figure(rows, path):
    """Mapping and planning against the pool size, both on the wall clock."""
    fig, ax = rp.figure()
    mapping = [r for r in rows if r['phase'] == 'mapping' and r['pool_size'] and r['wall_s'] > 0]
    points = []
    for offset, n in enumerate(sorted({r['features'] for r in mapping})):
        points += _scatter(ax, [(r['pool_size'], r['wall_s']) for r in mapping
                                if r['features'] == n], offset, f'mapping, n = {n}')
    _scatter(ax, sorted({(r['pool_size'], r['generation_wall_s']) for r in mapping
                         if r['generation_wall_s']}), len(rp.PALETTE) - 1, 'pool generation',
             marker='P')
    # n is the feature count throughout E6, so the pool size is |P| and never n.
    drawn = _slope(ax, points, 1, 'reference slope 1 in $|P|$ (no fit)')
    if points:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('pool size $|P|$')
    ax.set_ylabel('wall-clock time (s)')
    if points:
        ax.legend(fontsize='small')
    return rp.save(fig, path), drawn


def _selection_figure(rows, path):
    """Selection CPU time against b, one colour per indicator, marker area per
    pool size: a selection's cost moves with the pool size as well as with b,
    so the slope in b is only readable within one pool size."""
    fig, ax = rp.figure()
    selection = [r for r in rows if r['phase'] == 'selection' and r['b'] and r['cpu_s'] > 0]
    sizes = sorted({r['pool_size'] for r in selection})
    for offset, indicator in enumerate(runner.INDICATORS):
        for rank, size in enumerate(sizes):
            _scatter(ax, [(r['b'], r['cpu_s']) for r in selection
                          if r['indicator'] == indicator and r['pool_size'] == size],
                     offset, indicator if rank == 0 else None, area=10 + 12 * rank)
    # Positioned within the largest pool size alone, so that the eye is not
    # asked to read a slope in b off variation that is really in the pool size.
    biggest = sizes[-1] if sizes else None
    drawn = _slope(ax, [(r['b'], r['cpu_s']) for r in selection if r['pool_size'] == biggest], 2,
                   f'reference slope 2 in $b$ at pool size {biggest} (no fit)')
    if selection:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('distinct behaviours $b$')
    ax.set_ylabel('selection CPU time (s)')
    if selection:
        ax.legend(fontsize='small')
    return rp.save(fig, path), bool(drawn)


def report(cfg, results):
    """Every sample, the medians per (N, k), and the two log-log figures."""
    out = rp.report_dir(cfg, 'e6')
    rows = rp.all_rows(results)
    medians = _medians(rows, ('N', 'k', 'model', 'features', 'phase', 'indicator'))
    figures = out / 'figures'
    mapping_path, mapping_slope = _mapping_figure(rows, figures / 'e6_mapping_vs_generation.pdf')
    selection_path, selection_slope = _selection_figure(rows, figures / 'e6_selection_vs_b.pdf')

    columns = ['N', 'k', 'model', 'features', 'phase', 'indicator', 'samples', 'wall_median',
               'wall_q1', 'wall_q3', 'cpu_median', 'cpu_q1', 'cpu_q3', 'cpu_pooled_mean',
               'cpu_macro_mean', 'pool_size_median', 'b_median', 'generation_wall_median',
               'generation_cpu_median', 'wall_over_generation_median',
               'wall_over_generation_macro', 'cpu_over_generation_median',
               'cpu_over_generation_macro']
    written = [
        rp.write_csv(out / 'e6_timing.csv', rows, COLUMNS),
        rp.write_csv(out / 'e6_medians.csv', medians, columns),
        rp.table(out / 'tables' / 'e6_medians.tex', 'tab:e6-medians',
                 'Median cost of the second phase per requested pool size $N$, selection size $k$ '
                 'and model: mapping a whole pool into the behaviour space, and one selection. '
                 '$n$ is the number of features of the model timed, and the rows are kept apart '
                 'by model so that a change in $n$ is read within one model and not across a mix '
                 'of them; mapping rows have no $k$ and no indicator. Where a pool holds fewer '
                 'plans than $k$ the selection takes the whole pool and $k$ here is that clamped '
                 'value, the requested one being a column of e6\\_timing.csv. Every sample, at '
                 'every kappa and every repeat, is in e6\\_timing.csv; the medians here are over '
                 'the repeats and, for B-Novelty, over kappa. The last two columns are the median '
                 'over samples, and the mean of the per-domain means, of one sample\'s wall-clock '
                 'time divided by the wall clock the planner spent producing that same pool: each '
                 'ratio is formed sample by sample, so neither is the quotient of two medians. '
                 + _tex(GENERATION_CLOCK) + ' ' + _tex(PROTOCOL)
                 + ' The two figures carry reference slopes positioned through the median '
                 'sample and no fitted curve. ' + rp.TIE_RULE,
                 ['N', 'k', 'model', 'n features', 'phase', 'indicator', 'samples',
                  'median wall (s)', 'median CPU (s)', 'pooled CPU (s)', 'macro CPU (s)',
                  'median wall / generation', 'macro wall / generation'],
                 [[row['N'], row['k'], row['model'], row['features'], row['phase'],
                   row['indicator'], row['samples'], row['wall_median'], row['cpu_median'],
                   row['cpu_pooled_mean'], row['cpu_macro_mean'],
                   row['wall_over_generation_median'], row['wall_over_generation_macro']]
                  for row in medians], digits=6),
        mapping_path, selection_path,
    ]
    mapping_rows = [r for r in rows if r['phase'] == 'mapping']
    written.append(rp.manifest(cfg, 'e6', written, results, extra={
        'cost_model': COST, 'timing_protocol': PROTOCOL, 'no_fit': NO_FIT,
        'generation_clock': GENERATION_CLOCK,
        'repeats': cfg['e6']['repeats'], 'pool_sizes': cfg['e6']['pool_sizes'],
        'feature_counts': cfg['e6']['feature_counts'],
        'k_values': cfg['selection']['k_values'], 'kappa_values': cfg['selection']['kappa_values'],
        'samples': {'mapping': len(mapping_rows),
                    'selection': sum(1 for r in rows if r['phase'] == 'selection')},
        'zero_cpu_samples': sum(1 for r in rows if not r['cpu_s']),
        'reference_slope_pool_size': ('drawn' if mapping_slope else
                                      'not drawn: the samples cover a single pool size'),
        'reference_slope_b': ('drawn, within the largest pool size' if selection_slope else
                              'not drawn: the largest pool size covers a single value of b'),
        'mapping_wall_over_generation_median': _median(mapping_rows, 'wall_over_generation'),
        'mapping_wall_over_generation_macro': rp.macro(mapping_rows, 'wall_over_generation'),
        'missing_variants': [{'task_id': result['task_id'], **entry}
                             for result in rp.usable(results)
                             for entry in result.get('extra', {}).get('missing', [])],
    }))
    return written
