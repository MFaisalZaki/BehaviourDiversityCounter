"""E2: whether the three dissimilarity-based indicators separate plan sets that
B-Coverage scores alike.

Part A draws random subsets of exactly k distinct behaviours -- so B-Coverage
is constant over the sample by construction -- and scores each under all four
indicators. Part B runs the four selections on the same pools and scores each
returned set under all four indicators, which gives the selecting-by-scoring
cross table.
"""

import math
import random
import statistics
from collections import Counter
from itertools import combinations

from bdc_experiments import models, runner
from bdc_experiments import report as rp

#: The pairs of dissimilarity-based indicators whose rankings are compared.
PAIRS = ('bmaxsum/bmaxmin', 'bmaxsum/bnovelty', 'bmaxmin/bnovelty')

BASE = ['instance', 'domain', 'q', 'N', 'model', 'k', 'kappa', 'pool_size', 'b']

#: A pool cell of the paired comparisons, as the plan fixes it.
CELL = ('instance', 'q', 'N', 'model', 'k', 'kappa')

#: Two subset scores count as one value within this tolerance: they are sums of
#: the same rationals in a different order, so bit equality is too strict a
#: reading of "the indicator is constant across the sample".
DIGITS = 12


def _draw(cfg, task_id, b, k):
    """``(subsets, enumerated)``: up to ``[e2].subsets`` distinct k-subsets of
    the b behaviours, or all C(b, k) of them when there are no more than that.

    The seed is the repr of (run seed, task id, k) -- random.Random takes no
    tuple -- so a rerun draws the same subsets.
    """
    wanted = cfg['e2']['subsets']
    if math.comb(b, k) <= wanted:
        return [list(subset) for subset in combinations(range(b), k)], True
    rng, drawn = random.Random(repr((cfg['run']['seed'], task_id, k))), {}
    while len(drawn) < wanted:
        drawn.setdefault(tuple(sorted(rng.sample(range(b), k))), None)
    return [list(subset) for subset in drawn], False


def _subset_rows(base, k, kappas, scores, enumerated):
    """One row per (kappa, indicator): how the sample's values spread, and
    whether the indicator is constant over it."""
    rows = []
    for kappa in kappas:
        for indicator in runner.INDICATORS:
            values = (scores['bnovelty'][str(kappa)] if indicator == 'bnovelty'
                      else scores[indicator])
            counts = Counter(round(value, DIGITS) for value in values)
            rows.append({**base, 'k': k, 'kappa': kappa, 'part': 'subsets',
                         'indicator': indicator, 'n_subsets': len(values),
                         'enumerated': enumerated, 'n_values': len(counts),
                         'constant': len(counts) == 1,
                         'modal_fraction': max(counts.values()) / len(values),
                         'min': min(values), 'max': max(values),
                         'mean': statistics.fmean(values)})
    return rows


def _selection(counter, loaded, dump, k, indicator, kappa):
    """``(entry, selected)``: one selection, dumped whole. The caller fills in
    the four indicator values it scores at this kappa."""
    selected, wall, cpu = runner.select(counter, loaded['plans'], k, indicator, kappa)
    return ({'k': k, 'kappa': kappa, 'indicator': indicator, 'values': None,
             'kappa_free': indicator != 'bnovelty',
             'selection': runner.selection_record(loaded, dump, selected, wall, cpu)},
            selected)


def run_task(task_id, cfg):
    """Part A (random equal-count subsets) and part B (the four selections)."""
    ctx = runner.context(cfg, task_id)
    spec = models.registry(cfg)[ctx['extra'][0]]
    task, counter, loaded, model_record, dump = runner.setup(cfg, ctx, spec)
    plans, b, record = loaded['plans'], len(dump['distinct']), loaded['record']
    if not plans:
        return {'pool': record, 'model': model_record, 'rows': [],
                'extra': {'skipped': 'the pool loaded no plans, so there is nothing to score'}}

    kappas = cfg['selection']['kappa_values']
    base = {'instance': record['instance'], 'domain': record['domain'], 'q': record['q'],
            'N': record['requested'], 'model': spec.name, 'pool_size': len(plans), 'b': b}
    representative = {}                        # distinct behaviour -> a plan exhibiting it
    for entry in dump['plans']:
        representative.setdefault(entry['distinct'], entry['index'])

    rows, subsets, selections = [], [], []
    for k in cfg['selection']['k_values']:
        if b > k:                              # at b <= k the only subset is every behaviour
            sets, enumerated = _draw(cfg, task_id, b, k)
            scores = {name: [] for name in ('bcoverage', 'bmaxsum', 'bmaxmin')}
            scores['bnovelty'] = {str(kappa): [] for kappa in kappas}
            for subset in sets:
                chosen = [plans[representative[index]] for index in subset]
                for kappa in kappas:
                    values = runner.indicators(counter, chosen, kappa)
                    scores['bnovelty'][str(kappa)].append(values['bnovelty'])
                    if kappa == kappas[0]:
                        for name in ('bcoverage', 'bmaxsum', 'bmaxmin'):
                            scores[name].append(values[name])
            subsets.append({'k': k, 'b': b, 'enumerated': enumerated, 'sets': sets,
                            'scores': scores})
            rows.extend(_subset_rows(base, k, kappas, scores, enumerated))
        fixed = {}                             # only B-Novelty's greedy step reads kappa, so
        for indicator in runner.INDICATORS:    # the other three are selected once per k
            if indicator != 'bnovelty':
                fixed[indicator] = (*_selection(counter, loaded, dump, k, indicator, kappas[0]),
                                    len(selections))
                selections.append(fixed[indicator][0])
        for kappa in kappas:
            for indicator in runner.INDICATORS:
                if indicator in fixed:
                    entry, selected, at = fixed[indicator]
                else:
                    entry, selected = _selection(counter, loaded, dump, k, indicator, kappa)
                    at = len(selections)
                    selections.append(entry)
                values = runner.indicators(counter, selected, kappa)
                if entry['kappa'] == kappa:
                    entry['values'] = values
                else:                          # the same set, scored at a further kappa
                    selections.append({'k': k, 'kappa': kappa, 'indicator': indicator,
                                       'kappa_free': True, 'values': values,
                                       'same_selection_as': at})
                rows.append({**base, 'k': k, 'kappa': kappa, 'part': 'selection',
                             'selector': indicator, 'selected': len(selected),
                             'selected_distinct': len(set(entry['selection']['distinct'])),
                             'wall_s': entry['selection']['wall_s'],
                             'cpu_s': entry['selection']['cpu_s'],
                             **{f'score_{name}': value for name, value in values.items()}})

    return {'pool': record, 'model': model_record, 'rows': rows,
            'extra': {'subsets': subsets, 'selections': selections,
                      'seeding': 'random.Random(repr((run seed, task id, k)))',
                      'kappa_free': 'B-Coverage, B-MaxSum and B-MaxMin do not read kappa: each is '
                                    'selected once per k and scored at every kappa, so its later '
                                    'kappas are dumped as scores naming the selection they reuse '
                                    'by its index in this list'}}


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _constancy(rows):
    """Per (k, kappa, indicator): the pool cells sampled, the fraction of them
    on which the indicator took a single value over the whole sample, and the
    mean modal fraction. The B-Coverage line is the construction check."""
    out = []
    for keys, cells in rp.group(rows, ['k', 'kappa', 'indicator']).items():
        out.append({'part': 'aggregate', **dict(zip(('k', 'kappa', 'indicator'), keys)),
                    'cells': len(cells),
                    'constant_fraction': sum(r['constant'] for r in cells) / len(cells),
                    'mean_modal_fraction': statistics.fmean(r['modal_fraction'] for r in cells)})
    return out


def _taus(results):
    """Kendall tau-b between each pair of dissimilarity indicators over the
    subsets drawn on one pool: a row per (pool, model, k, kappa, pair).

    ``enumerated`` marks the pools where the sample is every k-subset there is;
    the tau is then the population value and no sampling p-value applies."""
    rows = []
    for result in results:
        pool = result['pool']
        head = {'instance': pool['instance'], 'domain': pool['domain'], 'q': pool['q'],
                'N': pool['requested'], 'model': result['model']['name'],
                'pool_size': pool['size']}
        for block in result.get('extra', {}).get('subsets', []):
            for kappa in sorted(int(key) for key in block['scores']['bnovelty']):
                values = dict(block['scores'], bnovelty=block['scores']['bnovelty'][str(kappa)])
                for pair in PAIRS:
                    first, second = pair.split('/')
                    tau, p = rp.kendall(values[first], values[second])
                    rows.append({**head, 'k': block['k'], 'kappa': kappa, 'b': block['b'],
                                 'pair': pair, 'n_subsets': len(block['sets']),
                                 'enumerated': block['enumerated'], 'tau': tau,
                                 'p': None if block['enumerated'] else p})
    return rows


def _macro_spread(rows, key='tau'):
    """The mean over domains of each domain's median and quartiles: report.macro
    averages the per-domain means, and the spread is averaged the same way."""
    per_domain = {}
    for row in rows:
        if row[key] is not None:
            per_domain.setdefault(row['domain'], []).append(row[key])
    spreads = [rp.summarise(values) for values in per_domain.values()]
    return {f'macro_{name}': statistics.fmean([s[name] for s in spreads]) if spreads else None
            for name in ('median', 'q1', 'q3')}


def _tau_summary(rows):
    """Median and IQR pooled and macro, and the fraction of pools with a
    negative tau, per (pair, k, kappa). A tau is missing where an indicator was
    constant over the sample and the correlation is undefined."""
    out = []
    for pair in PAIRS:
        for k, kappa in sorted({(row['k'], row['kappa']) for row in rows}):
            group = [r for r in rows if (r['pair'], r['k'], r['kappa']) == (pair, k, kappa)]
            known = [r for r in group if r['tau'] is not None]
            spread = rp.summarise([r['tau'] for r in group])
            out.append({'pair': pair, 'k': k, 'kappa': kappa, 'n': spread['n'],
                        'undefined': len(group) - len(known), 'median': spread['median'],
                        'q1': spread['q1'], 'q3': spread['q3'], **_macro_spread(known),
                        'negative_fraction': (sum(r['tau'] < 0 for r in known) / len(known)
                                              if known else None)})
    return out


def _cross(results):
    """Each selection's value under each scoring indicator, over the best of
    the four selections on the same pool, k and kappa."""
    rows = []
    for result in results:
        cells = {}
        for row in result['rows']:
            if row['part'] == 'selection':
                cells.setdefault((row['k'], row['kappa']), []).append(row)
        for group in cells.values():
            for scorer in runner.INDICATORS:
                best = max(row[f'score_{scorer}'] for row in group)
                for row in group:
                    rows.append({**{key: row[key] for key in BASE},
                                 'selector': row['selector'], 'scorer': scorer,
                                 'value': row[f'score_{scorer}'], 'best': best,
                                 # all four score zero: the ratio is missing, not 1
                                 'ratio': row[f'score_{scorer}'] / best if best > 0 else None,
                                 'discriminating': row['b'] > row['k']})
    return rows


def _cells(rows):
    """The distinct pool cells some rows come from."""
    return {tuple(row[key] for key in CELL) for row in rows}


def _matrix(rows, statistic):
    """selector x scorer means of the ratio, per (k, kappa) and over all of them.

    Each scorer carries its own cell count, because a pool on which all four
    selections score zero contributes no ratio to that scorer; ``b_le_k_cells``
    counts the cells where every selection returns every behaviour."""
    out = []
    for k, kappa in sorted({(row['k'], row['kappa']) for row in rows}) + [('all', 'all')]:
        block = rows if k == 'all' else [r for r in rows if (r['k'], r['kappa']) == (k, kappa)]
        for selector in runner.INDICATORS:
            mine = [row for row in block if row['selector'] == selector]
            entry = {'k': k, 'kappa': kappa, 'selector': selector,
                     'b_le_k_cells': len(_cells([r for r in mine if not r['discriminating']]))}
            for scorer in runner.INDICATORS:
                scored = [r for r in mine if r['scorer'] == scorer]
                entry[f'scored_{scorer}'] = statistic(scored, 'ratio')
                entry[f'n_{scorer}'] = len(_cells([r for r in scored if r['ratio'] is not None]))
            out.append(entry)
    return out


def _figure(rows, path):
    """The distribution of tau per indicator pair and k, kappas pooled."""
    fig, ax = rp.figure(size=(6.0, 3.4))
    ks = sorted({row['k'] for row in rows})
    width = 0.8 / max(len(ks), 1)
    for offset, k in enumerate(ks):
        colour = rp.PALETTE[offset % len(rp.PALETTE)]
        samples = {index: [r['tau'] for r in rows
                           if r['pair'] == pair and r['k'] == k and r['tau'] is not None]
                   for index, pair in enumerate(PAIRS)}
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
    ax.axhline(0.0, color='grey', linewidth=0.8)
    ax.set_xticks(range(len(PAIRS)))
    ax.set_xticklabels(PAIRS)
    ax.set_ylabel('Kendall tau-b')
    if ks:
        ax.legend(loc='lower right', fontsize='small')
    return rp.save(fig, path)


def report(cfg, results):
    """Every CSV, table and figure of E2, from the result files alone."""
    out = rp.report_dir(cfg, 'e2')
    usable = [r for r in results if not r.get('error') and not r.get('extra', {}).get('skipped')]
    subsets = [row for r in usable for row in r['rows'] if row['part'] == 'subsets']
    taus, cross = _taus(usable), _cross(usable)
    summary, constancy = _tau_summary(taus), _constancy(subsets)
    main = [row for row in cross if row['discriminating']]
    columns = (['k', 'kappa', 'selector'] + [f'scored_{n}' for n in runner.INDICATORS]
               + [f'n_{n}' for n in runner.INDICATORS])

    written = [
        # The aggregate block answers the headline question of part A: on what
        # share of the pool cells does each indicator vary over the sample?
        rp.write_csv(out / 'e2_random_subsets.csv', subsets + constancy,
                     BASE + ['part', 'indicator', 'n_subsets', 'enumerated', 'n_values',
                             'constant', 'modal_fraction', 'min', 'max', 'mean',
                             'cells', 'constant_fraction', 'mean_modal_fraction']),
        rp.write_csv(out / 'e2_kendall.csv', taus,
                     BASE + ['pair', 'n_subsets', 'enumerated', 'tau', 'p']),
        rp.write_csv(out / 'e2_cross.csv', _matrix(main, rp.pooled), columns),
        rp.write_csv(out / 'e2_cross_macro.csv', _matrix(main, rp.macro), columns),
        rp.write_csv(out / 'e2_cross_all_pools.csv',
                     [{**row, 'aggregate': name} for name, statistic in
                      (('pooled', rp.pooled), ('macro', rp.macro))
                      for row in _matrix(cross, statistic)],
                     columns + ['b_le_k_cells', 'aggregate']),
        rp.table(out / 'tables' / 'e2_kendall.tex', 'tab:e2-kendall',
                 'Kendall $\\tau_b$ between the rankings the dissimilarity-based indicators give '
                 'the random equal-count subsets of one pool: median and interquartile range over '
                 'pools, pooled and as the mean over domains of the per-domain figures, and the '
                 'fraction of pools with a negative $\\tau_b$. "undef." counts the pools where one '
                 'indicator is constant over the sample and $\\tau_b$ is undefined. The '
                 'B-MaxSum/B-MaxMin pair does not depend on $\\kappa$.',
                 ['pair', 'k', 'kappa', 'n', 'undef.', 'median', 'q1', 'q3', 'macro median',
                  'macro q1', 'macro q3', 'negative'],
                 [[row['pair'], row['k'], row['kappa'], row['n'], row['undefined'], row['median'],
                   row['q1'], row['q3'], row['macro_median'], row['macro_q1'], row['macro_q3'],
                   row['negative_fraction']]
                  for row in summary]),
        rp.table(out / 'tables' / 'e2_cross.tex', 'tab:e2-cross',
                 'Each selection rule scored under each indicator, as a fraction of the best of '
                 'the four selections on the same pool, averaged over all $k$ and $\\kappa$. '
                 'Pools with $b \\leq k$ are excluded, since there every rule returns every '
                 'behaviour; the all-pools version is the appendix CSV. ' + rp.TIE_RULE,
                 ['selection', 'aggregate'] + list(runner.INDICATORS),
                 [[row['selector'], name] + [row[f'scored_{s}'] for s in runner.INDICATORS]
                  for name, statistic in (('pooled', rp.pooled), ('macro', rp.macro))
                  for row in _matrix(main, statistic) if row['k'] == 'all']),
        _figure(taus, out / 'figures' / 'e2_tau.pdf'),
    ]

    # The construction check: B-Coverage is k on every drawn subset, so it must
    # be constant on every one of them.
    checked = [row for row in subsets if row['indicator'] == 'bcoverage']
    failing = [row for row in checked if not row['constant']]
    written.append(rp.manifest(cfg, 'e2', written, results, extra={
        'bcoverage_constant_check': 'FAIL' if failing else 'PASS' if checked else 'no subsets drawn',
        'cells_checked': len(checked), 'cells_failing': len(failing),
        # The same check as the aggregate block of e2_random_subsets.csv reports.
        'bcoverage_constant_fraction': {f"k={row['k']} kappa={row['kappa']}":
                                        row['constant_fraction'] for row in constancy
                                        if row['indicator'] == 'bcoverage'},
        'failures': [f"{row['instance']} {row['model']} k={row['k']} kappa={row['kappa']}"
                     for row in failing],
        'subsets_requested': cfg['e2']['subsets'],
        'note': 'on pools with b <= k every selection returns every behaviour, so the main cross '
                'table is restricted to b > k; e2_cross_all_pools.csv is the appendix version and '
                'its b_le_k_cells column counts the cells the restriction drops',
    }))
    return written
