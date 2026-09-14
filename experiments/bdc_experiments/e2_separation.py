"""E2 -- whether the three dissimilarity-based indicators separate plan sets
that B-Coverage scores alike.

Part A is selection-free: random subsets of exactly k of a pool's behaviours,
drawn here from the behaviour dump and scored off its matrix, so B-Coverage is
constant over the sample by construction. Part B reads the four selections at
each k off the recorded runs and scores each returned set under all four
indicators. Part C repeats part B on rovers under the astronaut's model at
each further weight setting, to see whether the weights change the selected
sets or only their values.
"""

import math
import random
import statistics
from collections import Counter
from itertools import combinations

from bdc_experiments import models, runner
from bdc_experiments import report as rp

PAIRS = ('bmaxsum/bmaxmin', 'bmaxsum/bnovelty', 'bmaxmin/bnovelty')
BASE = list(runner.BASE_FIELDS)

#: Two subset scores count as one value within this tolerance: they are sums of
#: the same rationals in a different order.
DIGITS = 12


def _draw(cfg, result, k):
    """``(subsets, enumerated)``: up to ``[e2].random_subsets`` distinct k-subsets
    of the b behaviours, or all C(b, k) when there are no more than that.
    Seeded from the run seed and the pool, model and k, so a rerun redraws them."""
    b, wanted = len(result['dump']['distinct']), cfg['e2']['random_subsets']
    if math.comb(b, k) <= wanted:
        return [list(s) for s in combinations(range(b), k)], True
    pool = result['pool']
    rng = random.Random(repr((cfg['run']['seed'], pool['instance'], pool['pool_stem'],
                              result['model']['name'], k)))
    drawn = {}
    while len(drawn) < wanted:
        drawn.setdefault(tuple(sorted(rng.sample(range(b), k))), None)
    return [list(s) for s in drawn], False


def _subsets(cfg, results):
    """Part A: per-subset scores, the per-(k, kappa, indicator) spread, and the taus."""
    kappas, drawn, rows, taus = cfg['selection']['kappa_values'], [], [], []
    for result in results:
        for k in cfg['selection']['k_values']:
            if len(result['dump']['distinct']) <= k:      # the only k-subset is every behaviour
                continue
            subsets, enumerated = _draw(cfg, result, k)
            scores = {kappa: [rp.score(result['dump'], s, kappa) for s in subsets] for kappa in kappas}
            drawn += [{**rp.head(result, k), 'subset': ' '.join(map(str, s)),
                       **{n: scores[kappas[0]][i][n] for n in runner.INDICATORS[:3]},
                       **{f'bnovelty_kappa{kappa}': scores[kappa][i]['bnovelty'] for kappa in kappas}}
                      for i, s in enumerate(subsets)]
            for kappa in kappas:
                for indicator in runner.INDICATORS:
                    values = [s[indicator] for s in scores[kappa]]
                    counts = Counter(round(v, DIGITS) for v in values)
                    rows.append({**rp.head(result, k, kappa), 'indicator': indicator,
                                 'n_subsets': len(values), 'enumerated': enumerated,
                                 'n_values': len(counts), 'constant': len(counts) == 1,
                                 'modal_fraction': max(counts.values()) / len(values),
                                 'min': min(values), 'max': max(values),
                                 'mean': statistics.fmean(values)})
                for pair in PAIRS:
                    first, second = pair.split('/')
                    tau, p = rp.kendall([s[first] for s in scores[kappa]],
                                        [s[second] for s in scores[kappa]])
                    taus.append({**rp.head(result, k, kappa), 'pair': pair,
                                 'n_subsets': len(subsets), 'enumerated': enumerated,
                                 'tau': tau, 'p': None if enumerated else p})
    return drawn, rows, taus


def _constancy(rows):
    """Per (k, kappa, indicator): on what fraction of the pools the indicator
    took a single value over the whole sample. B-Coverage is the check."""
    return [{'k': k, 'kappa': kappa, 'indicator': indicator, 'cells': len(cell),
             'constant_fraction': sum(r['constant'] for r in cell) / len(cell),
             'mean_modal_fraction': statistics.fmean(r['modal_fraction'] for r in cell)}
            for (k, kappa, indicator), cell in rp.group(rows, ('k', 'kappa', 'indicator')).items()]


def _tau_summary(taus):
    """Median and IQR pooled and macro, and the fraction of pools with a
    negative tau, per (pair, k, kappa). A tau is missing where an indicator
    was constant over the sample and the correlation is undefined."""
    out = []
    for (pair, k, kappa), cell in sorted(rp.group(taus, ('pair', 'k', 'kappa')).items()):
        known = [r for r in cell if r['tau'] is not None]
        spread = rp.summarise([r['tau'] for r in known])
        per_domain = [rp.summarise([r['tau'] for r in members])
                      for _, members in rp.group(known, ('domain',)).items()]
        out.append({'pair': pair, 'k': k, 'kappa': kappa, 'n': spread['n'],
                    'undefined': len(cell) - len(known), 'median': spread['median'],
                    'q1': spread['q1'], 'q3': spread['q3'],
                    **{f'macro_{s}': (statistics.fmean(d[s] for d in per_domain) if per_domain else None)
                       for s in ('median', 'q1', 'q3')},
                    'negative_fraction': (sum(r['tau'] < 0 for r in known) / len(known)
                                          if known else None)})
    return out


def _cross(cfg, results):
    """Part B: each selection's value under each indicator, over the best of
    the four selections on the same pool, k and kappa."""
    rows = []
    for result in results:
        for k in cfg['selection']['k_values']:
            for kappa in cfg['selection']['kappa_values']:
                scored = {s: rp.score(result['dump'], rp.entry(result, s, kappa)['distinct'][:k], kappa)
                          for s in runner.INDICATORS}
                for selector, values in scored.items():
                    for scorer in runner.INDICATORS:
                        best = max(v[scorer] for v in scored.values())
                        rows.append({**rp.head(result, k, kappa), 'selector': selector,
                                     'scorer': scorer, 'value': values[scorer], 'best': best,
                                     'ratio': values[scorer] / best if best > 0 else None,
                                     'discriminating': len(result['dump']['distinct']) > k})
    return rows


def _matrix(rows, statistic, pools):
    """selector x scorer means of the ratio, per (k, kappa) and over all of them."""
    out = []
    for k, kappa in sorted({(r['k'], r['kappa']) for r in rows}) + [('all', 'all')]:
        block = rows if k == 'all' else [r for r in rows if (r['k'], r['kappa']) == (k, kappa)]
        for selector in runner.INDICATORS:
            mine = [r for r in block if r['selector'] == selector]
            entry = {'k': k, 'kappa': kappa, 'selector': selector, 'pools': pools}
            for scorer in runner.INDICATORS:
                scored = [r for r in mine if r['scorer'] == scorer]
                entry[f'scored_{scorer}'] = statistic(scored, 'ratio')
                entry[f'n_{scorer}'] = sum(1 for r in scored if r['ratio'] is not None)
            out.append(entry)
    return out


def _weights(cfg, base_name):
    """Part C: the astronaut's model at each further weight setting against
    the declared one, on the same pools: is the returned set the same?"""
    variants = {v.name: v for v in models.weight_variants(cfg)}
    grouped = rp.by_pool(rp.selections(cfg, keep=lambda name: name == base_name or name in variants))
    rows = []
    for found in grouped.values():
        if base_name not in found:
            continue
        base = found[base_name]
        for name, other in found.items():
            if name == base_name:
                continue
            for k in cfg['selection']['k_values']:
                for kappa in cfg['selection']['kappa_values']:
                    for indicator in runner.INDICATORS:
                        mine = rp.entry(base, indicator, kappa)
                        theirs = rp.entry(other, indicator, kappa)
                        a = {tuple(t) for t in mine['behaviours'][:k]}
                        b = {tuple(t) for t in theirs['behaviours'][:k]}
                        rows.append({**rp.head(base, k, kappa), 'setting': name,
                                     'weights': ' '.join(str(f['weight']) for f in other['model']['features']),
                                     'indicator': indicator, 'same_set': a == b,
                                     'jaccard': len(a & b) / len(a | b),
                                     'value_declared': rp.score(base['dump'], mine['distinct'][:k], kappa)[indicator],
                                     'value_setting': rp.score(other['dump'], theirs['distinct'][:k], kappa)[indicator]})
    return rows


def _figure(taus, path):
    """The distribution of tau per indicator pair and k, kappas pooled."""
    fig, ax = rp.figure(size=(6.0, 3.4))
    ks = sorted({r['k'] for r in taus})
    rp.boxes(ax, PAIRS, ks, lambda pair, k: [r['tau'] for r in taus
                                              if r['pair'] == pair and r['k'] == k and r['tau'] is not None])
    ax.axhline(0.0, color='grey', linewidth=0.8)
    ax.set_ylabel('Kendall tau-b')
    if ks:
        ax.legend(title='k', loc='lower right', fontsize='small')
    return rp.save(fig, path)


def report(cfg):
    results = rp.selections(cfg, keep=models.is_primary)
    drawn, subsets, taus = _subsets(cfg, results)
    constancy, tau_summary = _constancy(subsets), _tau_summary(taus)
    cross = _cross(cfg, results)
    main = [r for r in cross if r['discriminating']]
    weights = _weights(cfg, models.PER_DOMAIN[0].name)
    weight_summary = [{'setting': s, 'k': k, 'kappa': kappa, 'indicator': i, 'pools': len(cell),
                       'same_set_fraction': rp.pooled(cell, 'same_set'),
                       'jaccard_mean': rp.pooled(cell, 'jaccard')}
                      for (s, k, kappa, i), cell in rp.group(weights, ('setting', 'k', 'kappa', 'indicator')).items()]
    out = rp.report_dir(cfg, 'e2')
    columns = (['k', 'kappa', 'selector', 'pools'] + [f'scored_{n}' for n in runner.INDICATORS]
               + [f'n_{n}' for n in runner.INDICATORS])
    written = [
        rp.write_csv(out / 'e2_random_subsets.csv', subsets + constancy,
                     BASE + ['indicator', 'n_subsets', 'enumerated', 'n_values', 'constant',
                             'modal_fraction', 'min', 'max', 'mean', 'cells', 'constant_fraction',
                             'mean_modal_fraction']),
        rp.write_csv(out / 'e2_subsets.csv', drawn),
        rp.write_csv(out / 'e2_kendall.csv', taus, BASE + ['pair', 'n_subsets', 'enumerated', 'tau', 'p']),
        rp.write_csv(out / 'e2_cross.csv', _matrix(main, rp.pooled, 'b>k') + _matrix(cross, rp.pooled, 'all'), columns),
        rp.write_csv(out / 'e2_cross_macro.csv', _matrix(main, rp.macro, 'b>k') + _matrix(cross, rp.macro, 'all'), columns),
        rp.write_csv(out / 'e2_weights.csv', weights + weight_summary,
                     BASE + ['setting', 'weights', 'indicator', 'same_set', 'jaccard', 'value_declared',
                             'value_setting', 'pools', 'same_set_fraction', 'jaccard_mean']),
        rp.table(out / 'tables' / 'e2_kendall.tex', 'tab:e2-kendall',
                 'Kendall $\\tau_b$ between the rankings the dissimilarity-based indicators give '
                 'the random equal-count subsets of one pool: median and interquartile range over '
                 'pools, pooled and as the mean over domains of the per-domain figures, and the '
                 'fraction of pools with a negative $\\tau_b$. "undef." counts the pools where one '
                 'indicator is constant over the sample. The B-MaxSum/B-MaxMin pair does not '
                 'depend on $\\kappa$.',
                 ['pair', 'k', 'kappa', 'n', 'undef.', 'median', 'q1', 'q3', 'macro median',
                  'macro q1', 'macro q3', 'negative'],
                 [[r['pair'], r['k'], r['kappa'], r['n'], r['undefined'], r['median'], r['q1'],
                   r['q3'], r['macro_median'], r['macro_q1'], r['macro_q3'], r['negative_fraction']]
                  for r in tau_summary]),
        rp.table(out / 'tables' / 'e2_cross.tex', 'tab:e2-cross',
                 'Each selection rule scored under each indicator, as a fraction of the best of '
                 'the four selections on the same pool, averaged over all $k$ and $\\kappa$, on '
                 'the pools with $b > k$. ' + rp.TIE_RULE,
                 ['selection', 'aggregate'] + list(runner.INDICATORS),
                 [[r['selector'], name] + [r[f'scored_{s}'] for s in runner.INDICATORS]
                  for name, statistic in (('pooled', rp.pooled), ('macro', rp.macro))
                  for r in _matrix(main, statistic, 'b>k') if r['k'] == 'all']),
        rp.table(out / 'tables' / 'e2_weights.tex', 'tab:e2-weights',
                 "The astronaut's model at each further weight setting against the declared one, "
                 'on the rovers pools: the fraction of (pool, $k$, $\\kappa$) cells on which the '
                 'selection returns the same behaviour set, and the mean Jaccard similarity of the '
                 'two sets. B-Coverage reads no weights, so its rows are a check.',
                 ['setting', 'indicator', 'cells', 'same set', 'Jaccard'],
                 [[r['setting'], r['indicator'], len(cell), rp.pooled(cell, 'same_set'),
                   rp.pooled(cell, 'jaccard')]
                  for (setting, indicator), cell in rp.group(weights, ('setting', 'indicator')).items()
                  for r in cell[:1]]),
        _figure(taus, out / 'figures' / 'e2_tau.pdf'),
    ]
    checked = [r for r in subsets if r['indicator'] == 'bcoverage']
    failing = [r for r in checked if not r['constant']]
    written.append(rp.manifest(cfg, 'e2', written, results, extra={
        'bcoverage_constant_check': 'FAIL' if failing else 'PASS' if checked else 'no subsets drawn',
        'cells_checked': len(checked), 'cells_failing': len(failing),
        'bcoverage_selects_same_set_under_every_weight': (
            rp.pooled([r for r in weights if r['indicator'] == 'bcoverage'], 'same_set')),
        'subsets_requested': cfg['e2']['random_subsets'],
        'seeding': 'random.Random(repr((run seed, instance, pool stem, model, k)))',
        'weight_settings': cfg['e2']['weight_settings'],
        'note': 'on pools with b <= k every selection returns every behaviour, so the table is '
                'over b > k; the all-pools matrices are the rows marked pools = all'}))
    return written
