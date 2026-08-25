"""Experiment B: what each rule selects.

How often do the five rules disagree, and what does the plan-level baseline
miss? One pool per task, k = 20, all five selectors picking from that identical
pool, then every returned set scored under every indicator *and* under
stability-MaxSum -- both directions, because a table showing only that the
baseline scores badly under our own indicator is circular and a reviewer will
say so.

The column that carries the section is ``hidden_pref_rate``. Before any
selection, and from a seed independent of everything else, a dimension f* and a
value v* that some plan in the pool exhibits are drawn and shown to no
selector; the question is then whether each selector's returned set contains a
plan with f* = v*. It is the only measurement here that the behaviour-space
indicators cannot rig, because no selector sees the preference. Everything else
shows the selections are *different*; this is the only thing that speaks to
*better*, and it is reported whichever way it comes out.

Two things will look like bugs and are not. B-MaxMin returns far fewer plans
than the k asked for -- thm:bmaxmin-degenerate appearing in the data -- so it
is read as a certificate over the other selections rather than as a selector in
its own right. And B-Novelty nearly duplicates B-MaxSum whenever b <= k_nn,
because the clamp reduces it to the mean pairwise distance there; ``b_pool`` is
in every row so that is visible rather than looking like a copy-paste error.
"""

import os
import random
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import harness
from paperexps.baseline import Stability, greedy_maxsum_stability

SCORE_FIELDS = [
    'task_id', 'domain', 'k', 'dimensions', 'n_dims', 'k_nn', 'selector', 'seed',
    'b_coverage', 'b_maxsum', 'b_maxmin', 'b_novelty', 'maxsum_stability',
    'redundancy', 'b_achieved', 'spread_go', 'spread_ru', 'spread_cb',
    'hidden_pref_hits', 'hidden_pref_draws', 'hidden_pref_rate',
    # Beyond the specified shape, and needed to read it honestly:
    # The exact probability that a draw from the same distribution is satisfied,
    # in closed form. Fifty draws carry a standard error near 0.07, which is
    # wider than any difference between the rules, so the sampled rate alone
    # cannot say whether two rules differ. This removes that noise entirely.
    'hidden_pref_exact', 'coverage_by_dimension',
    'n_selected',   # B-MaxMin returns fewer than k, so a shortfall must not
                    # be mistaken for redundancy, which is defined against k
    # A pool holding b < k behaviours forces k - b of the redundancy on every
    # selector alive, so the raw number cannot be read as the selector's fault
    # without these two beside it.
    'redundancy_floor', 'redundancy_excess',
    'n_pool', 'b_pool', 'year', 'q', 'multiset_stability',
]

OVERLAP_FIELDS = [
    'task_id', 'domain', 'k', 'selector_i', 'selector_j',
    'jaccard',                  # over the distinct BEHAVIOUR sets -- the headline
    'overlap_jaccard_plans',    # over the plan sets, for the gap between the two
    'seed',
]


def select_all(pool, k, k_nn, stability):
    """The five conditions, every one of them picking from the same pool."""
    plans = pool.plans
    selections = {}
    for indicator in ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty'):
        selections[indicator] = pool.counter.extract(
            plans, k=k, indicator=indicator, k_nn=k_nn)
    selections['maxsum_stability'] = greedy_maxsum_stability(plans, k, stability)
    return selections


def hidden_preferences(pool, draws, seed):
    """``draws`` independent (f*, v*) pairs, drawn before any selection.

    A dimension uniformly, then a value uniformly among those that dimension
    actually takes somewhere in the pool -- so the preference is always
    satisfiable, and rare values are as likely to be asked for as common ones.
    Weighting by how often the planner happened to produce a value would be
    asking an easier question than a user with a preference asks.
    """
    rng = random.Random(seed)
    values = {name: sorted({harness.dimension_value(behaviour, name)
                            for behaviour in pool.distinct})
              for name in pool.dimensions}
    usable = [name for name in pool.dimensions
              if values[name] and values[name] != [None]]
    if not usable:
        return []
    return [(name, rng.choice(values[name]))
            for name in (rng.choice(usable) for _ in range(draws))]


def hidden_preference_hits(selected, preferences):
    """How many of the drawn preferences the selection satisfies."""
    profiles = [plan.behaviour for plan in selected]
    hits = 0
    for name, value in preferences:
        if any(harness.dimension_value(behaviour, name) == value
               for behaviour in profiles):
            hits += 1
    return hits


def pool_dimension_values(pool):
    """{dimension -> the values it takes anywhere in the pool}."""
    return {name: {harness.dimension_value(behaviour, name)
                   for behaviour in pool.distinct}
            for name in pool.dimensions}


def hidden_preference_exact(selected, pool_values):
    """The exact probability that a draw satisfies this selection.

    The draw is a dimension uniformly, then one of that dimension's attainable
    values uniformly, so the probability is the mean over dimensions of the
    fraction of that dimension's values the selection exhibits. Reported beside
    the sampled rate because fifty draws cannot resolve the differences here --
    and because it makes plain what the sampled rate is really measuring: a
    dimension that takes one value across the whole pool contributes 1.0 to
    every selection alive, which on these benchmarks is most of the ru column.
    """
    usable = [name for name, values in pool_values.items() if values and values != {None}]
    if not usable:
        return '', ''
    behaviours = [plan.behaviour for plan in selected]
    total, parts = 0.0, []
    for name in usable:
        covered = {harness.dimension_value(behaviour, name) for behaviour in behaviours}
        covered &= pool_values[name]
        total += len(covered) / len(pool_values[name])
        parts.append(f'{name}:{len(covered)}/{len(pool_values[name])}')
    return total / len(usable), ','.join(parts)


def run_pool(pool, args):
    """The score rows and overlap rows for one task."""
    k, k_nn = args.k, args.k_nn
    counter = pool.counter
    stability = Stability(pool.plans, multiset=not args.set_stability)

    seed = harness.derive_seed(args.seed, pool.task_id, 'select')
    hidden_seed = harness.derive_seed(args.seed, pool.task_id, 'hidden-preference')
    preferences = hidden_preferences(pool, args.hidden_draws, hidden_seed)
    pool_values = pool_dimension_values(pool)

    selections = select_all(pool, k, k_nn, stability)

    score_rows, overlap_rows = [], []
    behaviour_sets, plan_sets = {}, {}
    for name, selected in selections.items():
        behaviours = [plan.behaviour for plan in selected]
        behaviour_sets[name] = set(behaviours)
        plan_sets[name] = {id(plan) for plan in selected}
        hits = hidden_preference_hits(selected, preferences)
        exact, coverage = hidden_preference_exact(selected, pool_values)
        row = {
            'task_id': pool.task_id, 'domain': pool.domain, 'k': k,
            'dimensions': '+'.join(pool.dimensions), 'n_dims': len(pool.dimensions),
            'k_nn': k_nn, 'selector': name, 'seed': seed,
            'b_coverage': counter.b_coverage(selected),
            'b_maxsum': counter.b_maxsum(selected),
            'b_maxmin': counter.b_maxmin(selected),
            'b_novelty': counter.b_novelty(selected, k_nn=k_nn),
            'maxsum_stability': stability.maxsum(selected),
            'redundancy': k - counter.b_coverage(selected),
            'b_achieved': counter.b_coverage(selected),
            'hidden_pref_hits': hits,
            'hidden_pref_draws': len(preferences),
            'hidden_pref_rate': (hits / len(preferences)) if preferences else '',
            'hidden_pref_exact': exact,
            'coverage_by_dimension': coverage,
            'n_selected': len(selected),
            'redundancy_floor': k - min(k, pool.b),
            'redundancy_excess': (k - counter.b_coverage(selected)) - (k - min(k, pool.b)),
            'n_pool': pool.n, 'b_pool': pool.b,
            'year': pool.year, 'q': pool.q,
            'multiset_stability': not args.set_stability,
        }
        for dimension in harness.SPREAD_DIMENSIONS:
            row[f'spread_{dimension}'] = harness.dimension_spread(
                counter, behaviours, dimension)
        score_rows.append(row)

    for i, name_i in enumerate(harness.SELECTORS):
        for name_j in harness.SELECTORS:
            overlap_rows.append({
                'task_id': pool.task_id, 'domain': pool.domain, 'k': k,
                'selector_i': name_i, 'selector_j': name_j,
                'jaccard': harness.jaccard(behaviour_sets[name_i],
                                           behaviour_sets[name_j]),
                'overlap_jaccard_plans': harness.jaccard(plan_sets[name_i],
                                                         plan_sets[name_j]),
                'seed': seed,
            })
    return score_rows, overlap_rows


def main(argv=None):
    parser = harness.base_parser(__doc__.splitlines()[0])
    parser.add_argument('--k', type=int, default=20, help='Selection size.')
    parser.add_argument('--hidden-draws', type=int, default=50,
                        help='Independent (f*, v*) draws per task.')
    parser.add_argument('--overlap-out',
                        help='Where to write the overlap CSV '
                             '(default: --out with _scores replaced by _overlap).')
    parser.add_argument('--set-stability', action='store_true',
                        help='Score the baseline with A(p) as a set rather than a '
                             'multiset -- the reading plandiversity uses.')
    args = parser.parse_args(argv)
    if args.dry_run:
        args.hidden_draws = min(args.hidden_draws, 10)

    score_rows, overlap_rows, skipped = [], [], []
    for path in harness.pools_from(args):
        with tempfile.TemporaryDirectory(prefix='paperexps-') as workdir:
            try:
                pool = harness.load_pool(path, args, workdir)
            except harness.PoolSkipped as reason:
                skipped.append((harness.task_id_of(path), str(reason)))
                print(f'{harness.task_id_of(path)}: skipped -- {reason}')
                continue
            scores, overlaps = run_pool(pool, args)
            score_rows += scores
            overlap_rows += overlaps
            baseline = next(r for r in scores if r['selector'] == 'maxsum_stability')
            print(f"{pool.task_id}: n={pool.n} b={pool.b} "
                  f"baseline redundancy={baseline['redundancy']} "
                  f"hidden_pref_rate={baseline['hidden_pref_rate']}")

    harness.write_csv(args.out, SCORE_FIELDS, score_rows)
    overlap_out = args.overlap_out or (
        args.out.replace('_scores.csv', '_overlap.csv')
        if '_scores.csv' in args.out else args.out.replace('.csv', '_overlap.csv'))
    harness.write_csv(overlap_out, OVERLAP_FIELDS, overlap_rows)
    print(f'wrote {len(score_rows)} score rows to {args.out}')
    print(f'wrote {len(overlap_rows)} overlap rows to {overlap_out}')
    if skipped:
        print(f'skipped {len(skipped)} pool(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
