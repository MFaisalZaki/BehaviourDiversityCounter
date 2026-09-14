"""Phase 0: an independent audit of the library against the paper.

Every indicator and phase-two rule the experiments lean on is checked against
``bdc_experiments.reference``, written straight from the definitions and
sharing no code with the library. Spaces are drawn at random and seeded from
the case number, so a failure names a seed that reproduces it. No planner is
needed anywhere: the stub counter of ``conftest.py`` hands the library
behaviour strings and costs, which is all the indicators read, and the one
test that leaves the stub spaces behind runs the same comparison over the four
committed smoke pools and their real feature models.
"""

import math
import os
import random
import re
import subprocess
import sys
from itertools import combinations

import pytest

from behaviour_diversity_counter.behaviour_diversity_counter import TIE_DECIMALS
from behaviour_diversity_counter import BehaviourDiversityCounter
from bdc_experiments import models, pools, runner
from bdc_experiments.config import load
from bdc_experiments.reference import (
    ref_bcoverage, ref_bmaxmin, ref_bmaxsum, ref_bnovelty, ref_distinct, ref_extract,
    ref_indicator, ref_stability)

TOL = 1e-9
KAPPAS = (1, 2, 3, 5)
INDICATORS = ('bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty')
#: Scores this close are tied as far as the library's ``best_index`` is
#: concerned; the reference compares raw floats.
TIE_TOLERANCE = 0.5 * 10 ** -TIE_DECIMALS


def pool_for(space, seed, low=0, high=12):
    """A cost-sorted pool over the space, and a counter holding it.

    Phase two receives the pool cost-sorted, which is what makes the paper's
    "arbitrary" tie-break the deterministic lowest-index one on both sides.
    """
    rng = random.Random(seed)
    n = rng.randint(low, high)
    distinct = n if seed % 3 == 0 else (rng.randint(1, n) if n else 1)
    counter = space.counter()
    plans = counter.make_plans(space.draw(rng, n, distinct))
    plans.sort(key=lambda plan: plan.cost)      # stable: ties keep pool order
    return counter, plans


def triples(plans):
    """The reference's plan triples ``(index, cost, behaviour)``."""
    return [(i, plan.cost, plan.values) for i, plan in enumerate(plans)]


def positions(plans, chosen):
    """Where a returned selection sits in the pool."""
    at = {id(plan): i for i, plan in enumerate(plans)}
    return [at[id(plan)] for plan in chosen]


def library_value(counter, plans, indicator, kappa):
    """The library's own reading of an indicator on a set of plans."""
    if indicator == 'bnovelty':
        return counter.b_novelty(plans, k_nn=kappa)
    return float({'bcoverage': counter.b_coverage, 'bmaxsum': counter.b_maxsum,
                  'bmaxmin': counter.b_maxmin}[indicator](plans))


def step_score(indicator, candidate, held, d, kappa):
    """The paper's greedy score of one candidate behaviour against the held
    behaviours; None when the candidate repeats one of them."""
    if candidate in held:
        return None
    if indicator == 'bmaxsum':
        return math.fsum(d(candidate, h) for h in held)
    if indicator == 'bmaxmin':
        return min(d(candidate, h) for h in held)
    return ref_bnovelty(list(held) + [candidate], d, kappa)


def shortfall(indicator, chosen, trip, d, kappa):
    """How much score a selection gave away, at its worst step.

    0.0 says every pick -- the opening pair included -- attained the maximum of
    its step, so a difference from the reference is a tie-break and nothing
    else. A positive value is what the disagreement actually cost.
    """
    if len(chosen) < 2 or indicator == 'bcoverage':
        return 0.0
    n = len(trip)
    best_pair = max(d(trip[i][2], trip[j][2]) for i, j in combinations(range(n), 2))
    worst = best_pair - d(trip[chosen[0]][2], trip[chosen[1]][2])
    held = [trip[chosen[0]][2], trip[chosen[1]][2]]
    for step in range(2, len(chosen)):
        scored = {pos: step_score(indicator, trip[pos][2], held, d, kappa)
                  for pos in range(n) if pos not in chosen[:step]}
        scored = {p: v for p, v in scored.items() if v is not None}
        if chosen[step] in scored:
            worst = max(worst, max(scored.values()) - scored[chosen[step]])
        held.append(trip[chosen[step]][2])
    return worst


def compare_extraction(counter, plans, space, k, indicator, kappa):
    """One extraction, library against reference: None when they agree, else
    ``(kind, lost, report)`` -- 'value' or 'set', and the score given away."""
    trip = triples(plans)
    chosen = positions(plans, counter.extract(plans, k, indicator=indicator, k_nn=kappa))
    wanted = ref_extract(indicator, trip, space.d, k, kappa)
    got_b = [trip[i][2] for i in chosen]
    want_b = [trip[i][2] for i in wanted]
    got_v = ref_indicator(indicator, got_b, space.d, kappa)
    want_v = ref_indicator(indicator, want_b, space.d, kappa)
    if abs(got_v - want_v) <= TOL and set(got_b) == set(want_b):
        return None
    lost = shortfall(indicator, chosen, trip, space.d, kappa)
    verdict = ('every library pick attained its maximum: an arbitrary tie, both sides right'
               if lost <= TOL else
               f'the library gave away {lost:.3e} at a step, within its own tie tolerance '
               f'{TIE_TOLERANCE:g}: TIE_DECIMALS={TIE_DECIMALS} rounding, reference right'
               if lost <= TIE_TOLERANCE + TOL else
               f'the library gave away {lost:.3e}, beyond any tie tolerance: A DEFECT')
    report = (f'{space} k={k} kappa={kappa} {indicator}\n'
              f'    library   {got_b} -> {got_v!r}\n'
              f'    reference {want_b} -> {want_v!r}\n'
              f'    {verdict}')
    return ('value' if abs(got_v - want_v) > TOL else 'set'), lost, report


def test_indicators_match_reference(space):
    """Every indicator, every kappa, 500 random spaces and pools."""
    for case in range(500):
        sp = space(case)
        counter, plans = pool_for(sp, case)
        behaviours = [plan.values for plan in plans]
        why = f'seed={case} {sp} |C|={len(plans)}'

        assert counter.b_coverage(plans) == ref_bcoverage(behaviours), why
        assert counter.b_maxsum(plans) == pytest.approx(
            ref_bmaxsum(behaviours, sp.d), abs=TOL), why
        assert counter.b_maxmin(plans) == pytest.approx(
            ref_bmaxmin(behaviours, sp.d), abs=TOL), why
        for kappa in KAPPAS:
            assert counter.b_novelty(plans, k_nn=kappa) == pytest.approx(
                ref_bnovelty(behaviours, sp.d, kappa), abs=TOL), f'{why} kappa={kappa}'


def test_extraction_matches_reference(space):
    """The same behaviours out, and the same value.

    The two are independent findings, so both are collected over the whole
    sweep and reported together: a value fault must not hide a set fault.
    """
    value_faults, set_faults = [], []
    for case in range(120):
        sp = space(case)
        counter, plans = pool_for(sp, 20000 + case, low=1, high=10)
        kappa = KAPPAS[case % len(KAPPAS)]
        for k in range(1, 7):
            for indicator in INDICATORS:
                verdict = compare_extraction(counter, plans, sp, k, indicator, kappa)
                if verdict is None:
                    continue
                kind, _lost, report = verdict
                (value_faults if kind == 'value' else set_faults).append(
                    f'seed={case} {report}')

    report = []
    if value_faults:
        report.append(
            f'{len(value_faults)} of {120 * 6 * len(INDICATORS)} extractions disagree in '
            'VALUE. The last line of each names the cause: an arbitrary tie leaves neither '
            'side wrong, a shortfall inside the tie tolerance is the library rounding the '
            'greedy score, more is a defect.\n' + '\n'.join(value_faults[:6]))
    if set_faults:
        report.append(
            f'{len(set_faults)} extractions score the same over different behaviours: a '
            'tie-break difference, documentable rather than wrong.\n'
            + '\n'.join(set_faults[:6]))
    if report:
        pytest.fail('\n\n'.join(report), pytrace=False)


def test_extraction_disagreements_are_tie_breaks_only(space):
    """A wider sweep over pools big enough for near-ties to be common, to
    characterise the disagreements rather than merely count them.

    Every one of them must be a step at which the library gave away no more
    than its own tie tolerance -- a tie-break, however far the two selections
    then drift apart. More would be the greedy rule itself being wrong.
    """
    given_away = []
    for case in range(300):
        sp = space(case)
        counter, plans = pool_for(sp, 70000 + case, low=6, high=16)
        kappa = KAPPAS[case % len(KAPPAS)]
        for k in (2, 4, 6, 8):
            for indicator in INDICATORS[1:]:
                verdict = compare_extraction(counter, plans, sp, k, indicator, kappa)
                if verdict is None:
                    continue
                _kind, lost, report = verdict
                assert lost <= TIE_TOLERANCE + TOL, (
                    f'seed={case} gave away {lost:.3e}, past the tie tolerance '
                    f'{TIE_TOLERANCE:g}: a defective greedy step.\n{report}')
                if lost > TOL:
                    given_away.append(f'seed={case} {report}')
    print(f'\n{len(given_away)} disagreement(s) where TIE_DECIMALS={TIE_DECIMALS} '
          f'rounding tied genuinely unequal scores; first:\n'
          + (given_away[0] if given_away else '  (none)'))


def test_golden_examples_still_pass():
    """tests/test_golden.py holds the numbers the paper itself states."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    done = subprocess.run(
        [sys.executable, '-m', 'pytest', os.path.join(root, 'tests', 'test_golden.py'),
         '-q', '-p', 'no:cacheprovider'],
        capture_output=True, text=True, cwd=root)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]


def test_twinning(space):
    """A plan repeating a behaviour already held moves no indicator at all."""
    for case in range(200):
        sp = space(case)
        counter, plans = pool_for(sp, 40000 + case, low=2, high=8)
        readings = [library_value(counter, plans, name, kappa)
                    for name in INDICATORS for kappa in KAPPAS]
        twin = counter.make_plans([(plans[case % len(plans)].values, 99)])
        after = [library_value(counter, plans + twin, name, kappa)
                 for name in INDICATORS for kappa in KAPPAS]

        assert after == readings, f'seed={case} {sp}'


def test_monotonicity_in_plans(space):
    """B-Coverage and B-MaxSum never fall; the other two do, and here is where."""
    maxmin_fall = novelty_fall = None
    for case in range(200):
        sp = space(case)
        counter, plans = pool_for(sp, 50000 + case, low=2, high=10)
        kappa = KAPPAS[case % len(KAPPAS)]       # every kappa of the sweep, not just 3
        held = []
        for plan in plans:
            before = [library_value(counter, held, name, kappa) for name in INDICATORS]
            held = held + [plan]
            after = [library_value(counter, held, name, kappa) for name in INDICATORS]
            why = (f'seed={case} {sp} kappa={kappa} adding {plan.values} '
                   f'to {[p.values for p in held[:-1]]}')
            assert after[0] >= before[0], f'B-Coverage fell: {why}'
            assert after[1] >= before[1] - TOL, f'B-MaxSum fell: {why}'
            if maxmin_fall is None and after[2] < before[2] - TOL:
                maxmin_fall = f'{why}: B-MaxMin {before[2]!r} -> {after[2]!r}'
            if novelty_fall is None and after[3] < before[3] - TOL:
                novelty_fall = f'{why}: B-Novelty {before[3]!r} -> {after[3]!r}'

    assert maxmin_fall, 'no B-MaxMin counterexample in the sample'
    assert novelty_fall, 'no B-Novelty counterexample in the sample'
    print(f'\nB-MaxMin is not monotone in the plan set: {maxmin_fall}'
          f'\nB-Novelty is not monotone in the plan set: {novelty_fall}')


def test_extraction_returns_k_plans(space):
    """min(k, |C|) plans come back, for every rule.

    Two things can shorten a selection: an indicator that fell on the way, and
    a pool with fewer behaviours than plans asked for. The second is the one
    the padding exists for, so the sweep is required to exercise it -- for each
    rule -- rather than merely to survive it. An empty pool and a k of zero or
    less are in the sweep too: they return nothing rather than raising.
    """
    fell = {'bmaxmin': None, 'bnovelty': None}
    padded = dict.fromkeys(INDICATORS, 0)
    for case in range(200):
        sp = space(case)
        counter, plans = pool_for(sp, 60000 + case, low=0, high=10)
        behaviours = {plan.values for plan in plans}
        for k in (-2, 0, 1, 3, 5, 7):
            for indicator in INDICATORS:
                kappa = KAPPAS[(case + k) % len(KAPPAS)]
                chosen = counter.extract(plans, k, indicator=indicator, k_nn=kappa)
                assert len(chosen) == max(0, min(k, len(plans))), (
                    f'seed={case} {sp} k={k} {indicator} kappa={kappa}: '
                    f'{len(chosen)} plans came back, not min(k, |C|)')
                if min(k, len(plans)) > len(behaviours):
                    # A duplicate behaviour had to be taken to reach k.
                    padded[indicator] += 1
                    assert counter.b_coverage(chosen) == len(behaviours)
                if indicator not in fell:
                    continue
                trail = [library_value(counter, chosen[:j], indicator, kappa)
                         for j in range(2, len(chosen) + 1)]
                if fell[indicator] is None and any(
                        nxt < cur - TOL for cur, nxt in zip(trail, trail[1:])):
                    fell[indicator] = (f'seed={case} {sp} k={k} {indicator} '
                                       f'kappa={kappa}: {[round(v, 6) for v in trail]}, '
                                       f'{len(chosen)} plans returned')

    assert fell['bmaxmin'] and fell['bnovelty'], (
        f'no falling selection found for {[n for n, v in fell.items() if v is None]}')
    assert all(padded.values()), f'the sweep never needed the padding: {padded}'
    print('\nselections whose indicator fell and which still returned min(k, |C|) plans:'
          f'\n  {fell["bmaxmin"]}\n  {fell["bnovelty"]}'
          f'\nselections padded with a duplicate behaviour: {padded}')


def test_bcoverage_extraction_is_the_stated_rule(space):
    """The plans the rule names, not merely the right behaviours: the cheapest
    plan exhibiting each, ties to the earliest position, behaviours in
    first-occurrence order, then padding in pool order.

    The paper leaves the representative open, but the library and the
    reference both state this rule, so positions are compared and not sets. It
    is exact over integer costs, so there is no tie tolerance to allow for and
    no legitimate way for the two sides to differ.
    """
    cheaper, reordered, padded = 0, 0, 0
    for case in range(200):
        sp = space(case)
        counter, plans = pool_for(sp, 80000 + case, low=0, high=10)
        trip = triples(plans)
        first = ref_distinct([b for _i, _c, b in trip])
        last = list(reversed(ref_distinct([b for _i, _c, b in reversed(trip)])))
        costs = {}
        for _i, cost, behaviour in trip:
            costs.setdefault(behaviour, []).append(cost)
        for k in (0, 1, 2, 3, 5, 9):
            chosen = positions(plans, counter.extract(plans, k, indicator='bcoverage',
                                                      k_nn=3))
            wanted = ref_extract('bcoverage', trip, sp.d, k, 3)
            assert chosen == wanted, (
                f'seed={case} {sp} k={k}: the library returned pool positions {chosen}, '
                f'the reference {wanted}; costs {[c for _i, c, _b in trip]}, '
                f'behaviours {[b for _i, _c, b in trip]}')
            # The three ways of covering the same behaviours by the wrong plans,
            # in the wrong order or in too few of them. Each must arise here.
            cheaper += k > 0 and any(len(set(c)) > 1 for c in costs.values())
            reordered += 0 < k < len(first) and first != last
            padded += min(k, len(plans)) > len(first)

    assert cheaper and reordered and padded, (
        f'the sample never separates the rule from its neighbours: {cheaper} cases of a '
        f'behaviour with plans of differing cost, {reordered} where first- and '
        f'last-occurrence order differ under a truncating k, {padded} needing padding')


# ----------------------------------------------------------------------
# The reference against the paper itself
# ----------------------------------------------------------------------
#: The paper's rover example, as tests/test_golden.py puts it to the library: a
#: behaviour is (number of rovers, three-letter collection order), the rover
#: count carrying the discrete dissimilarity and the order Hamming/3, both at
#: weight 1/2. Anchoring the reference to these numbers directly keeps the
#: audit from resting on the two implementations agreeing with each other.
ROVER_ONE = [(2, 'RIS'), (1, 'RIS'), (1, 'RIS')]
ROVER_TWO = [(1, 'RSI'), (1, 'RIS'), (2, 'SIR')]
ROVER_THREE = [(1, 'RSI'), (1, 'RIS'), (2, 'RSI')]

#: (behaviours, B-Coverage, B-MaxSum, B-MaxMin, {kappa: B-Novelty}).
PAPER_NUMBERS = [
    (ROVER_ONE, 2, 0.5, 0.5, {1: 0.5, 2: 0.5, 3: 0.5, 15: 0.5}),
    (ROVER_TWO, 3, 13 / 6, 1 / 3, {1: 1 / 2, 2: 13 / 18, 3: 13 / 18}),
    (ROVER_THREE, 3, 5 / 3, 1 / 3, {1: 7 / 18, 2: 5 / 9}),
    ([(1, 'RSI'), (2, 'SIR')], 2, 1.0, 1.0, {1: 1.0}),
    ([(1, 'RSI')], 1, 0.0, 0.0, {1: 0.0}),
]


def rover_d(x, y):
    """psi_M of the paper's example: 1/2 on the rover count, 1/2 Hamming/3."""
    hamming = sum(a != b for a, b in zip(x[1], y[1])) / len(x[1])
    return 0.5 * (0.0 if x[0] == y[0] else 1.0) + 0.5 * hamming


def test_the_reference_reproduces_the_papers_numbers():
    """Every dissimilarity and indicator the paper states of its example."""
    assert rover_d(ROVER_TWO[0], ROVER_TWO[1]) == pytest.approx(1 / 3)
    assert rover_d(ROVER_TWO[0], ROVER_TWO[2]) == pytest.approx(1.0)
    assert rover_d(ROVER_TWO[1], ROVER_TWO[2]) == pytest.approx(5 / 6)
    assert rover_d(ROVER_THREE[0], ROVER_THREE[2]) == pytest.approx(1 / 2)
    assert rover_d(ROVER_THREE[1], ROVER_THREE[2]) == pytest.approx(5 / 6)
    for behaviours, coverage, maxsum, maxmin, novelties in PAPER_NUMBERS:
        assert ref_bcoverage(behaviours) == coverage, behaviours
        assert ref_bmaxsum(behaviours, rover_d) == pytest.approx(maxsum), behaviours
        assert ref_bmaxmin(behaviours, rover_d) == pytest.approx(maxmin), behaviours
        for kappa, value in novelties.items():
            assert ref_bnovelty(behaviours, rover_d, kappa) == pytest.approx(value), (
                f'{behaviours} kappa={kappa}')


def test_the_reference_reproduces_the_papers_selections():
    """The selections test_golden.py asserts of the library, asked of the
    reference: padding to k, the cheapest representative, the earlier of two
    equal costs, the farthest opening pair, the lowest-position tie-break."""
    six = ROVER_TWO + [(1, 'RSI'), (2, 'SIR'), (1, 'RIS')]
    chosen = ref_extract('bcoverage', [(i, 1, b) for i, b in enumerate(six)], rover_d, 5, 1)
    assert len(chosen) == 5 and ref_bcoverage([six[i] for i in chosen]) == 3

    costed = [(0, 9, (1, 'RSI')), (1, 2, (1, 'RSI')), (2, 5, (2, 'SIR'))]
    assert ref_extract('bcoverage', costed, rover_d, 2, 1) == [1, 2]
    assert ref_extract('bcoverage', [(0, 3, (1, 'RSI')), (1, 3, (1, 'RSI'))],
                       rover_d, 1, 1) == [0]

    three = [(i, 1, b) for i, b in enumerate(ROVER_TWO)]
    pair = ref_extract('bmaxsum', three, rover_d, 2, 1)
    assert ref_bmaxsum([ROVER_TWO[i] for i in pair], rover_d) == pytest.approx(1.0)

    twins = [(0, 1, (1, 'RSI')), (1, 1, (2, 'SIR')), (2, 1, (2, 'SIR'))]
    for indicator in INDICATORS:
        assert ref_extract(indicator, twins, rover_d, 2, 1)[-1] == 1, indicator


def test_extraction_matches_reference_on_the_committed_pools(tmp_path):
    """The same comparison again, on the pools the experiments actually run.

    The rest of the audit runs on random stub spaces, whose dissimilarities are
    drawn to make near-ties common. These four committed pools -- no planner
    and no benchmark checkout needed -- carry the real feature models and the
    stability one, and answer what the stub spaces cannot: whether
    ``best_index``'s rounding can merge two genuinely different scores in the
    spaces used here. The smallest gap between two distinct dissimilarities is
    reported with the count.
    """
    cfg = load('smoke', results_dir=tmp_path)
    pools.ensure_pools(cfg)
    comparisons, faults, gaps, seen = 0, [], [], []
    for path in pools.pool_files(cfg):
        pool = pools.read_pool(path)
        task = pools.task_of(pool)
        info = runner.instance_info(cfg, pool)
        for spec in [models.generic_spec(cfg), models.domain_model(cfg, pool['domain']), models.STABILITY]:
            counter = models.build_counter(spec, task, info)
            loaded = pools.load_pool(path, counter=counter, task=task)
            dump = pools.behaviour_dump(cfg, counter, loaded,
                                        models.model_record(spec, counter, task, info))
            plans, distinct = loaded['plans'], dump['distinct']
            if dump['matrix'] is not None:
                matrix = dump['matrix']
            else:       # the stability model: the distance over the action sets
                matrix = [[ref_stability(x[0].split(' ; '), y[0].split(' ; ')) for y in distinct]
                          for x in distinct]

            def d(i, j, matrix=matrix):
                """psi_M between two behaviours of the dump, by their index."""
                return matrix[i][j]

            # A behaviour is its index into the dump's `distinct`, which is all
            # the reference asks of one: it only compares them for equality.
            trip = [(e['index'], e['cost'], e['distinct']) for e in dump['plans']]
            b = len(distinct)
            where = f'{pool["domain"]}/{path.stem} {spec.name} ({len(plans)} plans, {b}b)'
            seen.append(where)   # printed: the sweep is only as wide as the pools are
            apart = sorted({matrix[i][j] for i in range(b) for j in range(i + 1, b)})
            gaps.extend(y - x for x, y in zip(apart, apart[1:]))
            for k in (2, 3, 5, 8):
                for kappa in (1, 2, 3):
                    for indicator in INDICATORS:
                        chosen = positions(plans, counter.extract(
                            plans, k, indicator=indicator, k_nn=kappa))
                        wanted = ref_extract(indicator, trip, d, k, kappa)
                        got_b = [trip[i][2] for i in chosen]
                        want_b = [trip[i][2] for i in wanted]
                        got_v = ref_indicator(indicator, got_b, d, kappa)
                        want_v = ref_indicator(indicator, want_b, d, kappa)
                        comparisons += 1
                        assert len(chosen) == min(k, len(plans)), f'{where} k={k}'
                        if set(got_b) != set(want_b) or abs(got_v - want_v) > TOL:
                            faults.append(f'{where} k={k} kappa={kappa} {indicator}\n'
                                          f'    library   {got_b} -> {got_v!r}\n'
                                          f'    reference {want_b} -> {want_v!r}')

    assert comparisons, 'no committed pool was read'
    assert not faults, (f'{len(faults)} of {comparisons} extractions on the committed '
                        'pools disagree with the reference.\n' + '\n'.join(faults[:6]))
    # A pool of one behaviour contributes no gap at all -- driverlog under the
    # stability model is one -- so the measurement needs a pool that has two.
    assert gaps, 'no committed pool exposes two distinct behaviours to measure a gap between'
    assert min(gaps) > TIE_TOLERANCE, (
        f'two distinct dissimilarities on the committed pools lie {min(gaps):.3e} apart, '
        f'inside the tie tolerance {TIE_TOLERANCE:g}: rounding at '
        f'TIE_DECIMALS={TIE_DECIMALS} can merge genuinely different scores here')
    print(f'\n{comparisons} extractions over {len(seen)} (pool, model) pairs agree with '
          f'the reference; smallest gap between two distinct dissimilarities '
          f'{min(gaps):.4f}, tie tolerance {TIE_TOLERANCE:g}\n  ' + '\n  '.join(seen))


def test_every_dimension_is_definite_on_the_committed_pools(tmp_path):
    """Check 7: every per-dimension dissimilarity of every model, the
    stability one included, is zero only on equal values, over every pair of
    values the dimension takes on the smoke pools (Def. feature)."""
    cfg = load('smoke', results_dir=tmp_path)
    pools.ensure_pools(cfg)
    pairs = 0
    for path in pools.pool_files(cfg):
        pool = pools.read_pool(path)
        task, info = pools.task_of(pool), runner.instance_info(cfg, pool)
        for spec in models.selection_specs(cfg, pool['domain']) + models.timing_specs(cfg, pool['domain']):
            counter = models.build_counter(spec, task, info)
            counter.b_coverage(pools.load_pool(path, counter=counter, task=task)['plans'])
            for dim in counter.dimensions.values():
                values = sorted(map(str, dim.domain))
                for x in values:
                    assert dim.distance(f'{dim.name}:{x}', f'{dim.name}:{x}') == 0.0, (spec.name, dim.name, x)
                for x, y in combinations(values, 2):
                    assert dim.distance(f'{dim.name}:{x}', f'{dim.name}:{y}') > 0, (spec.name, dim.name, x, y)
                    pairs += 1
    assert pairs > 50, f'only {pairs} pairs of distinct values: the smoke pools are not exercising this'


def test_the_stability_model_is_the_stability_distance(task, plan_l1_then_l2, plan_l2_then_l1,
                                                        plan_two_trucks):
    """Check 8: the stability model on the transport fixture. A behaviour
    is a distinct action set, psi_M is ``ref_stability`` on every pair, and
    B-MaxSum selection under it is ``ref_extract_bmaxsum`` with the stability
    distance as d."""
    counter = BehaviourDiversityCounter(task, [('stability', None)])
    twin = type(plan_l1_then_l2)(list(plan_l1_then_l2.actions))   # a second plan, same actions
    plans = [plan_l1_then_l2, plan_l2_then_l1, plan_two_trucks, twin]
    actions = [[str(a) for a in plan.actions] for plan in plans]
    assert counter.b_coverage(plans) == len({frozenset(a) for a in actions}) == 3
    for i, j in combinations(range(len(plans)), 2):
        counter.b_coverage(plans)
        assert counter._pair_distance(plans[i].behaviour, plans[j].behaviour) == pytest.approx(
            ref_stability(actions[i], actions[j]))
    trip = [(i, 1, frozenset(a)) for i, a in enumerate(actions)]
    d = lambda x, y: ref_stability(x, y)
    for k in (1, 2, 3, 4):
        chosen = positions(plans, counter.extract(plans, k, indicator='bmaxsum', k_nn=1))
        assert chosen == ref_extract('bmaxsum', trip, d, k, 1), k


def test_the_audit_never_relies_on_the_default_k_nn():
    """The library's own default kappa is 3 and the paper fixes none, so every
    call the audit makes into ``b_novelty`` or ``extract`` must name its own."""
    here = os.path.dirname(os.path.abspath(__file__))
    call = re.compile(r'\.(?:extract|b_novelty)\((?:[^()]|\([^()]*\))*\)')
    borrowed = re.compile(r'^\s*(?:from|import)\b.*\bDEFAULT_K_NN\b', re.M)
    for name in ('conftest.py', 'test_audit.py'):
        with open(os.path.join(here, name)) as handle:
            source = handle.read()
        assert not borrowed.search(source), f'{name} imports the library default kappa'
        silent = [m.group(0) for m in call.finditer(source) if 'k_nn' not in m.group(0)]
        assert not silent, f'{name} leaves kappa to the library in: {silent}'
