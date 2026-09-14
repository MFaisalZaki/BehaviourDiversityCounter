# Phase 0: auditing the library against the paper

Before any experiment was allowed to depend on it, `BehaviourDiversityCounter`
was checked against an independently written reference implementation of the
paper's definitions. This note lists every check, its outcome, the one library
fix the audit produced, and the tie-breaking rule the evaluation reports.

The reference is in two places, both standard library only, no rounding,
written from the definitions of the paper alone by an author who had not read
the library's implementation of the same functions:
`experiments/bdc_experiments/reference.py` holds the four indicators and the
stability distance of the literature's model, which the reports also use to
read values off a behaviour dump; the four extraction rules (`ref_extract`)
live at the top of `tests/experiments/test_audit.py`, since the audit is
their only reader. The audit itself is that file, over random behaviour
spaces built in `tests/experiments/conftest.py`: 2 to 4 dimensions of 2 to 6
values each, one family of per-dimension dissimilarity that is a metric and one
that is definite but not a metric, uniform weights in about half the spaces and
declared unequal weights summing to 1 in the other half.

## The tie-breaking rule

The paper leaves tie-breaking arbitrary. The implementation does not:

> Ties are broken deterministically by the lowest index in the cost-sorted
> pool, at the opening pair and at every later step.

Every report states this sentence (`report.TIE_RULE`), and the pool is sorted
by `(cost, position in the pool file)` before any selection runs, so "lowest
index" is a property of the data and not of the order a planner happened to
emit its plans in.

## The checks

| # | Check | Outcome |
|---|---|---|
| 1 | Every library indicator equals the reference indicator to 1e-9, over 500 random spaces and pools, with and without duplicate behaviours, for kappa in {1, 2, 3, 5} | pass |
| 2 | `extract(..., indicator, k_nn)` returns the same behaviour set as the reference extraction, and — asserted separately — the same indicator value, for every indicator and k in 1..6 | pass, after the fix below |
| 3 | The paper's worked examples in `tests/test_golden.py` still pass | pass |
| 4 | Twinning: adding a plan whose behaviour is already held leaves all four indicators exactly unchanged | pass |
| 5 | B-Coverage and B-MaxSum never fall as a plan is added; B-MaxMin and B-Novelty have counterexamples in the sample | pass, counterexample quoted below |
| 6 | Extraction returns exactly `min(k, |C|)` plans even when the indicator fell during selection | pass |
| 7 | Every per-dimension dissimilarity of every model, `stability` included, is zero only on equal values, over every pair of values the dimension takes on the smoke pools (Def. feature's definiteness) | pass |
| 8 | The stability model on the transport fixture: `b` is the number of distinct action sets, `psi_M` equals `ref_stability` on every pair, and B-MaxSum selection under it equals `ref_extract_bmaxsum` with `ref_stability` as `d`; the same comparison over the committed smoke pools includes the stability model | pass |

Every one of those calls passes `k_nn` explicitly. The library's
`DEFAULT_K_NN = 3` is never relied on anywhere in the evaluation: the paper
fixes no kappa, so an experiment that took the default would be reporting a
number the paper does not define.

### Check 5: the counterexample

B-Coverage and B-MaxSum never fell anywhere in the sample — 200 spaces, every
prefix of every pool, 2 to 10 plans. Both non-monotone indicators have
counterexamples, and the first one found is the same step for both:

    Space(seed=1, dims=[2, 5], non-metric psi, uniform weights [0.5, 0.5])
    adding ('v0','v0') to {('v0','v2'), ('v1','v3')}:
        B-MaxMin      0.6254666610571267 -> 0.05146049545324627
        B-Novelty(3)  0.6254666610571267 -> 0.3653197191295887

The new behaviour lands 0.0515 from `('v0','v2')`, nearer than the 0.6255
separating the original pair, so the minimum over pairs collapses and every
behaviour's nearest-neighbour mean falls with it. This is claim C4 in
miniature; the paper reads both indicators at a fixed set size.

## The library fix

**`best_index` rounded candidate scores too coarsely, so greedy could add a
plan that is not the maximiser.**

`best_index` rounds every candidate score with `np.round(scores, TIE_DECIMALS)`
before `argmax`. At `TIE_DECIMALS = 3`, two candidates whose true scores
differed by anything below 5e-4 counted as tied and the lowest plan index won —
even where one of them was strictly the better. The paper's rule is "add the
candidate maximising the indicator", with ties, and only ties, broken
arbitrarily; a candidate 8.8e-05 behind the best is not tied with it. In the
audit's random sample the greedy gave away 8.832e-05 of B-MaxSum in one of 2880
extractions (seed 20, four non-metric dimensions, declared unequal weights).

The constant's purpose is sound. Greedy compares sums of the same distances
taken in different orders, so two mathematically equal candidates routinely
differ in the last bit, and letting that decide the pick makes the selection
depend on an unrelated change to how a score is accumulated. But the noise
being absorbed is of order 1e-15, and three decimals is a thousand times
coarser than that. `TIE_DECIMALS` is now **9**
(commit `fea47c6`, `behaviour_diversity_counter/behaviour_diversity_counter.py`).

**What that changes on the benchmark: nothing.** The plan asked whether the
rounding could merge two genuinely different scores "in the spaces used here",
and predicted it could not because the dissimilarities are rationals with small
denominators. Measured rather than assumed, over the pools of all nine
benchmark domains and the four committed smoke pools: **648 selections run at
both settings, none changed**, and the smallest gap between two distinct
pairwise dissimilarities on those pools is 0.05 — fifty times the old rounding
resolution. The prediction was right for the benchmark and wrong for a general
space, which is why the constant was tightened rather than left alone.

The reference needed the mirror-image fix at the same time. It compared raw
floats with a bare `>`, so two candidates equal in exact arithmetic but a few
ulps apart were *not* tied, and the lowest position did not win — the reference
was not honouring its own stated rule. It now carries `TIE_TOLERANCE = 1e-9`.
Fixing only one side would have swapped one class of disagreement for the
other; with both in place the audit is green.

The disagreement that first exposed this was found on a real pool, not in the
random sample: driverlog/2002/pfile3 under `driverlog_dispatcher`, B-Novelty at
k = 5, kappa = 3. Three candidate behaviours score exactly 19/72 there, and
`math.fsum` separated two of them by 5.55e-17.

## The three things the plan asked to look at closely

**`_extract_greedy` ranks only candidates with a new behaviour ("fresh"),
whereas the paper's B-MaxSum ranks all candidates by the combined-set value.**
Under the definiteness assumption the two agree, and check 2 confirms it on
every random space: a candidate whose behaviour is new contributes a strictly
positive sum of dissimilarities to the held set, while a duplicate contributes
exactly zero, so a fresh candidate always outranks a duplicate and the
restriction never changes the pick. The paper's own statement of the rule gives
a duplicate a gain of zero, so the two formulations are the same rule written
two ways.

**`best_index` rounds before argmax.** Answered above, and now measured.

**`b_novelty` breaks neighbour ties with `np.partition`, where the paper says
arbitrary.** This cannot change the value: `np.partition` returns *a* set of
kappa' smallest distances, and any other choice among tied neighbours is the
same multiset of numbers, hence the same mean. The reference sorts and slices
instead, and check 1 confirms the two agree to 1e-9 on every space, including
the spaces whose dissimilarity tables are quantised to a tenth so that exact
ties among neighbours are common rather than measure-zero.

## Is the audit real?

An audit that cannot fail proves nothing, so the audit was itself audited: the
installed library was mutated in process — from a scratchpad pytest plugin,
with the repository's own copy untouched — and the suite re-run against each
mutant. Caught: `b_maxmin` returning the mean instead of the minimum;
`b_novelty` dividing by an unclamped kappa; `b_coverage` counting plans instead
of distinct behaviours; `b_maxsum` and `b_maxmin` taken over all plan pairs
rather than distinct behaviours; `psi_M` itself rounded to three decimals; the
tie-break reversed to the last maximum; greedy taking the second-best candidate
at every step; `extract` returning the best-scoring prefix instead of
`min(k, |C|)` plans; and `TIE_DECIMALS` loosened to 1 and to 2.

Three mutations initially survived, all of them about *which plan* represents a
behaviour rather than *which behaviours* come back: keeping the most expensive
plan per behaviour instead of the cheapest, dropping the duplicate-behaviour
padding for B-Coverage and B-MaxSum, and taking behaviours in last-occurrence
order. The audit now asserts plan identity, plan count and behaviour order as
well as behaviour sets and indicator values, and catches all three.

## What the audit does not cover

The stub counter bypasses the real dimension classes, the simulator and the
cost model, so the audit says nothing about whether `go`, `cbin`, `ru` and `rn`
extract what their docstrings claim: it tests the indicators and the selection
rules, which is what the paper defines. The dimensions are covered by
`tests/test_dimensions.py`, and the real behaviour spaces are exercised
end to end by `tests/experiments/test_experiments_smoke.py` and by the
library-against-reference comparison that the audit runs on the committed
smoke pools.

## The stability model

`dimensions/stability.py` is the literature's model stated as one feature:
the extracting function reads a plan's action *set*, the dimension is the set
of such sets, and the dissimilarity is the stability distance of Srivastava
et al. (2007). It is definite on action sets, as Def. feature requires, so
two plans are twins under it exactly when they have the same action set; it
is not a metric-based model in the paper's sense, since two orderings of one
action set are at distance zero. Check 8 confirms that `extract_BMaxSum`
under it is the post-hoc greedy of Katz and Sohrabi (2020) with the same tie
rule. On the committed driverlog and rovers pools every plan is a permutation
of one action set, so the stability model sees one behaviour where the feature
models see six or eight; the case study's stability reading shows this.

The library's `_extract_b_coverage` docstring no longer cites the retired
optimality theorem; the paper (2026-09-14) makes no claim about the selection
functions, and the audit compares them with the reference rules only.
