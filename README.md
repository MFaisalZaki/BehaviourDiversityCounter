# Behaviour Diversity Counter

A package to infer the behaviour of a plan in a given set of plans, besides computing behaviour diversity count for this set of plans.

Two plans can look different action-by-action and still *do the same thing*. This package
decides what "the same thing" means by projecting each plan onto a set of user-chosen
**dimensions** — the order it achieves the goals, what it costs, which resources it burns —
and treating the combination of those projections as the plan's **behaviour**. Plans that
agree on every dimension are one behaviour, however different their action sequences.

From a set of plans you can then get the paper's four indicators, every one of them
computed over the **distinct** behaviours, so a duplicate plan changes none of them:

- `b_coverage(plans)` — B-Coverage: how many distinct behaviours the set actually
  covers. (`bdc(plans)` is a deprecated alias, under the indicator's former name,
  Behaviour Diversity Count.)
- `b_maxsum(plans)` — B-MaxSum: the sum of pairwise distances between the distinct
  behaviours.
- `b_maxmin(plans)` — B-MaxMin: the smallest pairwise distance. Fewer than two
  distinct behaviours score `0`, not `+inf`: a set offering the user no alternative
  should rank lowest, not highest.
- `b_novelty(plans, k_nn=3)` — B-Novelty: the mean, over the distinct behaviours, of
  each behaviour's mean distance to its `k' = min(k_nn, b - 1)` nearest neighbours.
  Also `0` below two behaviours.
- `extract(plans, k, indicator=...)` — select `k` plans maximising one of them:
  `'bcoverage'` (default), `'bmaxsum'`, `'bmaxmin'` or `'bnovelty'` — see Extracting
  diverse subsets.
- `behaviours(plans)` — the set of distinct behaviour strings the plans exhibit.

The counter itself is bound to a task and its dimensions; the plan set is an argument
to every indicator, so one counter can score any number of plan sets — including the
subsets `extract` returns.

Plans are replayed against the task, so they must be applicable to it: a plan whose
preconditions fail raises `InapplicablePlanError` rather than being counted.

The package consumes plans; it does not produce them. Generate them with any
[unified-planning](https://github.com/aiplan4eu/unified-planning) engine (`up-symk` for
top-k/diverse planning, `up-fast-downward` for single plans), then hand the list over.

## Install

```bash
poetry install                    # the library: unified-planning + lark
poetry install --with planners    # ...plus up-symk / up-fast-downward, to generate plans
```

The library imports only `unified-planning` and `lark`, so those are its only runtime
dependencies. The planner engines are an optional group because this package consumes
plans rather than producing them — install them if you want to generate the plan list in
the same environment.

## Quick start

```python
from unified_planning.shortcuts import *
from unified_planning.plans import SequentialPlan, ActionInstance
from behaviour_diversity_counter import BehaviourDiversityCounter

# A task: deliver to l1 and l2 using truck tr1.
Location, Truck = UserType('Location'), UserType('Truck')
at = Fluent('at', BoolType(), t=Truck, l=Location)
delivered = Fluent('delivered', BoolType(), l=Location)

move = InstantaneousAction('move', t=Truck, f=Location, to=Location)
t, f, to = move.parameter('t'), move.parameter('f'), move.parameter('to')
move.add_precondition(at(t, f))
move.add_effect(at(t, f), False)
move.add_effect(at(t, to), True)

drop = InstantaneousAction('drop', t=Truck, l=Location)
dt, dl = drop.parameter('t'), drop.parameter('l')
drop.add_precondition(at(dt, dl))
drop.add_effect(delivered(dl), True)

task = Problem('transport')
task.add_fluent(at, default_initial_value=False)
task.add_fluent(delivered, default_initial_value=False)
task.add_action(move); task.add_action(drop)

l0, l1, l2 = (Object(n, Location) for n in ('l0', 'l1', 'l2'))
tr1 = Object('tr1', Truck)
task.add_objects([l0, l1, l2, tr1])
task.set_initial_value(at(tr1, l0), True)
task.add_goal(delivered(l1)); task.add_goal(delivered(l2))

# Two plans that differ only in the order they reach the goals.
def plan(*steps):
    return SequentialPlan([ActionInstance(a, tuple(ObjectExp(o) for o in ps))
                           for a, ps in steps])

l1_first = plan((move, (tr1, l0, l1)), (drop, (tr1, l1)),
                (move, (tr1, l1, l2)), (drop, (tr1, l2)))
l2_first = plan((move, (tr1, l0, l2)), (drop, (tr1, l2)),
                (move, (tr1, l2, l1)), (drop, (tr1, l1)))

counter = BehaviourDiversityCounter(task, [('go', None)])
plans = [l1_first, l2_first]

counter.b_coverage(plans)   # 2
counter.behaviours(plans)   # {'go:delivered(l1)->delivered(l2)',
                            #  'go:delivered(l2)->delivered(l1)'}
counter.b_maxsum(plans)     # 1.0 -- one pair of behaviours, fully reordered
counter.b_maxmin(plans)     # 1.0 -- with one pair, the min and the sum coincide
counter.b_novelty(plans)    # 1.0 -- and so does the mean nearest-neighbour distance
counter.extract(plans, k=1) # one plan, covering one behaviour
```

Both plans cost the same and use the same truck, so under `('cb', ...)` or `('ru', ...)`
they would collapse to a single behaviour. The dimensions you pick *are* the definition of
diversity for your problem.

## Constructing the counter

```python
BehaviourDiversityCounter(task, dimensions, weights=None)
```

| argument | meaning |
| --- | --- |
| `task` | the `unified_planning` `Problem` the plans were built for |
| `dimensions` | an iterable of `(dimension_key, addinfo)` pairs — see below |
| `weights` | `{dimension_key: float}` for the separable distance `d(b, b') = Σᵢ wᵢ · dᵢ(bᵢ, b'ᵢ)`; `None` gives the uniform `1/n`, under which the distance is the mean over the dimensions |

`counter.set_weights({...})` changes them afterwards, and clears the pair-distance
cache as it does — that cache is keyed by the behaviour pair alone, so a surviving
entry would answer with the previous weight vector and say nothing about it.

The plan sets are not held by the counter: `bdc`, `b_maxsum`, `behaviours` and `extract`
each take any iterable of `SequentialPlan` as an argument. Each plan is replayed through
a `SequentialSimulator`, and each dimension turns the resulting state trace into a token.
The tokens are joined with ` $$ ` into one behaviour string per plan, which is attached
to the plan object as `plan.behaviour` and cached by plan identity — a plan is simulated
at most once per counter, however many sets it appears in. Pairwise behaviour distances
are memoised on the unordered pair, so repeated indicator calls and the greedy extraction
never recompute a distance they have already seen.

## Dimensions

| key | class | `addinfo` | example token |
| --- | --- | --- | --- |
| `go` | `GoalPredicatesOrderingDimension` | `None` | `go:delivered(l1)->delivered(l2)` |
| `cb` | `MakespanOptimalCostDimension` | `{'q': 1.5}` | `cb:4` |
| `rc` | `ResourceCountDimension` | path to a `(:resource ...)` file | `rc:tr1=4,tr2=0` |
| `ru` | `ResourceUsedDimension` | path to a `(:resource ...)` file | `ru:tr1,tr2` |
| `uv` | `UtilityValueDimension` | `{'utility-goals': {expr: int}}` | `utility_value:8 -- delivered(l1)=5,delivered(l2)=3` |
| `fn` | `NumericFunctionDimension` | path to a `(:function ...)` file | `fuel:8` |

**`go` — goal ordering.** The order in which the goal predicates first become true.
Its estimate is `len(goals)!`. Goals never achieved sort to the front (index `-1`).
`GoalPredicatesOrderingDimension` is a thin specialisation of
`LandmarkPredicatesOrderingDimension`, which can order any predicate set; only the goal
variant is wired into `dimensions_map`.

**`cb` — cost / makespan.** Plan length. `q` is the bound relative to the cheapest plan
seen: `q = 1.0` means optimal-only (estimate `1`), `q = 1.5` admits costs up to
`int(1.5 × optimal)`. The estimate reads `min(self.domain)`, so it is only meaningful
after the plans have been walked.

**`rc` / `ru` — resources.** Both read the same file and look at which objects named in it
appear as action parameters. `rc` keeps the per-object *counts*, emitted in sorted order so
the string is stable across processes; `ru` keeps only the *set* of objects used, so it
ignores how heavily each was used. Both estimate `2^n - 1` non-empty subsets — note this is
computed by materialising every subset, so it is exponential in the number of declared
resources.

**`uv` — utility value.** Sums the weights of goals that were *ever* true along the trace,
not just at the end. Keys are goal expressions, not strings.

**`fn` — numeric functions.** Bins a numeric fluent's final value and reports the bin
index. Bins are built from `range(min, max - delta, delta)`, so `0..100` step `10` gives
nine bins covering `0..90`; any value above the last bin's range falls back into it, which
means the top bin absorbs `90..100` as well.

## Behaviour string format

```
go:delivered(l1)->delivered(l2) $$ cb:4 $$ ru:tr1
└──────── one token per dimension, joined with ' $$ ' ────────┘
```

Each `distance()` locates its own token by splitting on `$$` and matching its `name:`
prefix — prefix, not substring, because a name like `ru` occurs inside object names such as
`truck1`.

## Input file formats

Resources and functions are declared in small Lark-parsed files. Both take a name followed
by `min`, `max` and `delta`; names may be parenthesised (`fuel(tr1)`).

```lisp
(:resource tr1 0 10 1)
(:resource tr2 0 10 1)
```

```lisp
(:function fuel 0 100 10)
```

For `fn`, `delta` is the bin width: a value is reported as the index of the bin it lands
in, so `fuel = 80` over `0..100` step `10` becomes bin `8`.

## The behaviour distance

Every indicator but `b_coverage` is built on one pairwise distance between behaviours,
the paper's separable

    d(b, b') = Σᵢ wᵢ · dᵢ(bᵢ, b'ᵢ)

with the weights defaulting to `1/n`, under which it is the mean of the per-dimension
`distance()` values and each pair scores in `[0, 1]`.

`b_maxsum(plans)` discards duplicate behaviours, then sums that distance over every
unordered pair of the distinct behaviours that remain. It is a sum over pairs, not an
average, so it grows with the number of distinct behaviours and can exceed `1`. Fewer
than two distinct behaviours score `0.0`, as they do under `b_maxmin` and `b_novelty`.

Three dimensions implement `distance()`, each normalised into `[0, 1]` so they combine
sensibly under uniform weights:

| dimension | distance |
| --- | --- |
| `go` | Hamming over the two orderings, divided by the number of goals |
| `cb` | `abs(c1 - c2) / max(c1, c2)` over the two plan costs |
| `ru` | Jaccard complement — `1 - |A ∩ B| / |A ∪ B|` — over the used sets |

`rc`, `uv` and `fn` do not implement one and raise `AssertionError`, so every indicator
but `b_coverage` can only be computed over dimension sets drawn from `go`, `cb` and `ru`.

## Extracting diverse subsets

`extract(plans, k, indicator=..., k_nn=3, trace=False)` selects `k` plans from the given
pool, maximising the chosen indicator:

- `'bcoverage'` (default; `'bdc'` is an alias) scans the pool in order and takes a plan
  only when its behaviour has not been seen yet. Once every behaviour is covered, the
  remaining slots are filled with duplicates, which leave the indicator unchanged. It
  calls no distance function at all.
- `'bmaxsum'` is greedy: it repeatedly adds the plan whose behaviour has the greatest
  summed distance to the behaviours already selected. The first pick is arbitrary, since
  singleton sets score zero, and duplicates gain nothing, so they are only picked once
  every remaining candidate repeats a selected behaviour.
- `'bmaxmin'` is farthest-first: it starts from the pair of behaviours at maximum
  distance, then repeatedly adds the plan whose behaviour maximises the minimum distance
  to those already chosen.
- `'bnovelty'` is greedy on B-Novelty, over the plans that contribute a behaviour not
  already selected — the same convention the other two follow, and B-Novelty is the one
  indicator that needs it stated, being the one a duplicate leaves *exactly* unchanged.

Like the distance itself, all but `'bcoverage'` are only defined over `go`, `cb` and `ru`.

**B-MaxMin and B-Novelty are not monotone**: adding a plan can lower them. What comes
back is therefore the highest-scoring *prefix* of the greedy order rather than its final
`k` plans — the longest prefix that attains it, since only a strict fall is a reason to
hand the user fewer plans than they asked for. For B-MaxMin that prefix is usually the
seed pair, whatever `k` was: a third behaviour can only lower a minimum. That is a
property of the indicator, not a bug in the extraction.

`trace=True` returns a `Selection` instead of a plan list, so the non-monotonicity is
reportable rather than merely worked around:

```python
selection = counter.extract(plans, k=20, indicator='bmaxmin', trace=True)
selection.plans      # the best-scoring prefix -- what a caller should use
selection.order      # all 20 picks, in the order the greedy made them
selection.scores     # the indicator after each pick; scores[i] is of order[:i+1]
selection.best_step  # len(selection.plans)
```

Ties are broken by lowest plan index in pool order, everywhere, through a tolerance:
greedy scores are sums of the same distances accumulated in different orders, so two
mathematically equal candidates routinely differ by one unit in the last place, and
letting that decide the pick is reproducible but not stable.

## Instrumentation

`counter.enable_counters()` turns on three per-run counters, for measuring where the
time goes:

| counter | counts |
| --- | --- |
| `n_distance_evals` | calls to the pairwise behaviour distance |
| `n_distance_misses` | those that missed the cache and were actually computed |
| `n_simulator_calls` | `SequentialSimulator.apply` calls |

Calls and misses are counted apart because the cache is what makes a behaviour distance
cheap — there are only `b` behaviour strings to compare, however many plans exhibit them.
`reset_counters()` zeroes them between runs, `counter.counters` reads them, and
`enable_counters(False)` turns them off again. They are off by default, and off costs
nothing at all: enabling swaps the hot methods through the instance dictionary rather
than leaving a flag to test on every distance lookup.

## Known issues

**The `uv` domain estimate is not a true upper bound.** Each dimension can estimate its
domain size (`_estimate_domain()` / `estimated_domain_size`); the counter no longer
aggregates these, but they remain part of the dimension interface.
`UtilityValueDimension._estimate_domain` builds each candidate as
`sum -- <all declared utilities>`, and that second part is identical for every subset, so
the set collapses to the distinct achievable *sums*. But `plan_behaviour` encodes *which*
goals were achieved, which distinguishes subsets that share a sum: two goals worth `5`
each give an `estimated_domain_size` of `2` (the sums `5` and `10`) against `3` real
behaviours (l1 only, l2 only, both). The same routine also enumerates subsets from
`r = 1`, excluding the empty one, so a plan that achieves nothing has no candidate either.
Fixing it means encoding candidates the way `plan_behaviour` does, and deciding whether
the empty subset counts as a behaviour.

**B-MaxSum is only defined over `go`, `cb` and `ru`.** The other three dimensions have no
`distance()` and raise `AssertionError` — see the B-MaxSum metric section.

### Fixed

- **`fn` was unusable.** Its parser inverted `min` and `max` against the grammar order,
  crashing `plan_behaviour` with `IndexError`; and `plan_behaviour` returned
  `','.join(val)` over an already-joined string, yielding `'f,u,e,l,:,8'` for `'fuel:8'`.
- **B-MaxSum crashed on `cb`.** `distance()` read `.actions` off its arguments, expecting
  plan objects, while `b_maxsum` passes behaviour strings. It now parses its
  own token and normalises into `[0, 1]`.
- **`rc` tokens were ambiguous.** Counts were joined with ` $$ `, the separator used
  *between* dimensions, with no `rc:` prefix. They are now one prefixed, comma-separated,
  sorted token — sorted because the counts came from a set, whose iteration order varies
  between processes, so the same plan produced different strings from run to run.
- **Inapplicable plans were counted silently.** `_simulate_` returned `[]` on a failed
  precondition, which the dimensions read as "no goal was ever achieved" — under `go` that
  is byte-identical to a valid plan achieving its goals in declaration order. It now raises
  `InapplicablePlanError` (a `ValueError`) naming the action and step that failed.

Each is covered by a regression test; `tests/` documents the original defect in the
docstring so the reason for each assertion survives.

## Tests

```bash
poetry install
poetry run pytest
```

```
tests/conftest.py         a tiny transport task, hand-checkable behaviour strings
tests/test_parsers.py     the (:resource ...) / (:function ...) file parsers
tests/test_dimensions.py  each dimension: tokens, domains, estimates, distances
tests/test_counter.py     bdc / extract / b_maxsum, and edge cases
```

The expected strings are worked out by hand from the fixture task rather than recorded from
the code, so a change in what a dimension *means* shows up as a failure.
