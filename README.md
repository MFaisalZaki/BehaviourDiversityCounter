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
  covers. (The paper formerly called this the Behaviour Diversity Count.)
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
BehaviourDiversityCounter(task, dimensions)
```

| argument | meaning |
| --- | --- |
| `task` | the `unified_planning` `Problem` the plans were built for |
| `dimensions` | an iterable of `(dimension_key, addinfo)` pairs — see below |

Each pair is one **feature** in the paper's sense: a dimension, its extracting function,
a per-dimension distance in `[0, 1]`, and a weight `w > 0`. The key names the first three;
the weight is declared in the `addinfo` — `('go', {'weight': 0.25})`, or
`('ru', {'file': path, 'weight': 0.75})` for the dimensions that take a declaration
file. Declaring is all or nothing: a partial declaration raises `ValueError`, and with
**no** weight declared every dimension gets the uniform `1/n`, the paper's own choice for
its rover example (`1/2`, `1/2`). Under uniform weights the behaviour distance is the
mean of the per-dimension distances and lies in `[0, 1]`; under declared weights it lies
in `[0, Σᵢ wᵢ]`.

Each dimension **holds and applies** its own weight inside `distance()`, so the counter
only sums what the dimensions hand it; the counter decides the values, since every rule
about them is a rule about the whole set.


The plan sets are not held by the counter: `b_coverage`, `b_maxsum`, `behaviours` and `extract`
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
| `cb` | `MakespanOptimalCostDimension` | `None` | `cb:4` |
| `rc` | `ResourceCountDimension` | path to a `(:resource ...)` file | `rc:tr1=4,tr2=0` |
| `ru` | `ResourceUsedDimension` | path to a `(:resource ...)` file | `ru:tr1,tr2` |
| `uv` | `UtilityValueDimension` | `{'utility-goals': {expr: int}}` | `utility_value:8 -- delivered(l1)=5,delivered(l2)=3` |
| `fn` | `NumericFunctionDimension` | path to a `(:function ...)` file | `fn:fuel=8` |

**`go` — goal ordering.** The order in which the goal predicates first become true.
Goals never achieved sort to the front (index `-1`).

**`cb` — cost.** The plan's cost in the paper's sense: the sum of its action costs under
the task's `MinimizeActionCosts` metric, which is the plan length when the task declares
none. Its `addinfo` carries nothing but an optional weight.

**`rc` / `ru` — resources.** Both read the same file and look at which objects named in it
appear as action parameters. `rc` keeps the per-object *counts*, emitted in sorted order so
the string is stable across processes; `ru` keeps only the *set* of objects used, so it
ignores how heavily each was used.

**`uv` — utility value.** Sums the weights of goals that were *ever* true along the trace,
not just at the end. Keys are goal expressions, not strings.

**`fn` — numeric functions.** Bins a numeric fluent's final value and reports the bin
index. A dimension is a finite set, so a numeric criterion enters the space only after
quantisation, with the user fixing the bin width: `(:function f min max delta)` bins
`[min, max)` into bins of width `delta`, so `0..100` step `10` gives ten bins `0..9`.
Values below `min` land in the first bin and values at or above `max` in the last.

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

**On `k_nn = 3`.** Novelty search uses 15 and NSLC 20, but those count neighbours in a
population and archive of thousands. Here the neighbours come from the distinct
behaviours of one pool — tens — and `k_nn` is clamped to `b - 1`. Measured over random
pools, `k_nn = 15` makes B-Novelty *exactly* the mean pairwise distance (B-MaxSum over
`C(b, 2)`) for **100%** of pools with 16 or fewer behaviours; `k_nn = 3` never does above
four. Above the clamp the choice barely matters — at `b = 40` the overlap between the
B-Novelty and B-MaxSum selections is ~0.15 whether `k_nn` is 1, 3, 15 or 20 — so the only
thing the field's value would buy here is a second name for B-MaxSum on small pools.

Every dimension implements `distance()`, normalised into `[0, 1]` before its weight as the
paper's definition of a feature requires, so the weights are the only place one dimension
counts for more than another. Each is definite (zero exactly on equal values) and a
metric, so the definiteness assumption the paper's twinning theorem rests on holds for
any combination of them:

| dimension | distance |
| --- | --- |
| `go` | Hamming over the two orderings, divided by the number of goals |
| `stability` | the literature's model as one feature: the behaviour is the plan's action set and the distance is `1 - Jaccard` over two such sets (the stability distance of Srivastava et al.) |
| `cb` | `abs(c1 - c2) / max(c1, c2)` over the two plan costs |
| `ru` | Jaccard distance — `1 - |A ∩ B| / |A ∪ B|` — over the used sets |
| `rc` | weighted Jaccard over the count vectors — `1 - Σ min(c, c') / Σ max(c, c')`; the `ru` distance when every count is 0 or 1 |
| `uv` | weighted Jaccard over the achieved utilities — the utility of the goals both achieve against that of the goals either does |
| `fn` | per function `|i - i'| / (bins - 1)`, which respects the bin order as the paper asks of a quantised dimension, averaged over the declared functions |

## Extracting diverse subsets

`extract(plans, k, indicator=..., k_nn=3)` selects `k` plans from the given pool,
maximising the chosen indicator. It is the selection phase of the paper's two-phase
scheme: the pool comes from any planner that returns cost-bounded plans, and one pool
serves every indicator. `k` plans come back whenever the pool holds that many.

- `'bcoverage'` (the default) takes one plan per behaviour, in the order the behaviours
  first appear in the pool, and stops after `k`. Which plan represents a behaviour is left
  open by the paper, which takes the **cheapest plan in the pool that exhibits it**, as
  MAP-Elites keeps the fittest solution per cell; cost ties fall to the earliest plan. Once
  every behaviour is covered, the remaining slots are filled with duplicates in pool order,
  which leave the indicator unchanged. It calls no distance function at all.
- `'bmaxsum'` and `'bmaxmin'` are **one greedy rule under two aggregators**, after the
  shape [IBM diversescore](https://github.com/IBM/diversescore) uses — there, one scoring
  routine takes an `aggregator_metric` instead of each metric bringing its own
  implementation. Both keep, per candidate plan, the aggregate distance from its
  behaviour to the behaviours already selected, take the best candidate, then fold the
  newly selected behaviour into what remains. The aggregator is the only thing that
  changes inside the loop:

  | indicator | aggregator | paper's procedure |
  | --- | --- | --- |
  | `'bmaxsum'` | `+` | the greedy of Katz and Sohrabi (2020) |
  | `'bmaxmin'` | `min` | farthest-first (Ravi et al. 1994) |

  The paper adopts both from the literature and makes no claim about how close either
  comes to the optimum.

  **Both open on the farthest pair.** A singleton set has no pairs, so it scores zero
  under either operator — the opening pick gets no signal from the objective, and
  something has to supply one. The farthest pair maximises the indicator over every
  two-plan set, since the aggregate over a pair is the single distance between them.
  Under `min` the opening pair *is* the value of the selection and no later pick can
  raise it, so a bad start caps the whole run. Under `+` the seed is one summand among
  C(k, 2), so it matters less — but not so little that opening on plan 0 is defensible:

  | B-MaxSum, against brute force | opening on plan 0 | opening on the farthest pair |
  | --- | --- | --- |
  | reaches the optimum | 47.3% | **89.3%** |
  | mean ratio to optimum | 0.939 | **0.997** |
  | worst case observed | 0.508 | **0.866** |

  (7,713 random pools over a Euclidean metric; the same comparison over the paper's own
  `nr`/`co` dimensions gives 70.8% → 83.1%.) The seed costs O(b²) distance evaluations
  against the loop's O(b·k) — at b = 1000, k = 5 about 100× the distance calls, paid once
  into the cache the loop reads.
- `'bnovelty'` is the paper's plain greedy on B-Novelty: at every step it adds the plan
  maximising the indicator over the selection *plus that plan*. It does not share the
  loop above, because adding a behaviour moves the neighbourhood of every behaviour
  already held, so the candidate's value is a recomputation over the combined set rather
  than a fold over a per-candidate aggregate. It opens on the farthest pair for the same
  reason the other two do: over two behaviours each one's only neighbour is the other.

Under every rule, candidates are ranked among the plans whose behaviour is **new** to the
selection, and duplicates are taken only once every remaining candidate repeats a held
behaviour — the convention the paper states for B-MaxSum. Under `min` that falls straight
out of the aggregate (the distance from a behaviour to itself is zero); under `+` and for
B-Novelty it is imposed. B-Novelty is the one rule for which it needs saying: the
indicator is not monotone, so a fresh behaviour can lower the value below what a
duplicate would have preserved, and the fresh one is still taken.

**B-MaxMin and B-Novelty are not monotone**: adding a plan can lower them, and for
B-MaxMin a third behaviour can only lower a minimum over pairs. The selection nonetheless
returns the `k` plans the greedy picks rather than truncating to the best-scoring prefix,
as the paper argues: the indicator is there to certify that a set of the *requested* size
holds no two options too close together, not to choose that size. The fall is visible in
the indicator reported for the returned set instead of being concealed by a shorter
answer than the one asked for.

Ties are broken by lowest plan index in pool order, everywhere, through a tolerance:
greedy scores are sums of the same distances accumulated in different orders, so two
mathematically equal candidates routinely differ by one unit in the last place, and
letting that decide the pick is reproducible but not stable.

## Known issues

None known. The paper's worked examples are pinned by `tests/test_golden.py`: a
failure there means the library and the paper have parted company.

### Fixed

- **Three dimensions were not features.** `rc`, `uv` and `fn` had no `distance()`, so
  every indicator but B-Coverage raised on them. Each now has a definite metric in
  `[0, 1]` (see the distance table).
- **`fn` dropped its top bin.** Bins came from `range(min, max - delta, delta)`, so the
  last declared bin was folded into the one below it and was twice the user's width.
- **B-Coverage kept the first plan per behaviour**, whereas the paper keeps the cheapest.
- **`cb` was the plan length**, whereas the paper's cost is the sum of the action costs.
- **Weights defaulted to `1.0` each**, so the behaviour distance ranged over `[0, n]`;
  the uniform `1/n` of the paper's example is the default again.
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
tests/test_parsers.py     the (:resource ...) / (:function ...) declaration parser
tests/test_dimensions.py  each dimension: tokens and distances
tests/test_counter.py     the indicators and extract over the transport task, and edge cases
tests/test_golden.py      the paper's worked examples and selection conventions, on a stub
tests/experiments/        the evaluation: the Phase 0 audit and the smoke sweep
```

The expected strings are worked out by hand from the fixture task rather than recorded from
the code, so a change in what a dimension *means* shows up as a failure.

## The paper's evaluation

The empirical evaluation lives in `experiments/` as the package `bdc_experiments`, with its
own CLI: `bdcexp generate | run | report | jobs` unpacks the forbid-iterative pools shipped
under `experiments/data/` against the `classical-domains` benchmark, runs one selection
sweep and one timing sweep over them, and
writes the CSVs, LaTeX tables and figures the six subsections of the paper's evaluation
consume. Five of the six questions read the same selection sweep, since a selection at
any smaller `k` is a prefix of the run to the largest. Everything it produces goes under
one `runs/<name>/` directory, which is the artefact that ships with the paper: the pools
with their plans, a behaviour dump per model and pool holding every behaviour and the
dissimilarity matrix, one raw result file per task, and reports that are a pure function
of those two. [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) is how to run
it and what every output means.

Before any experiment was allowed to depend on this library, it was audited against an
independently written reference implementation of the paper's definitions. That audit found
one real defect — greedy selection could add a plan that is not the maximiser, because
candidate scores were rounded too coarsely before the argmax — and
[`docs/AUDIT.md`](docs/AUDIT.md) records every check, the fix, and the mutation testing that
shows the audit can actually fail.
