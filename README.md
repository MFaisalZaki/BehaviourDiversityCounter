# Behaviour Diversity Counter

A package to infer the behaviour of a plan in a given set of plans, besides computing behaviour diversity count for this set of plans.

Two plans can look different action-by-action and still *do the same thing*. This package
decides what "the same thing" means by projecting each plan onto a set of user-chosen
**dimensions** — the order it achieves the goals, what it costs, which resources it burns —
and treating the combination of those projections as the plan's **behaviour**. Plans that
agree on every dimension are one behaviour, however different their action sequences.

From a set of plans you can then get:

- `count()` — how many distinct behaviours the set actually covers.
- `estimated_behaviour_count()` — how many the task could admit (see the caveat under
  Known issues before relying on it as an upper bound).
- `optimise(k)` — pick `k` plans spreading across as many behaviours as possible.
- `compute_novelty_score()` — mean pairwise distance between behaviours, in `[0, 1]`.

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
from behaviour_diversity_counter.behaviour_diversity_counter import BehaviourDiversityCounter

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

counter = BehaviourDiversityCounter(task, [l1_first, l2_first], [('go', None)])

counter.count()                      # 2
counter.collected_behaviours         # {'go:delivered(l1)->delivered(l2)',
                                     #  'go:delivered(l2)->delivered(l1)'}
counter.estimated_behaviour_count()  # 2   -- 2 goals admit 2! orderings
counter.compute_novelty_score()      # 1.0 -- every position differs
counter.optimise(k=1)                # one plan, covering one behaviour
```

Both plans cost the same and use the same truck, so under `('cb', ...)` or `('ru', ...)`
they would collapse to a single behaviour. The dimensions you pick *are* the definition of
diversity for your problem.

## Constructing the counter

```python
BehaviourDiversityCounter(task, planlist, f)
```

| argument | meaning |
| --- | --- |
| `task` | the `unified_planning` `Problem` the plans were built for |
| `planlist` | any iterable of `SequentialPlan`; it is materialised into a list |
| `f` | an iterable of `(dimension_key, addinfo)` pairs — see below |

Each plan is replayed through a `SequentialSimulator`, and each dimension turns the
resulting state trace into a token. The tokens are joined with ` $$ ` into one behaviour
string per plan, which is also attached to the plan object as `plan.behaviour`.

## Dimensions

| key | class | `addinfo` | example token |
| --- | --- | --- | --- |
| `go` | `GoalPredicatesOrderingSimulator` | `None` | `go:delivered(l1)->delivered(l2)` |
| `cb` | `MakespanOptimalCostSimulator` | `{'q': 1.5}` | `cb:4` |
| `rc` | `ResourceCountSimulator` | path to a `(:resource ...)` file | `rc:tr1=4,tr2=0` |
| `ru` | `ResourceUsedSimulator` | path to a `(:resource ...)` file | `ru:tr1,tr2` |
| `uv` | `UtilityValueSimulator` | `{'utility-goals': {expr: int}}` | `utility_value:8 -- delivered(l1)=5,delivered(l2)=3` |
| `fn` | `FunctionsSimulator` | path to a `(:function ...)` file | `fuel:8` |

**`go` — goal ordering.** The order in which the goal predicates first become true.
Its estimate is `len(goals)!`. Goals never achieved sort to the front (index `-1`).
`GoalPredicatesOrderingSimulator` is a thin specialisation of
`LandmarkPredicatesOrderingSimulator`, which can order any predicate set; only the goal
variant is wired into `features_map`.

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

## Novelty score

`compute_novelty_score()` averages, over every unordered pair of plans, the mean of the
per-dimension `distance()` values. It returns `0.0` for fewer than two plans or no
dimensions.

Three dimensions implement `distance()`, each normalised into `[0, 1]` so they can be
averaged together:

| dimension | distance |
| --- | --- |
| `go` | Hamming over the two orderings, divided by the number of goals |
| `cb` | `abs(c1 - c2) / max(c1, c2)` over the two plan costs |
| `ru` | Jaccard complement — `1 - |A ∩ B| / |A ∪ B|` — over the used sets |

`rc`, `uv` and `fn` do not implement one and raise `AssertionError`, so a novelty score can
only be computed over feature sets drawn from `go`, `cb` and `ru`.

## Known issues

**The `uv` estimate is not a true upper bound.** `UtilityValueSimulator._estimate_domain`
builds each candidate as `sum -- <all declared utilities>`, and that second part is
identical for every subset, so the set collapses to the distinct achievable *sums*. But
`plan_behaviour` encodes *which* goals were achieved, which distinguishes subsets that
share a sum. Two goals worth `5` each give an estimate of `2` against `3` real behaviours:

```python
# utility-goals = {delivered(l1): 5, delivered(l2): 5}
counter.count()                      # 3  -- l1 only, l2 only, both
counter.estimated_behaviour_count()  # 2  -- only the distinct sums 5 and 10
```

The same routine also enumerates subsets from `r = 1`, excluding the empty one, so a plan
that achieves nothing has no candidate either. Since `estimated_behaviour_count()`
multiplies across dimensions, a low `uv` estimate drags the whole product below the real
count. Fixing it means encoding candidates the way `plan_behaviour` does, and deciding
whether the empty subset counts as a behaviour.

**Novelty is only defined over `go`, `cb` and `ru`.** The other three dimensions have no
`distance()` and raise `AssertionError` — see the Novelty score section.

### Fixed

- **`fn` was unusable.** Its parser inverted `min` and `max` against the grammar order,
  crashing `plan_behaviour` with `IndexError`; and `plan_behaviour` returned
  `','.join(val)` over an already-joined string, yielding `'f,u,e,l,:,8'` for `'fuel:8'`.
- **Novelty crashed on `cb`.** `distance()` read `.actions` off its arguments, expecting
  plan objects, while `compute_novelty_score` passes behaviour strings. It now parses its
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
tests/conftest.py       a tiny transport task, hand-checkable behaviour strings
tests/test_parsers.py   the (:resource ...) / (:function ...) file parsers
tests/test_features.py  each dimension: tokens, domains, estimates, distances
tests/test_counter.py   count / optimise / estimate / novelty, and edge cases
```

The expected strings are worked out by hand from the fixture task rather than recorded from
the code, so a change in what a dimension *means* shows up as a failure.
