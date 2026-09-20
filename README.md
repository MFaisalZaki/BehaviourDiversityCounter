# Behaviour Diversity Counter

A library that tells plans apart by what they *do* rather than by the actions they
contain, and measures and selects for diversity on that basis.

Two plans can differ action by action and still do the same thing. This package projects
each plan onto a set of user-chosen **dimensions**, such as the order it achieves the
goals, what it costs, or which resources it uses, and treats the combination of those
projections as the plan's **behaviour**. Plans that agree on every dimension are one
behaviour, however different their action sequences. A set of dimensions is a
**diversity model**, and the set of behaviours it can produce is the **behaviour space**.

Over a set of plans the library computes four indicators, each over the *distinct*
behaviours the set exhibits, so a duplicate plan changes none of them:

| method | indicator |
| --- | --- |
| `b_coverage(plans)` | B-Coverage: the number of distinct behaviours |
| `b_maxsum(plans)` | B-MaxSum: the sum of pairwise dissimilarities between the distinct behaviours |
| `b_maxmin(plans)` | B-MaxMin: the smallest pairwise dissimilarity |
| `b_novelty(plans, kappa=3)` | B-Novelty: the mean, over the distinct behaviours, of each one's mean dissimilarity to its `min(kappa, b - 1)` nearest neighbours |

B-MaxSum, B-MaxMin and B-Novelty score `0` when the set holds fewer than two distinct
behaviours. `extract(plans, k, indicator=...)` selects `k` plans from a pool for any of
the four, and `behaviours(plans)` returns the distinct behaviour strings.

The package consumes plans; it does not produce them. Generate them with any
[unified-planning](https://github.com/aiplan4eu/unified-planning) engine and hand the
list over. Plans are replayed against the task, so they must be applicable to it: a plan
whose preconditions fail raises `InapplicablePlanError` rather than being counted.

## Contents

```
behaviour_diversity_counter/
  behaviour_diversity_counter.py   BehaviourDiversityCounter: the indicators and extract
  simulation.py                    replaying a plan: its state trace and cost
  dimensions/
    base.py                        BehaviourDimension, the interface every dimension implements
    declarations.py                the (:resource ...) / (:function ...) declaration parser
    goal_predicate_ordering.py     go
    cost_bound_makespan_optimal.py cb
    cost_bin.py                    cbin
    resources.py                   rc, ru
    utility_value.py               uv
    functions.py                   fn
    stability.py                   stability
```

The package exports `BehaviourDiversityCounter`, `InapplicablePlanError`, `plan_cost`,
`dimensions_map` and `DEFAULT_KAPPA`.

## Install

```bash
poetry install                    # the library: unified-planning, lark, numpy
poetry install --with planners    # ...plus up-symk and up-fast-downward, to generate plans
```

Python 3.10 to 3.13.

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

Both plans cost the same and use the same truck, so under `('cb', None)` or `('ru', ...)`
alone they would collapse to a single behaviour. The dimensions you pick *are* the
definition of diversity for your problem.

## Constructing the counter

```python
BehaviourDiversityCounter(task, dimensions, trace_cache=None)
```

| argument | meaning |
| --- | --- |
| `task` | the `unified_planning` `Problem` the plans were built for |
| `dimensions` | an iterable of `(dimension_key, addinfo)` pairs, see below |
| `trace_cache` | optional `{id(plan): (states, cost)}` mapping shared between counters over the same task, so a plan is simulated once across all of them |

Each pair is one **feature**: a dimension, its extracting function and a per-dimension
dissimilarity in `[0, 1]`. The key names all three; the `addinfo` carries whatever the
dimension needs. An unknown key raises `ValueError`.

The counter holds no plan set. Every indicator and `extract` take any iterable of
`SequentialPlan`. Each plan is replayed once through a `SequentialSimulator`, each
dimension turns the state trace into one token, and the tokens are joined with ` $$ `
into one behaviour string, attached to the plan as `plan.behaviour` and cached by plan
identity. Pairwise dissimilarities are memoised on the unordered pair.

## Dimensions

| key | `addinfo` | example token |
| --- | --- | --- |
| `go` | `None`, or `{'max-goals': m}` | `go:delivered(l1)->delivered(l2)` |
| `cb` | `None` | `cb:4` |
| `cbin` | `{'optimal-cost': c, 'q': 2.0, 'width': 0.1}` | `cbin:3` |
| `rc` | `{'resources': [declaration strings]}` | `rc:2` |
| `ru` | `{'resources': [declaration strings]}` | `ru:tr1,tr2` |
| `uv` | `{'utility-goals': {expr: value}}` | `utility_value:8 -- delivered(l1)=5,delivered(l2)=3` |
| `fn` | `{'functions': [declaration strings]}` | `fn:fuel=8` |
| `stability` | `None` | `stability:drop(tr1, l1) ; move(tr1, l0, l1)` |

**`go`, goal ordering.** The order in which the goal atoms first become true. Atoms are
taken in the order the problem's goals introduce them; `max-goals` keeps only the first
`m`. Goals never achieved sort to the front.

**`cb`, cost.** The plan's cost: the sum of its action costs under the task's
`MinimizeActionCosts` metric, or the plan length when the task declares none.

**`cbin`, cost bin.** The bin of `cost / optimal-cost` in a grid of width `width` over
`[1, q]`, so `q = 2.0` at the default width gives ten bins. Ratios below 1 land in the
first bin and ratios at or above `q` in the last. `q` defaults to `1.0`, one bin, and
`width` to `0.1`.

**`rc` and `ru`, resources.** Both look at which declared objects appear as action
parameters. `rc` is how many of them the plan uses at all; `ru` is the set it uses.

**`uv`, utility value.** Which utility goals were ever true along the trace, with their
values and total. Keys are goal expressions, not strings.

**`fn`, numeric functions.** The final value of each declared numeric fluent, reported as
a bin index: `(:function f min max delta)` bins `[min, max)` into bins of width `delta`,
so `0..100` step `10` gives bins `0..9`. Values below `min` land in the first bin and
values at or above `max` in the last.

**`stability`, action set.** The plan's set of actions. This is the literature's
plan-level model stated as one feature: two orderings of one action set are one
behaviour.

### Declaration strings

Resources and functions are declared as a list of strings, one declaration each, under
the `resources` key of the `addinfo` for `rc` and `ru`, and the `functions` key for `fn`.
Both take a name followed by `min`, `max` and `delta`; names may be parenthesised
(`fuel(tr1)`). A declaration file becomes such a list with
`open(path).read().splitlines()`.

```python
('ru', {'resources': ['(:resource tr1 0 10 1)', '(:resource tr2 0 10 1)']})
('fn', {'functions': ['(:function fuel 0 100 10)']})
```

### Behaviour strings

```
go:delivered(l1)->delivered(l2) $$ cb:4 $$ ru:tr1
```

One token per dimension, joined with ` $$ `. Each dimension locates its own token by its
`name:` prefix.

## The model's dissimilarity

The model's dissimilarity `ψ_M` is defined on two plans. Each dimension compares the
value it extracts from one plan with the value it extracts from the other, and `ψ_M` is
the mean of those per-dimension dissimilarities:

    ψ_M(π, π') = (1/n) · Σᵢ ψᵢ(extractᵢ(π), extractᵢ(π'))

Each `ψᵢ` lies in `[0, 1]`, so `ψ_M` does too. `counter.dissimilarity(plan, plan')`
returns it. A plan enters `ψ_M` only through its behaviour, so two plans with the same
behaviour are at distance `0` and every indicator but B-Coverage evaluates `ψ_M` once per
pair of distinct behaviours. Inside each dimension, `ψᵢ` reads its own token out of the
two behaviour strings:

| dimension | dissimilarity |
| --- | --- |
| `go` | Hamming distance over the two orderings, divided by the number of goals |
| `cb` | `|c - c'| / max(c, c')` over the two costs |
| `cbin` | `|i - i'| / (bins - 1)` over the two bin indices; `0` with one bin |
| `rc` | `|c - c'| / (c + c')` over the two counts; `0` when both are `0` |
| `ru` | Jaccard distance, `1 - |A ∩ B| / |A ∪ B|`, over the used sets |
| `uv` | weighted Jaccard distance over the achieved utilities, `1 - Σ min(u, u') / Σ max(u, u')` |
| `fn` | per function `|i - i'| / (bins - 1)`, averaged over the declared functions |
| `stability` | Jaccard distance over the two action sets |

## Extracting diverse subsets

```python
counter.extract(plans, k, indicator='bcoverage', kappa=3)
```

Selects `k` plans from the pool for the chosen indicator, one of `'bcoverage'`,
`'bmaxsum'`, `'bmaxmin'` or `'bnovelty'`. `k` plans come back whenever the pool holds that
many.

- `'bcoverage'` takes one plan per behaviour, the cheapest in the pool exhibiting it, in
  the order the behaviours first appear. Once every behaviour is covered the remaining
  slots are filled in pool order. It reads no dissimilarity.
- `'bmaxsum'` and `'bmaxmin'` open on the two plans farthest apart, then repeatedly add
  the candidate whose dissimilarities to the selection sum highest (`'bmaxsum'`) or whose
  smallest dissimilarity to the selection is largest (`'bmaxmin'`).
- `'bnovelty'` opens on the farthest pair, then repeatedly adds the plan maximising
  B-Novelty over the selection plus that plan.

Under every rule, candidates are ranked among the plans whose behaviour is new to the
selection, and a repeated behaviour is taken only once no fresh one remains. B-MaxMin and
B-Novelty are not monotone: adding a plan can lower them. The selection still returns
the `k` plans the greedy picks, and the indicator reported for the returned set shows any
fall.

Ties are broken by the lowest plan index in pool order, everywhere. Scores agreeing to
nine decimals count as tied, so that floating-point noise in the order of summation does
not decide a pick.
