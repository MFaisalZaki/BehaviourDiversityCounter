# Paper experiments — Reshaping Diversity Planning: Take 2

Empirical evaluation for the paper, over the ForbidIterative plan pools in
`data/fi-generated-plans-dir.zip`. **No planner is ever run**: the 6027 pools
are the input, and the experiments only select from them and score what they
selected.

| # | Question | Output |
|---|----------|--------|
| **A** | **Selection cost**: how does selection time grow with the pool size, for each rule? | `exp_a/*.csv` → `exp_a_summary.csv` |
| **B** | **What each rule selects**: how often do the five rules disagree, and what does the plan-level baseline miss? | `exp_b/*.csv`, `exp_b_overlap/*.csv` → `exp_b_summary.csv`, `exp_b_tests.csv`, `exp_b_overlap_summary.csv` |
| **C** | **The effect of weights**: how sensitive is the selection to the user's weights, and does raising a weight move the selection the way the paper claims? | `exp_c/*.csv` → `exp_c_summary.csv`, `exp_c_tests.csv` |

The five conditions are identical in all three:

| condition | what it maximises |
|---|---|
| `bcoverage` | B-Coverage — the number of distinct behaviours |
| `bmaxsum` | B-MaxSum — the total pairwise behaviour distance |
| `bmaxmin` | B-MaxMin — the smallest pairwise behaviour distance |
| `bnovelty` | B-Novelty — the mean distance to the `k_nn` nearest behaviours |
| `maxsum_stability` | greedy MaxSum over Stability — the plan-level baseline, which never touches the behaviour space |

**One pool per task, shared by every condition.** All five select from the
identical list of plans; nothing else differs between conditions. That is the
whole reason the comparison is valid, so there is no per-condition filtering,
sorting or regeneration anywhere.

## Running it

```bash
./setup_benchmark.sh                     # interactive: asks which configuration, then does everything
./setup_benchmark.sh --yes               # all defaults, no prompts
./setup_benchmark.sh --config smoke --yes            # three pools, ~1 minute end to end
./setup_benchmark.sh --config discriminating --yes   # the stratum where the rules can differ
./setup_benchmark.sh --list-configs
```

It asks which experiment configuration to run, where everything should live
(one root directory holds the venv, the classical-domains checkout, the
experiment and the sandbox), the per-pool limits, the slurm knobs, and any extra
python packages the venv should get. From there it creates the virtualenv,
installs the library and the harness, clones classical-domains, unpacks the
pools, writes the limits into the experiment, and generates the job arrays. It
ends by printing the `sbatch` command that starts the sweep.

Every prompt has a matching flag, so a scripted run is the same script.

### The stages

```
bdcevalcli init      → an experiment directory (limits + pool selection + knobs)
bdcevalcli discover  → which FI pools a configuration selects
bdcevalcli generate  → one run command per (experiment, pool), plus slurm arrays
bdcevalcli run       → ONE experiment over ONE pool                (slurm calls this)
bdcevalcli analyze   → the summary tables and the headline numbers
bdcevalcli report    → the paper figures
```

Everything except `run` is stdlib-only — none of it imports `unified_planning`
— so a sweep can be generated and analysed on a laptop while only the compute
nodes carry the planning stack. (`report`'s figures want matplotlib; `analyze`'s
tables do not: the Wilcoxon test is written out in `paperexps/stats.py` rather
than imported from scipy.)

```bash
bash <sandbox>/slurm/submit_all.sh       # cluster
bash <sandbox>/run_local.sh 8            # or locally, 8 pools at a time
bash <sandbox>/run_local.sh 1            # Experiment A: see the warning below
```

> **Experiment A measures wall-clock.** Running it alongside anything else on
> the same machine inflates exactly what it is trying to measure. Use one job,
> or a node that is not sharing its CPUs.

## Layout

```
paper-experiments/
  exp-configurations/            experiment templates: default, discriminating, smoke
  data/
    fi-generated-plans-dir.zip   the FI pools (one JSON per task × requested-k)
    ru-info-dir/                 per-domain (:resource ...) declarations
  paperexps/
    cli.py        the bdcevalcli entry point
    config.py     exp-details.json: limits, pool selection, experiment knobs
    discover.py   which pools a configuration selects (no simulation)
    generate.py   run commands + slurm job arrays
    common.py     pool file → PDDL task + resource matching
    harness.py    one pool → task, plans, counter, behaviours (once, shared)
    baseline.py   Stability and greedy MaxSum over it — the plan-level baseline
    exp_a.py exp_b.py exp_c.py
    stats.py      Wilcoxon signed-rank + Holm, without scipy
    aggregate.py  the CSVs → summary tables and headline numbers
    plots.py      the figures
  setup_benchmark.sh
```

## Matching pools to tasks

A pool file `1.0-100-classical-2006-rovers-15-fi-bc-results.json` is
`{q}-{k}-{track}-{year}-{domain}-{inst}`. Its `info` block names the exact files
inside the classical-domains checkout (`info.domain` = `rovers/domain.pddl`
relative to `classical/`, `info.problem` = the problem basename), so task
matching never guesses. The filename's `{domain}-{year}` and `{inst}` key the
resource declarations: `data/ru-info-dir/{domain}/{domain}-{year}.json` holds
`instances[inst]`, where `inst` is the 1-based position of the problem in the
domain's problem list sorted by file name (catalog-run-2's convention). Pools of
48 bytes are planner timeouts and are skipped at generation time.

## Behaviour space

The paper's space: goal-predicate ordering (`go`) plus, where the instance
declares resources, the set of resources used (`ru`). `ru` rather than `rc`
because the indicators need a per-dimension distance, which only `go`, `cb` and
`ru` implement. Domains without resources measure diversity along `go` alone,
and are therefore out of Experiment C, which needs two dimensions to weight
against each other.

The distance is the paper's separable one, `d(b, b') = Σᵢ wᵢ · dᵢ(bᵢ, b'ᵢ)`,
with weights defaulting to `1/n`.

## Determinism

Every runner takes `--seed` and records it in every output row; every other seed
is derived from that one and a salt naming its purpose, so Experiment B's
hidden-preference draws are independent of everything else while the whole run
still reproduces from one number. Tie-breaking is by lowest plan index in pool
order, everywhere, and never depends on set or dict iteration order — golden
test 7 runs all five selectors in subprocesses under different
`PYTHONHASHSEED`s, which is what such a bug actually looks like.

Selection hashes in Experiment C are sha1, not Python's `hash()`, which is
randomised per process and would compare across an array job's jobs as noise.

## Reading the results

### Experiment B is reported twice, and the split is the result

**61% of the k = 1000 pools hold fewer than k = 20 distinct behaviours** — 28%
hold exactly one. On such a pool no selection rule can differ from any other:
every rule covers every behaviour there is, and `redundancy = k - b_coverage`
measures the pool rather than the rule. `analyze` therefore reports every table
over all tasks *and* over the tasks whose pool holds at least k behaviours, and
each row carries `redundancy_floor` (`k - min(k, b_pool)`, forced on every rule
alive) beside `redundancy_excess` (the rule's own share). Reading the raw
median without those two attributes the benchmark's behaviour collapse to the
baseline.

### Two things that look like bugs and are not

- **B-MaxMin returns far fewer plans than the k asked for.** A third behaviour
  can only lower the minimum pairwise distance, so the best-scoring prefix of
  the farthest-first order is usually the seed pair — `thm:bmaxmin-degenerate`
  appearing in the data. It is read as a certificate over the *other*
  selections rather than as a selector in its own right. `n_selected` is in
  every row so a short selection is never mistaken for a redundant one.
- **B-Novelty nearly duplicates B-MaxSum when `b ≤ k_nn`.** The clamp
  `k' = min(k_nn, b - 1)` leaves every behaviour averaging over all the others,
  which is B-MaxSum divided by `C(b, 2)`. `b_pool` is in every row so this is
  visible rather than looking like a copy-paste error.

### The hidden-preference column

Before any selection, and from a seed salted apart from every other, a dimension
`f*` and a value `v*` that some plan in the pool exhibits are drawn and shown to
no selector; the question is whether each selector's returned set contains a
plan with `f* = v*`. Fifty independent draws per task.

It is the only measurement in the section that the behaviour-space indicators
cannot rig, because no selector sees the preference. Everything else shows the
selections are *different*; this is the only thing that speaks to *better*, and
it is reported whichever way it comes out.

Note its two saturating tails, which are properties of the measure and not of
the rules: where `b ≤ k` every rule scores 1.0, and where `b ≫ k` every rule
scores about `k/b`. The rules can only separate where `b` is modestly above `k`.

## Method notes

- **`A(p)` is a multiset in the baseline**, following Katz and Sohrabi (2020):
  an action occurring three times contributes three times. The `plandiversity`
  implementation this was lifted from uses `frozenset` and documents it —
  "Repeated actions are collapsed" — under which a plan that drives back and
  forth four times is identical to one that drives once, which would hand the
  baseline a weaker notion of difference than its own authors defined and
  flatter the behaviour space by exactly that much. `Stability(multiset=False)`
  restores the set reading; golden test 8 asserts the two actually differ.

- **The baseline uses its own greedy**, Katz and Sohrabi's: open on the farthest
  pair, then add whichever plan most increases the total. `extract(...,
  'bmaxsum')` opens arbitrarily, because every singleton scores zero. Each rule
  is left as its own authors define it; where the asymmetry helps a condition,
  it helps the baseline.

- **Overlap is reported at the behaviour level.** `jaccard` is between the
  distinct behaviour sets `B(Sᵢ)` and `B(Sⱼ)`, not between the plan sets: two
  selectors can return entirely disjoint plans that exhibit exactly the same
  behaviours, which is maximal disagreement at the plan level and none at all at
  the level the paper is about. `overlap_jaccard_plans` carries the plan-level
  number beside it, because the gap between the two is itself worth a sentence.

- **Spread is per dimension and never aggregated**, and uses the dimension's own
  raw distance rather than its weighted contribution, so `spread_go` means the
  same thing at every point of Experiment C's weight sweep.

- **Extraction is timed apart from selection.** Replaying every plan through the
  `SequentialSimulator` is by far the dominant cost, happens once per pool and
  is shared by all five conditions; folding it into the selection time would
  hide the result and invite the objection that the dimensions were chosen to be
  cheap. Every timed selection also starts on a cold distance cache — the cache
  is what the claim is about, so a repeat inheriting a warm one would be timing
  nothing.

- **No significance test on wall-clock.** Those are timings on one machine, not
  a sample from a population. Experiments B and C are tested with the Wilcoxon
  signed-rank test, paired per task, at p < 0.05, Holm-corrected across the
  selectors compared against the same baseline.

- **Cross-scoring runs both ways.** Every returned set is scored under all four
  indicators *and* under stability-MaxSum. A table showing only that the
  baseline scores badly under our own indicator is circular and a reviewer will
  say so.

## The library side

Four things the paper needs that the library did not have, all in
`behaviour_diversity_counter/`:

- `b_maxmin` and `b_novelty`, with the `'bmaxmin'` (farthest-first) and
  `'bnovelty'` (greedy) selectors. Both indicators can *fall* when a plan is
  added, so `extract` returns the highest-scoring prefix of the greedy order —
  the longest one that attains it, since only a strict fall is a reason to hand
  the user fewer plans — and `extract(..., trace=True)` returns the whole
  k-step trace.
- `b_coverage` as the primary name for what the paper no longer calls the
  Behaviour Diversity Count. The `bdc()` alias and the `'bdc'` selector key have
  been removed.
- Per-dimension weights, defaulting to `1/n`, set through `set_weights()`.
  Setting them clears the pair-distance cache, which is keyed by the behaviour
  pair alone — Experiment C reweights one counter object 21 times per task, and
  a surviving cache entry would produce 21 identical rows and raise nothing.

Experiment A's opt-in distance/simulator counters were removed as unused: no
`exp_a.py` was ever written against them. Reinstating them is a revert of the
commit that took them out.

`tests/test_golden.py` holds the paper's worked examples against a stub counter
needing neither PDDL nor a simulator. If those fail, nothing downstream is
trustworthy.

```bash
python -m pytest                # the whole suite
python -m pytest tests/test_golden.py -q
```
