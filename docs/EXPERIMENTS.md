# Running the evaluation

The evaluation of *Behaviour Spaces for Diversity Planning* lives in the
`bdc_experiments` package under `experiments/`. It has three commands, one
configuration file, and one output directory that is the artefact shipped with
the paper.

Everything under `paper-experiments/` is the previous harness. It is not read,
imported or run by anything described here.

## Install

```console
poetry install --extras analysis --with planners
```

* `--extras analysis` adds `scipy` and `matplotlib`. Only the report stage
  needs them; a compute node that merely runs tasks can leave them out.
* `--with planners` adds `up-symk`, whose wheel ships the SymK binary phase one
  drives. A host that only reports does not need it either.

The commands below are written as `poetry run bdcexp ...`; inside an activated
virtualenv `bdcexp` alone is enough.

## The three commands

```console
bdcexp generate <config> [--instance ID] [--list] [--force]
bdcexp run      <config> <select|time> [--task ID] [--list] [--force] [--jobs N]
bdcexp report   <config> <setup|e1|e2|e3|all>
```

`<config>` is a path to a TOML file, or the name of one shipped with the
package: `default` or `smoke`. Every command takes `--results-dir` to send a
run somewhere other than the config's `[run].results_dir`.

`generate` and `run` are resumable and idempotent: `generate` skips a pool
file that already exists, `run` skips a task that already has a result file,
and both take `--force` to redo one anyway.

### Phase one: the pools

```console
bdcexp generate default            # clones classical-domains, then runs SymK
bdcexp generate default --list     # the instance ids, one per line
```

The first call clones `AI-Planning/classical-domains` at the commit pinned in
`[benchmark].commit` into `runs/<name>/benchmark/`, then runs SymK once per
`(instance, mode, q, N)` for every `N` in `[generation].pool_sizes` and
`[e3].pool_sizes`. Instances are taken in instance-number order and the domain
stops at `[benchmark].instances_per_domain` solved ones.

### Phase two: two kinds of task

The paper's three questions are answered by two kinds of task, because the
first two read the same thing: the plans each selection function returns.

```console
bdcexp run default select              # the selection sweep, one process
bdcexp run default select --jobs 8     # the same, in a local process pool
bdcexp run default select --list       # the task ids, one per line
bdcexp run default time                # E3's timing
```

A `select` task is one pool under one model. It runs each of the four
selections once, to the largest `k` of `[selection].k_values`, and dumps the
order in which each took its plans. The
selection functions extend their answer one plan at a time, so the set
selected at any smaller `k >= 2` is a prefix of that run; the tests check this
on the committed pools. B-Coverage, B-MaxSum and B-MaxMin do not read `kappa`
and are run once; B-Novelty once per `kappa`. The models a pool is run under
are the generic control, the stability model, the domain's own model where one
exists, and the astronaut's model at each further weight setting of
`[e2].weight_settings` (rovers only).

A `time` task is one pool of `[e3].pool_sizes` under one of E3's models (one,
two and, where the domain has a model, three features). It times,
`[e3].repeats` times over, a cold mapping of the whole pool into the behaviour
space and each selection at each `k`.

A task that fails writes its traceback into its own result file and the
report counts it as a failure; it never takes the sweep down with it. A task
that has nothing to do -- an empty pool, say -- records why it was skipped.

### Reports

```console
bdcexp report default setup        # benchmark.csv and models.csv
bdcexp report default e2           # one question
bdcexp report default all          # the setup and all three
```

Reports are pure functions of `results/` and `behaviours/`. They never load a
pool, build a counter or run a selection: a selected set at `k` is the first
`k` plans of the recorded run, and its indicators are read off the behaviour
dump's matrix. So a report can be rebuilt on a laptop from the run directory
alone, with no planner installed.

| Report | Reads | Question |
|---|---|---|
| `e1` | the `select` results of the configured domains under their own models and under the stability model, at `q = 2.0` | the case study: one instance per domain, the pool's behaviours, the four selections at `k = 3`, every returned pair feature by feature, and the stability selection read under the domain's model |
| `e2` | the `select` results of the generic and per-domain models, and the weight variants | random equal-count subsets drawn and scored off the dumps; the selection cross table; the weight settings |
| `e3` | the `time` results | mapping and selection time against the pool size, `b` and the planner |

### On a cluster

```console
scripts/slurm_array.sh default                # select and time
scripts/slurm_array.sh default select         # one kind
```

One `sbatch --array` per task kind, one element per task, and a report job
that waits for both and runs `bdcexp report <config> all`. `SLURM_TIME`,
`SLURM_MEM` and `SLURM_LOGS` override the defaults. If a site's array limit
bites, split the generated `<kind>.tasks` file with `split -n l/<parts>` and
submit the parts.

### The smoke sweep

```console
bdcexp run    smoke select
bdcexp run    smoke time
bdcexp report smoke all
```

`configs/smoke.toml` runs against four pools committed under
`configs/smoke_pools/`, with the PDDL they were generated from beside them, so
it needs neither SymK nor the benchmark checkout. The first `run` copies those
pools into the run directory. This is what the tests and CI use; the whole
suite finishes in well under a minute.

## The output directory

```
runs/<name>/
  config.toml                 the configuration as run (its hash is in each manifest)
  benchmark/                  the classical-domains checkout
  pools/<domain>/<instance>/<mode>-q<q>-N<N>.json          phase one, plans included
  behaviours/<model_hash>/<domain>/<instance>/<pool>.json  per-plan behaviour + cost, b x b matrix
  results/<select|time>/<task>.json                        one raw dump per task
  reports/<setup|e1..e3>/                                  CSVs, tables/*.tex, figures/*.pdf, manifest.json
```

JSON everywhere for raw data, CSV only under `reports/`. Plan action strings
are stored verbatim, so a plan can be re-run with VAL or any planner; behaviour
tuples are lists of strings; dissimilarity matrices are stored in full, except
under the stability model, where a behaviour is the plan's action set, `b` runs
up to the pool size, and the matrix is left out (`null`): a reader recomputes a
stability distance from two action sets. Floats are stored unrounded and
rounded only in LaTeX. Every file carries a `schema` name and a version integer.

A `select` result names its task id, the configuration hash, the git revision,
the pool record, the model record (features, weights, the objects and goal
atoms they resolved to, the dimension sizes and their product `|BS|`), one
row per selection, and under `extra.selections` the run itself: the selected
plans' indices in the cost-sorted pool, their costs, behaviours, distinct
behaviour indices and action strings, the two clocks, and the four indicators
of the returned set. The rule is that no number in a result can fail to be
recomputed from the same file plus the behaviour dump;
`tests/experiments/test_recompute.py` enforces it.

## Configuration reference

One TOML file drives everything. Unknown sections and unknown keys are errors,
so a typo cannot silently change nothing.

| Section | Key | Meaning |
|---|---|---|
| `run` | `seed` | the seed every random draw is derived from |
| | `results_dir` | where the whole run is written |
| | `time_limit_generation_s` | CPU seconds per SymK call |
| | `memory_limit_generation_mb` | address space per SymK call |
| | `time_limit_selection_s` | wall seconds per phase-two task |
| `benchmark` | `source`, `commit` | the classical-domains repository and the pinned commit |
| | `domains` | directory names under `classical/`, not api.py `name` fields |
| | `instances_per_domain` | how many solved instances a domain contributes |
| `generation` | `planner` | `symk` |
| | `modes` | `topq`, `topk`, or both |
| | `pool_sizes` | the `N` of the pools the selection sweep runs over |
| | `q_values` | the quality bounds |
| `selection` | `k_values`, `kappa_values` | the selection grid every report reads |
| `models` | `generic` | the control model's features (library keys) and its two knobs |
| | `stability` | the literature's model has no knobs; the entry only says it is in play |
| `e1` | `domains` | one instance of each is the case study; `q = 2.0`, `k = 3` and `kappa = 1` are fixed by the paper |
| | `min_behaviours` | the case study's selection rule |
| `e2` | `random_subsets` | random equal-count subsets drawn per pool and `k` |
| | `weight_settings` | the astronaut's model's weights; each setting beyond the declared one is a model |
| `e3` | `pool_sizes`, `repeats` | the timing grid |

## Adding a diversity model

A model is data. Add a `ModelSpec` to `PER_DOMAIN` in
`experiments/bdc_experiments/models.py`:

```python
ModelSpec('zenotravel_planner', ('zenotravel',),
          (FeatureSpec('ru', {'types': ('aircraft',)}, 0.5),
           FeatureSpec('go', {}, 0.5)))
```

* `key` is a key of the library's `dimensions_map` (`go`, `cbin`, `ru`, `rn`,
  `rc`, `stability`, ...).
* `params` are the feature's own knobs and nothing instance-specific. A
  `types` entry names the PDDL types whose objects are the feature's
  resources; they are resolved against the problem's user types, or -- for the
  untyped STRIPS encodings -- against the unary predicate of the same name in
  the initial state.
* `weight` is the feature's weight, or `None` for all features of a model, in
  which case the library's uniform `1/n` applies.
* `domains` is a tuple of directory names, or `None` for every domain.

The spec's digest (`model_hash`) is part of the behaviour dump's path, so a
changed model is a changed path and nothing stale is read back. The setup
report's model table is generated from the registry.

## The output catalogue

### Columns every row carries

Nine fields identify the observation, and every CSV that reports per-pool rows
begins with them.

| Column | Meaning |
|---|---|
| `instance` | `<domain>/<ipc-year>/<problem-file-stem>` |
| `domain` | the benchmark directory the instance came from |
| `q` | the quality bound of the pool: plans cost at most `q * c*` |
| `N` | plans **requested** of the planner (`pool_size` is what came back) |
| `model` | the diversity model, or the variant of one |
| `k` | the selection size the row is about |
| `kappa` | the B-Novelty neighbourhood, always explicit, never the library default |
| `pool_size` | plans actually loaded, after the replay and cost filters |
| `b` | distinct behaviours the pool exhibits under this model |

A behaviour is referred to by its index into `distinct` in the behaviour dump,
and a plan by its index into the cost-sorted pool; a column holding several of
either is a space-separated list.

### E1, `sec:exp-case`

One instance per configured domain. `selector` is one of the four indicators
(the selection under the domain's model) or `stability` (the set B-MaxSum
selects under the stability model, read under the domain's model).

| File | Columns beyond the nine |
|---|---|
| `e1_behaviours.csv` | `distinct` (its index), `plans` (how many exhibit it), `cheapest_cost`, `f_<feature>` (the behaviour's value on each feature) |
| `e1_selections.csv` | `selector`, `position` in the returned order, `plan`, `cost`, `distinct`, `f_<feature>`, `set_<indicator>` (all four indicators of the returned set under the domain's model) |
| `e1_pairwise_diffs.csv` | `selector`, `plan_i`, `plan_j`, `feature`, `value_i`, `value_j`, `differs`, `psi` (the pair's dissimilarity under the domain's model), `differing_features` |
| `e1_stability.csv` | `selector`, `plan_i`, `plan_j`, `stability` (the pair's stability distance, from the action strings), `same_behaviour` (under the domain's model), `psi`, `covered` (behaviours the set covers) and `attainable` (`min(k, b)`) |

`e1_note.md` states, per domain, which instance the rule picked and why,
whether the agents feature was constant on it, two plans of the fullest
behaviour with their action strings, and what the stability selection covers.
The tables are one per domain: `tables/e1_behaviours_<domain>.tex`,
`tables/e1_selections_<domain>.tex` and `tables/e1_pairwise_<domain>.tex`.

### E2, `sec:exp-separation`

| File | Columns beyond the nine |
|---|---|
| `e2_random_subsets.csv` | `indicator`, `n_subsets` drawn, `enumerated` (all `C(b,k)` rather than sampled), `n_values` distinct values seen, `constant` (the check for B-Coverage), `modal_fraction`, `min`, `max`, `mean`; then the aggregate rows per `(k, kappa, indicator)`: `cells`, `constant_fraction`, `mean_modal_fraction` |
| `e2_subsets.csv` | every subset drawn: `subset` (behaviour indices), the three kappa-free indicators, and `bnovelty_kappa<kappa>` at each |
| `e2_kendall.csv` | `pair`, `n_subsets`, `enumerated`, `tau` (tau-b; empty where an indicator was constant over the sample), `p` |
| `e2_cross.csv`, `e2_cross_macro.csv` | `selector` (rows), `scored_<indicator>` (columns): that selection's value under that indicator, over the best of the four selections on the same pool; `n_<indicator>` the pools behind it; `pools` is `b>k` or `all` |
| `e2_weights.csv` | per pool, `k`, `kappa`, `setting` and `indicator`: `same_set`, `jaccard` (of the two returned behaviour sets), `value_declared`, `value_setting`; then the summary rows per `(setting, k, kappa, indicator)`: `pools`, `same_set_fraction`, `jaccard_mean` |

### E3, `sec:exp-cost`

| File | Columns beyond the nine |
|---|---|
| `e3_timing.csv` | one row per sample: `features` (the `n` of the cost expression), `phase` (`mapping` or `selection`), `indicator`, `repeat`, `wall_s`, `cpu_s`, `generation_wall_s` and `generation_cpu_s` from the pool record, `wall_over_generation`, `cpu_over_generation`, `exhausted` |
| `e3_medians.csv` | per `(N, k, model, features, phase, indicator)`: `samples`, `wall_median` with quartiles, `cpu_median` with quartiles, `cpu_pooled_mean`, `cpu_macro_mean`, `pool_size_median`, `b_median`, `generation_wall_median`, `wall_over_generation_median`, `wall_over_generation_macro` |

Only the wall-clock comparison against generation is like for like: the pool
record measures the planner subprocess's CPU with `getrusage(RUSAGE_CHILDREN)`
and its wall clock in the parent. Both ratios are in the CSV; the table and the
figure use the wall clock.

### Setup, `sec:exp-setup`

| File | Columns |
|---|---|
| `benchmark.csv` | `domain`, `ipc_year`, `instances`, `pools`, `pools_q<q>` per quality bound, `exhausted`, `timed_out`, `empty`, `plans` |
| `models.csv` | `model`, `domains`, `feature`, `description`, `params`, `dimension_size_rule`, `dissimilarity`, `weight` -- the generic control, the per-domain models and the stability model; the variants are named in the result files |

Each report also writes `manifest.json`: the git revision and whether the
tree was dirty, the config path and hash, the benchmark's pinned and actual
commit, the planner and the exact search strings, the limits, the seed, the
selection grid, package versions, the Python and platform, the tie-breaking
rule, the task counts (ok / failed / skipped), every failure with its message,
every output path, and the report's own checks under `extra`.

## Recomputing a number from the raw dumps

Ground rule: no number in a result file can fail to be recomputed from that
file plus the behaviour dump. Here is that claim, exercised: one `select`
result, the selections it recorded, and B-MaxSum and B-MaxMin rebuilt from the
dissimilarity matrix alone -- no counter, no planner, no pool.

```python
import json
from itertools import combinations
from pathlib import Path

run = Path('runs/default')
result = json.loads(next((run / 'results' / 'select').glob('*generic.json')).read_text())
model, pool = result['model'], result['pool']
dump = json.loads((run / 'behaviours' / model['hash'] / pool['domain']
                   / pool['instance'].split('/')[-1] / f"{pool['pool_stem']}.json").read_text())
matrix = dump['matrix']

for entry in result['extra']['selections']:
    behaviours = sorted(set(entry['distinct']))          # U_M(Psi)
    pairs = list(combinations(behaviours, 2))
    assert abs(sum(matrix[i][j] for i, j in pairs) - entry['values']['bmaxsum']) < 1e-9
    assert abs(min((matrix[i][j] for i, j in pairs), default=0.0) - entry['values']['bmaxmin']) < 1e-9
    print(entry['indicator'], entry['kappa'], entry['values'])
```

`tests/experiments/test_recompute.py` runs the same check, all four
indicators included, over every selection of the smoke sweep.

## Rebuilding the reports

`bdcexp report` regenerates everything under `reports/` from `results/` and
`behaviours/` alone. The rebuild is byte-for-byte identical except for
`manifest.json`, which records when it was written and the revision it was
written from; `tests/experiments/test_experiments_smoke.py` checks the rest.

## Size of the package

The brief budgets about 1500 lines of code in `experiments/`, excluding tests
and configs, counting non-blank, non-comment, non-docstring lines.

| Part | Code lines |
|---|---|
| Infrastructure (`config`, `benchmark`, `generate`, `pools`, `models`, `runner`, `report`, `cli`) | 1092 |
| `reference.py` (the Phase 0 arbiter) | 130 |
| The two task kinds (`select`, `timing`) | 75 |
| The three reports | 516 |
| **Total** | **1813** |

The package is over budget by about three hundred lines. The reports each
write the CSVs, tables and figures their subsection of the paper names, and
the infrastructure carries the resumable runner, the planner driver and the
manifest the plan asks for; nothing was left out.

## Known limits

* The stability model's behaviour is the plan's action set, not its sequence:
  the stability distance reads nothing else, so two orderings of one action
  set are one behaviour, as the paper's setup says. `b` under it can be far
  below the pool size; the case study's stability reading shows where.
* `logistics00` cannot be used: its domain file declares `(in ?obj ?obj)`, a
  repeated parameter name that unified-planning rejects. `logistics98` is the
  1998 STRIPS encoding of the same domain and is used instead.
* A `select` task replays its pool once per model rather than once per pool.
  With the default grid that is about twenty tasks per pool; each is short,
  and the array launcher runs them side by side.
* A `time` task clears the counter's private behaviour-distance cache before
  every selection sample so that each pays the `b^2` distances; there is no
  public reset.
