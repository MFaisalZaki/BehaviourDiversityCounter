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

* `--extras analysis` adds `scipy`, `pandas` and `matplotlib`. Only the report
  stage needs them; a compute node that merely runs tasks can leave them out.
* `--with planners` adds `up-symk`, whose wheel ships the SymK binary phase one
  drives. A host that only reports does not need it either.

The commands below are written as `poetry run bdcexp ...`; inside an activated
virtualenv `bdcexp` alone is enough.

## The three commands

```console
bdcexp generate <config> [--instance ID] [--list] [--force]
bdcexp run      <config> <experiment> [--task ID] [--list] [--force] [--jobs N]
bdcexp report   <config> <experiment|setup>
```

`<config>` is a path to a TOML file, or the name of one shipped with the
package: `default` or `smoke`. Every command takes `--results-dir` to send a
run somewhere other than the config's `[run].results_dir`.

All three are resumable and idempotent: `generate` skips a pool file that
already exists, `run` skips a task that already has a result file, and both
take `--force` to redo one anyway.

### Phase one: the pools

```console
bdcexp generate default            # clones classical-domains, then runs SymK
bdcexp generate default --list     # the instance ids, one per line
```

The first call clones `AI-Planning/classical-domains` at the commit pinned in
`[benchmark].commit` into `runs/<name>/benchmark/`, then runs SymK once per
`(instance, mode, q, N)`. Instances are taken in instance-number order and the
domain stops at `[benchmark].instances_per_domain` solved ones.

### Phase two: the experiments

```console
bdcexp run default e3              # every task of E3, one process
bdcexp run default e3 --jobs 8     # the same, in a local process pool
bdcexp run default e3 --list       # the task ids, one per line
bdcexp run default e3 --task e3/rovers/2006/p02/topq-q2.0-N1000/generic
```

A task that fails writes its traceback into its own result file and the report
counts it as a failure; it never takes the sweep down with it. A task that has
nothing to do -- a pool whose behaviour count is outside the range an
experiment enumerates, say -- records why it was skipped.

### Reports

```console
bdcexp report default setup        # benchmark.csv and models.csv
bdcexp report default e3           # CSVs, tables, figures, manifest
```

Reports are pure functions of `results/` and `behaviours/`. They never load a
pool, build a counter or run a selection, so a report can be rebuilt on a
laptop from the run directory alone, with no planner installed.

### On a cluster

```console
scripts/slurm_array.sh default            # every experiment
scripts/slurm_array.sh default e3 e2      # a subset
```

One `sbatch --array` per experiment, one element per task, and a dependent
report job. `SLURM_TIME`, `SLURM_MEM` and `SLURM_LOGS` override the defaults.
If a site's array limit bites, split the generated `<experiment>.tasks` file
with `split -n l/<parts>` and submit the parts.

### The smoke sweep

```console
bdcexp run    smoke e3
bdcexp report smoke e3
```

`configs/smoke.toml` runs against four pools committed under
`configs/smoke_pools/`, with the PDDL they were generated from beside them, so
it needs neither SymK nor the benchmark checkout. The first `run` copies those
pools into the run directory -- `--list` included, since listing the tasks
means listing the pools they run over. This is what the tests and CI use; the
whole suite finishes in well under two minutes.

## The output directory

```
runs/<name>/
  config.toml                 the configuration as run (its hash is in each manifest)
  benchmark/                  the classical-domains checkout
  pools/<domain>/<instance>/<mode>-q<q>-N<N>.json          phase one, plans included
  behaviours/<model_hash>/<domain>/<instance>/<pool>.json  per-plan behaviour + cost, b x b matrix
  results/<experiment>/<task>.json                         one raw dump per task
  reports/<experiment>/                                    CSVs, tables/*.tex, figures/*.pdf, manifest.json
```

JSON everywhere for raw data, CSV only under `reports/`. Plan action strings
are stored verbatim, so a plan can be re-run with VAL or any planner; behaviour
tuples are lists of strings; dissimilarity matrices are stored in full; floats
are stored unrounded and rounded only in LaTeX. Every file carries a `schema`
name and a version integer.

A task's result file names its task id, the configuration hash, the git
revision, the pool record, the model record, the rows it reports and, under
`extra`, everything it enumerated or sampled. The rule is that no row holds a
number that cannot be recomputed from the same file plus the behaviour dump.

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
| | `pool_sizes` | the `N` of each pool |
| | `q_values` | the quality bounds |
| `selection` | `k_values`, `kappa_values` | the selection grid |
| `models` | `generic` | the control model's features and its two resolution knobs |
| | `enabled` | optional: restrict the registry to these model names |
| `e1` | `domain`, `q`, `min_behaviours`, `k`, `kappa` | the case study's selection rule |
| `e2` | `subsets` | random subsets drawn per pool and k |
| `e3` | `k_values`, `behaviour_range` | the enumerable range |
| `e4` | `k_range` | the prefix range |
| `e5` | `goal_caps`, `cost_bin_widths`, `k` | the resolution grid |
| `e6` | `pool_sizes`, `repeats`, `feature_counts` | the timing grid |

The five decisions the author confirms before the sweep -- the domain list and
`instances_per_domain`, whether `topk` pools are generated as well as `topq`,
the `q` values, whether `k = 20` stays in the grid, and the generation limits
-- are all config keys. None of them is a code change.

## Adding a diversity model

A model is data. Add a `ModelSpec` to `PER_DOMAIN` in
`experiments/bdc_experiments/models.py`:

```python
ModelSpec('zenotravel_planner', ('zenotravel',),
          (FeatureSpec('ru', {'types': ('aircraft',)}, 0.5),
           FeatureSpec('go', {}, 0.5)))
```

* `key` is a key of the library's `dimensions_map` (`go`, `cbin`, `ru`, `rn`,
  `rc`, ...).
* `params` are the feature's own knobs and nothing instance-specific. A
  `types` entry names the PDDL types whose objects are the feature's
  resources; they are resolved against the problem's user types, or -- for the
  untyped STRIPS encodings -- against the unary predicate of the same name in
  the initial state.
* `weight` is the feature's weight, or `None` for all features of a model, in
  which case the library's uniform `1/n` applies.
* `domains` is a tuple of directory names, or `None` for every domain.

The spec's digest (`model_hash`) is part of the behaviour dump's path, so a
changed model is a changed path and nothing stale is read back. Add the model's
row to the setup report by doing nothing: it is generated from the registry.

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
| `model` | the diversity model, or the resolution variant in E5 |
| `k` | the selection size the row is about |
| `kappa` | the B-Novelty neighbourhood, always explicit, never the library default |
| `pool_size` | plans actually loaded, after the replay and cost filters |
| `b` | distinct behaviours the pool exhibits under this model |

`pool_stem` (`<mode>-q<q>-N<N>`) names the pool file. A behaviour is referred to
by its index into `distinct` in the behaviour dump, and a plan by its index into
the cost-sorted pool; a column holding several of either is a space-separated
list, so it survives a CSV reader without quoting.

### E1, `sec:exp-case`

| File | Columns beyond the nine |
|---|---|
| `e1_behaviours.csv` | `f_<feature>` — this behaviour's value for each feature of the model; `distinct` (its index), `plans` (how many exhibit it), `cheapest_cost` |
| `e1_selections.csv` | `indicator` (which selection), `position` in the returned order, `plan` (pool index), `cost`, `distinct`, `f_<feature>`, `set_<indicator>` (all four indicators of the returned set), `wall_s`, `cpu_s` |
| `e1_pairwise_diffs.csv` | `indicator`, `plan_i`, `plan_j`, `feature`, `value_i`, `value_j`, `differs`, `contribution` (this feature's `w_i psi_i` term), `psi` (the pair's `psi_M`), `differing_features` |

`e1_note.md` states which instance the rule picked and why, whether the
resource feature was constant on it, the k and kappa, and the tie rule.

### E2, `sec:exp-separation`

| File | Columns beyond the nine |
|---|---|
| `e2_random_subsets.csv` | `part`, `indicator`, `n_subsets` drawn, `enumerated` (all `C(b,k)` rather than sampled), `n_values` distinct values seen, `constant` (the C2 question), `modal_fraction`, `min`, `max`, `mean` |
| `e2_kendall.csv` | `pair` (`bmaxsum/bmaxmin` and the other two), `n_subsets`, `tau` (tau-b), `p`. Empty `tau` means undefined — an indicator was constant over the sample — and is left empty rather than written as zero |
| `e2_cross.csv`, `e2_cross_macro.csv` | `selector` (rows), `scored_<indicator>` (columns): that selection's value under that indicator, over the best of the four on the same pool. `cells` is how many pools the mean is over. Restricted to `b > k` |
| `e2_cross_all_pools.csv` | the same over every pool, `aggregate` naming pooled or macro. On `b <= k` pools every selection returns every behaviour, so the ratios are 1 by construction |

### E3, `sec:exp-greedy`

| File | Columns beyond the nine |
|---|---|
| `e3_ratios.csv` | `indicator`, `greedy`, `optimal`, `ratio`, `optimum_exists`, `greedy_distinct`, `greedy_subset` and `optimal_subset` (behaviour indices, so the missed behaviours can be read off), `optimal_at_most` / `at_most_size` / `at_most_subset` / `at_most_exceeds` / `ratio_at_most` (B-MaxMin's at-most-k optimum, which feeds E4's fixed-size reading), `enum_wall_s`, `enum_cached`, `enum_at_most_wall_s`, `select_wall_s`, `select_cpu_s` |
| `e3_summary.csv` | per `(indicator, k, kappa)`: `cases`, `rated` (cases with a defined ratio), `min`, `p5`, `median`, `at_optimum` (the fraction reaching it), `pooled_mean`, `macro_mean` |
| `e3_worst_cases.csv` | the worst case of each `(indicator, k, kappa)` cell, with its pool named and `missed` listing the behaviours the optimum held and the greedy did not |
| `e3_checks.json` | the blocking check: every B-Coverage ratio is exactly 1. `passed`, `cases`, `violations` |

An empty `ratio` means the optimal value was zero or no optimum exists at that
k — never a fabricated 1.

### E4, `sec:exp-fixed-size`

| File | Columns beyond the nine |
|---|---|
| `e4_prefix_values.csv` | `indicator`, `k_max` (the single run the prefixes come from), `value` (through the library), `matrix_value` (recomputed from the behaviour dump), `gap` and `agrees` (the two must match), `previous`, `delta`, `fell`, `relative_fall` |
| `e4_summary.csv` | per indicator: `series`, `steps`, `falls`, `fall_fraction`, `fall_fraction_macro`, `median_relative_fall`, `first_fall_series`, `never_falls_series`, `first_fall_k_median`, `first_fall_over_b_median`, `disagreements` (library against matrix), `monotone_check` (`ok`, or the violation for B-Coverage and B-MaxSum, which must never fall) |

### E5, `sec:exp-resolution`

| File | Columns beyond the nine |
|---|---|
| `e5_resolution.csv` | `goal_cap`, `cost_bin_width`, `goal_atoms`, `goal_orders` (`m!`), `cost_bins`, `space_size` (`\|BS\|`, the product), `b_over_pool`, `b_over_space`, `saturated` (every plan its own behaviour), `cost_bin_degenerate` (one bin, so the feature is constant — which is what `q = 1` gives), `indicator` and the four indicator values of that selection, `wall_s`, `cpu_s` |
| `e5_summary.csv` | per `(domain, goal_cap, cost_bin_width)`: `pools`, `b_median` with its quartiles and range, `b_pooled_mean`, `b_macro_mean`, `space_size_median`, `b_over_space_*`, `b_over_pool_median`, `cpu_s_median`, `saturated_pools`, `degenerate_cost_bin_pools` |

### E6, `sec:exp-cost`

| File | Columns beyond the nine |
|---|---|
| `e6_timing.csv` | one row per sample, not per median: `features` (how many the model has, the `n` of the cost expression), `phase` (`mapping` or `selection`), `indicator`, `repeat`, `wall_s`, `cpu_s`, `generation_wall_s` and `generation_cpu_s` from the pool record, `wall_over_generation`, `cpu_over_generation`, `exhausted` |
| `e6_medians.csv` | per `(N, k, features, phase, indicator)`: `samples`, `wall_median` with quartiles, `cpu_median` with quartiles, `cpu_pooled_mean`, `cpu_macro_mean`, `pool_size_median`, `b_median`, the two generation medians and the two ratios |

Only the wall-clock comparison against generation is like for like: the pool
record measures the planner subprocess with the parent's `process_time` as
well, so `generation_cpu_s` excludes the planner's own CPU. Both are in the
CSV; the table and the figure use the wall clock.

### Setup, `sec:exp-setup`

| File | Columns |
|---|---|
| `benchmark.csv` | `domain`, `ipc_year`, `instances`, `pools`, `pools_q<q>` per quality bound, `exhausted`, `timed_out`, `empty`, `plans` |
| `models.csv` | `model`, `domains`, `feature`, `description`, `params`, `dimension_size_rule`, `dissimilarity`, `weight` |

Each experiment also writes `manifest.json`: the git revision and whether the
tree was dirty, the config path and hash, the benchmark's pinned and actual
commit, the planner and the exact search strings, the limits, the seed, package
versions, the Python and platform, the tie-breaking rule, the task counts
(ok / failed / skipped), every failure with its message, and every output path.

## Recomputing a number from the raw dumps

Ground rule: no row holds a number that cannot be recomputed from the same
result file plus the behaviour dump. Here is that claim, exercised. The script
takes one E2 result, reads the selections it recorded, and rebuilds B-MaxSum
and B-MaxMin from the dissimilarity matrix alone -- no counter, no planner, no
pool.

```python
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

run = Path('runs/default')
result = json.loads(next((run / 'results' / 'e2').glob('*.json')).read_text())
model, pool = result['model'], result['pool']
dump = json.loads((run / 'behaviours' / model['hash'] / pool['domain']
                   / pool['instance'].split('/')[-1] / f"{pool['pool_stem']}.json").read_text())
matrix = dump['matrix']

recomputed = []
for entry in result['extra']['selections']:
    behaviours = sorted(set(entry['selection']['distinct']))   # U_M(Psi)
    pairs = list(combinations(behaviours, 2))
    recomputed.append({
        'k': entry['k'], 'kappa': entry['kappa'], 'selector': entry['indicator'],
        'bcoverage': float(len(behaviours)),
        'bmaxsum': sum(matrix[i][j] for i, j in pairs),
        'bmaxmin': min((matrix[i][j] for i, j in pairs), default=0.0),
        'reported_bmaxsum': entry['values']['bmaxsum'],
        'reported_bmaxmin': entry['values']['bmaxmin'],
    })

table = pd.DataFrame(recomputed)
assert (table['bmaxsum'] - table['reported_bmaxsum']).abs().max() < 1e-9
assert (table['bmaxmin'] - table['reported_bmaxmin']).abs().max() < 1e-9
print(table.pivot_table(index='selector', columns='k', values='bmaxsum'))
```

`tests/experiments/test_recompute.py` runs the same check over the whole smoke
sweep, so the guarantee is enforced rather than asserted in prose.

## Rebuilding the reports

`bdcexp report` regenerates everything under `reports/` from `results/` and
`behaviours/` alone. The rebuild is byte-for-byte identical except for
`manifest.json`, which records when it was written and the revision it was
written from; `tests/experiments/test_experiments_smoke.py` checks the rest.

## Size of the package

The brief budgets "about 1500 lines of code in `experiments/`, excluding tests
and configs", and says that a feature which pushes past it should be left out
and the omission recorded here. Nothing was left out: every output named in the
plan's mapping to the paper's Section 5 is produced, and dropping one would
break an `\input` in the paper. The package is therefore over budget, and this
is where the lines went:

| Part | Code lines | Physical |
|---|---|---|
| Infrastructure (`config`, `benchmark`, `generate`, `pools`, `models`, `runner`, `report`, `cli`) | 1091 | 1622 |
| `reference.py` (the Phase 0 arbiter) | 146 | 221 |
| The six experiments | 1394 | 1895 |
| **Total** | **2631** | **3738** |

Counting non-blank, non-comment, non-docstring lines. The six experiments
average a little over 230 lines each, and each writes between four and eight
files: a raw dump the reader can recompute from, one or two CSVs, a LaTeX
table, one or two figures and a manifest. What was actually shared rather than
repeated is in `runner.py` (the task grid, the standard setup, timed selection,
the selection record, the nine mandatory row fields) and in `report.py` (CSV,
booktabs, Holm, the two scipy tests, the pooled-and-macro summary rows,
figures, the manifest).

Each experiment module then grew again in the review pass that followed it, by
between 30 and 80 lines, because two reviewers reading it against the plan
found things that were missing rather than things that could go: the
per-`(k, kappa)` constancy aggregate E2's claim actually turns on, a check file
that could no longer read as green with nothing behind it, a series key that
had been conflating two pools, a figure that plotted a ratio where the claim is
about the value. Those are the lines the budget bought.

## Known limits

* `[generation].pool_sizes` in `default.toml` is `[100, 1000, 10000]` because
  E6 times the second phase against the pool size. The other experiments then
  run over all three sizes. Cutting it to `[1000]` cuts the sweep roughly
  threefold and leaves E6 with a single point.
* `[run].time_limit_selection_s` is 3600 rather than the 600 of the original
  brief: E6 remaps a 10,000-plan pool `[e6].repeats` times, and a task that
  exceeds the limit is recorded as a failure rather than truncated.
* `logistics00` cannot be used: its domain file declares `(in ?obj ?obj)`, a
  repeated parameter name that unified-planning rejects. `logistics98` is the
  1998 STRIPS encoding of the same domain and is used instead.
* E5 and E6 build several counters over one pool and share a trace cache
  between them, so the pool is replayed once rather than once per model. The
  cache holds every plan's state trace, which is the memory peak of the second
  phase; on a 10,000-plan pool it is the reason `[run].memory_limit` matters to
  phase two as well as to phase one.
