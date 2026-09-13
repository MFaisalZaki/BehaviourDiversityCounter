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
pools into the run directory. This is what the tests and CI use; the whole
suite finishes in well under two minutes.

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
