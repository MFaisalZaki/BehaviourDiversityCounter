# Implementation plan: the empirical evaluation of "Behaviour Spaces for Diversity Planning"

Repository: `MFaisalZaki/BehaviourDiversityCounter`, branch `claude-exp-implementation`.
Written for the agent that will implement it. Read all of it before touching a file.

## 0. Ground rules

1. **Build the evaluation from scratch.** Nothing under `paper-experiments/` may be
   imported, copied, adapted, or read for design. That includes `paperexps/*.py`,
   `exp-cfg-files/default.json`, `setup_benchmark.sh`, `data/fi-generated-plans-dir.zip`,
   `data/ru-info-dir/`, and `data/patches/`. Leave that directory untouched; the author
   will remove it separately. Do not add the new code under it. Do not reuse its pool
   file naming, its result schema, its config keys, or its experiment numbering.
2. **The library is the paper's artefact and is used, after an audit.** The package
   `behaviour_diversity_counter/` implements the behaviour space, the four indicators and
   the four selection functions. Phase 0 verifies it against the paper's definitions with
   independently written reference code. Fix the library where the audit finds a
   mismatch; otherwise call it, do not reimplement it inside the experiments.
3. **Every experiment answers one sentence of the paper.** Section 1 lists the sentences.
   An experiment that does not trace to one of them is out of scope.
4. **Nothing is fabricated.** A task that fails writes its traceback into its own result
   file and the report counts it as a failure. A number that cannot be computed is
   reported as missing, never as zero.
5. **Everything is reproducible from the config, the seed, and the commit.** Every report
   writes a manifest naming the git revision, package versions, planner version and
   build, config file hash, and start and end times.
6. Python 3.10 to 3.13, Poetry. The library's runtime dependencies stay as they are;
   the experiments may add `scipy`, `pandas` and `matplotlib` under an optional group.
   Do not put analysis dependencies in the library's default group.
7. **Minimal code.** This is a sweep, not a framework. No plugin registries, no abstract
   base classes, no configuration inheritance, no generated shell scripts beyond one
   short array launcher, no caching layer beyond one JSON file per pool and model.
   One flat package, one module per experiment with two plain functions
   (`run_task` and `report`), one runner, one CLI with three subcommands. Budget:
   about 1500 lines of code in `experiments/` excluding tests and configs. If a
   feature pushes past that, leave it out and say so in the docs. Prefer the standard
   library; `pandas` is for reports only, never for the run stage.
8. **Dump raw data, recompute freely.** Every task writes the raw material a reader
   would need to recompute every number in the paper without rerunning a planner or a
   selection: the plans selected (indices and action strings), their behaviours, the
   full behaviour-dissimilarity matrix of the pool under the model, every random subset
   drawn, every enumerated optimum, every timing sample. Reports are pure functions of
   those dumps and never touch a pool or a counter. Section 9 gives the layout; the
   whole `runs/<name>/` directory is the artefact that ships with the paper.

## 1. What the paper claims and what each experiment tests

The paper (Sections 3 and 4 plus Table 1) makes the claims below. The experiment on the
right is the only place that claim is tested. Section numbers are given for the author's
benefit; do not rely on them, the paper is in revision.

| # | Claim | Where | Experiment |
|---|---|---|---|
| C1 | Two behaviours are compared componentwise, so the features on which two plans differ can be read off | Motivation, Sec. 4.1 | E1 case study |
| C2 | The three dissimilarity-based indicators separate plan sets that B-Coverage scores alike | Table 1 row "Separates equal-count sets" | E2 |
| C3 | Greedy selection is exact for B-Coverage; the other three selection functions are heuristics, and the paper states no approximation bound for them | Sec. 4.2, Table 1 caption | E3 |
| C4 | B-MaxMin and B-Novelty are not monotone in plans, so they are read at a fixed set size | Sec. 4.2, Table 1 rows 1 and 3 | E4 |
| C5 | Feature resolution decides what the space distinguishes; the cell count grows as the product of dimension sizes | Motivation last paragraph, Sec. 4.1 | E5 |
| C6 | The representation's cost is knowledge engineering, not computation; indicator cost is `O(k n c_ext + b^2 n c_dist)` | Abstract, Sec. 4 closing, appendix cost paragraph | E6 |

Where an experiment also produces a check of a stated property (every B-Coverage
greedy ratio equals 1, B-Coverage and B-MaxSum never fall as plans are added), the
report flags any violation prominently. A violation is a finding about the library or the
paper and must not be smoothed over.

## 2. The paper's definitions, restated for implementation

These are the facts the code must match. `M` is a diversity model, `Psi` a plan set.

- **Feature** `f = (Delta, extract, psi, w)`: a finite value set, an extracting function
  from plans to `Delta`, a dissimilarity `psi: Delta x Delta -> [0,1]` with `psi(x,x)=0` and
  symmetry, and a weight `w in (0,1]`.
- **Diversity model** `M = <f_1..f_n>`, weights sum to 1. Model dissimilarity
  `psi_M(a,b) = sum_i w_i psi_i(extract_i(a), extract_i(b))`.
- **Behaviour** of a plan: the tuple `(extract_1(pi), ..., extract_n(pi))`.
  `B_M(Psi)` = set of distinct behaviours. `U_M(Psi)` = one plan per behaviour.
- **B-Coverage** `= |B_M(Psi)|`.
- **B-MaxSum** `= sum over unordered pairs of U_M(Psi) of psi_M`; 0 below two behaviours.
- **B-MaxMin** `= min over unordered pairs of U_M(Psi) of psi_M`; 0 below two behaviours.
- **B-Novelty(kappa)**: with `b = |U_M(Psi)|`, `kappa' = min(kappa, b-1)`, the mean over
  plans in `U_M(Psi)` of the mean dissimilarity to their `kappa'` nearest neighbours in
  `U_M(Psi)`; 0 when `b < 2`. Ties among neighbours broken arbitrarily.
- **Definiteness assumption**: `psi_M(a,b) = 0` iff same behaviour. Holds when every
  `psi_i` is zero only on equal values.
- **Two-phase scheme**: phase one, any planner produces a pool `C` of plans with cost
  `<= c`; phase two, `extract_lambda(M, C, k)` selects at most `k` plans.
- **extract_BCoverage**: one pass over `C`; keep the cheapest plan per behaviour, up to
  `k` behaviours; if fewer than `k` behaviours exist, fill with arbitrary plans. Exact.
- **extract_BMaxSum**: if `k < 2` or `|C| < 2` return any `min(k,|C|)` plans. Open on the
  pair maximising `psi_M`. Then repeatedly add the candidate maximising B-MaxSum of the
  combined set (its gain: sum of `psi_M` to the selected plans if its behaviour is new,
  else 0). Ties arbitrary. A heuristic; the paper states no bound.
- **extract_BMaxMin**: same opening; then farthest-first, add the candidate maximising
  `min over U_M(Psi) of psi_M(candidate, held)`. Duplicates score 0. A heuristic; the
  paper states no bound.
- **extract_BNovelty**: same opening; while `k` not reached and some behaviour of `C` is
  not held, add the candidate with a new behaviour maximising B-Novelty of the combined
  set; then fill with arbitrary plans. A heuristic; the paper states no bound.
- All three dissimilarity-based selections return the `k` plans selected, never a
  shorter prefix, even when the indicator fell during selection.

Tie-breaking is "arbitrary" in the paper. The implementation must make it deterministic
and record the rule (lowest index in the cost-sorted pool). State this in the report.

## 3. What exists in the repository and how to treat it

| Path | Status | Treatment |
|---|---|---|
| `behaviour_diversity_counter/behaviour_diversity_counter.py` | indicators, `extract`, caches | Use. Audit in Phase 0. Note `DEFAULT_K_NN = 3`; the paper fixes no kappa, so **every experiment passes `k_nn` explicitly** and never relies on the default. |
| `behaviour_diversity_counter/dimensions/*.py` | `go`, `cbin`, `rn`, `ru`, `rc`, `uv`, `fn`, `cb` | Use `go`, `cbin`, `rn`, `ru`. Docstrings cite theorem names that no longer exist in the paper; leave them, the author will sync. |
| `behaviour_diversity_counter/simulation.py` | replay, cost | Use. |
| `tests/conftest.py` | tiny transport task fixture | May be imported by new tests. |
| `tests/test_golden.py` | paper's worked examples on a stub counter | Keep passing. Extend in Phase 0 if the audit adds worked examples. |
| `paper-experiments/` | previous harness and pools | **Do not import, read, or modify.** |
| `pyproject.toml` | includes `paperexps` and the `bdcevalcli` script | Add the new package and script; do not remove the old entries, the author will. |

## 4. Target layout

```
experiments/                       new top-level package, name `bdc_experiments`
  __init__.py
  config.py        load the TOML into a plain dict, validate keys, hash it (~60 lines)
  benchmark.py     fetch classical-domains; enumerate (domain, instance) with problem+domain paths
  generate.py      phase one: run SymK per (instance, q, N); write one pool file
  pools.py         pool file format; load with replay, cost filter, sort; behaviour dump per model
  models.py        the model specs as data (a dict per model) and one build function
  reference.py     brute-force reference implementations of indicators and optima (Phase 0)
  runner.py        task ids, run one task with error capture, list tasks, load dumps
  e1_case_study.py       each: run_task(task, cfg) -> dict, report(dumps, cfg, out) -> None
  e2_separation.py
  e3_greedy_vs_optimum.py
  e4_fixed_size.py
  e5_resolution.py
  e6_cost.py
  report.py        write CSV, write a booktabs table, save a figure, write a manifest (~120 lines)
  cli.py           `bdcexp generate | run | report` (~80 lines)
  configs/
    default.toml
    smoke.toml     one tiny instance per domain, seconds to run, used by tests and CI
    smoke_pools/   a few KB of committed pools so tests never need SymK
scripts/
  slurm_array.sh   ~30 lines: one array per experiment from `bdcexp run --list`
tests/
  experiments/     tests for everything above; runs on smoke.toml without SymK
docs/
  EXPERIMENTS_PLAN.md   this file
  EXPERIMENTS.md        how to run, what each output is (written in Phase 6)
```

No `stats.py`: the report modules call `scipy.stats.wilcoxon`, `scipy.stats.kendalltau`
and a five-line Holm directly. No `tasks.py` separate from the runner. No experiments
subpackage. If a helper is used by one module only, it lives in that module.

## 5. Phase 0: audit the library against Section 2

Goal: independent evidence that `BehaviourDiversityCounter` computes the paper's
definitions, before any experiment depends on it. Write `experiments/reference.py` first,
without reading the library's implementation of the same function.

Reference functions, all pure, operating on a list of behaviour tuples and a
dissimilarity callable `d(b1, b2)`:

```
ref_bcoverage(behaviours) -> int
ref_bmaxsum(behaviours, d) -> float
ref_bmaxmin(behaviours, d) -> float
ref_bnovelty(behaviours, d, kappa) -> float
ref_optimum(behaviours, d, k, indicator, kappa) -> (value, argmax subset)   # exhaustive over subsets of exactly k
ref_optimum_at_most(behaviours, d, k, indicator, kappa)                     # over subsets of size 1..k
ref_extract_bcoverage / ref_extract_bmaxsum / ref_extract_bmaxmin / ref_extract_bnovelty
    (plans as (index, cost, behaviour) triples, d, k, kappa) -> list of indices
    written literally from the Section 2 pseudo-code, with lowest-index tie-break
```

Tests (`tests/experiments/test_audit.py`):

1. Random behaviour spaces (2 to 4 dimensions, 2 to 6 values each, random metric and
   random non-metric definite dissimilarities, 500 seeds): library indicator equals
   reference indicator to 1e-9 for every random plan set, with and without duplicate
   behaviours, for kappa in {1, 2, 3, 5}.
2. Same spaces: library `extract(..., indicator)` returns the same **behaviour set** as
   the reference extraction for every indicator and k in 1..6. Compare behaviour sets,
   not plan indices, and separately assert the returned set's indicator value equals the
   reference's. If they differ only by tie-breaking, document the library's rule and
   make the reference match it; if they differ in value, the library is wrong.
3. Paper's worked examples (already in `tests/test_golden.py`) still pass.
4. Twinning: adding a plan with a held behaviour leaves all four indicators unchanged.
5. Monotonicity in plans holds for B-Coverage and B-MaxSum on every random set; a
   counterexample exists for B-MaxMin and B-Novelty in the random sample (report the
   first found).
6. `extract` for B-MaxMin and B-Novelty returns exactly `min(k, |C|)` plans even when
   the indicator fell during selection.

Known things to look at closely during the audit:

- `_extract_greedy` ranks only candidates with a new behaviour ("fresh"); the paper's
  B-MaxSum ranks all candidates by the combined-set value. Under definiteness the two
  agree. Confirm with test 2 and note the equivalence in the audit note.
- `best_index` rounds scores to 3 decimals before argmax. Check this cannot merge two
  genuinely different scores in the spaces used here (dissimilarities are rationals with
  small denominators, so it should not, but verify on the E3 pools too).
- `b_novelty` breaks neighbour ties by `np.partition`; the paper says arbitrary. Fine,
  but the reference must produce the same value for any tie order (it does, the mean
  over tied neighbours is the same); assert that.

Deliverable: `docs/AUDIT.md` listing each check, its outcome, any library fix made (with
the commit), and the tie-breaking rule in one sentence.

## 6. Phase 1: infrastructure

### 6.1 Config (`config.py`, `configs/default.toml`)

One TOML file drives everything. Load it with `tomllib` into a dict, check the keys
against a hard-coded set, hash the file. No dataclasses, no schema library.

```toml
[run]
seed = 20260913
results_dir = "runs/default"          # everything the sweep writes goes under here
time_limit_generation_s = 1800
memory_limit_generation_mb = 8000
time_limit_selection_s = 600

[benchmark]
source = "https://github.com/AI-Planning/classical-domains.git"
commit = "<pin a commit hash>"
domains = ["rovers", "logistics", "driverlog", "satellite", "depot", "zenotravel", "tpp", "gripper", "blocks"]
instances_per_domain = 15             # first 15 by instance number that phase one solves

[generation]
planner = "symk"
modes = ["topq"]                       # top-quality with bound q * c*; "topk" as alternative
pool_sizes = [1000]                    # N: plans requested per pool
q_values = [1.0, 2.0]

[selection]
k_values = [5, 10, 20]
kappa_values = [1, 2, 3]

[models]
generic = { features = ["goal_order", "cost_bin"], goal_cap = 6, cost_bin_width = 0.1 }
# per-domain models are declared in models.py; this section only sets their knobs

[e3]
k_values = [3, 4, 5]
behaviour_range = [6, 24]              # pools with b in this range are enumerable

[e4]
k_range = [2, 20]

[e5]
goal_caps = [3, 4, 5, 6, 7, 8]
cost_bin_widths = [0.05, 0.1, 0.25]

[e6]
pool_sizes = [100, 1000, 10000]
repeats = 5
```

### 6.2 Benchmark (`benchmark.py`)

- Clone `classical-domains` at the pinned commit into `runs/<name>/benchmark/`.
- Enumerate instances by parsing each domain directory's `api.py` with `ast`, never by
  importing it. Resolve the domain file per problem (some directories ship one domain
  file per problem).
- Instance id: `<domain>/<ipc-year>/<problem-file-stem>`. Use the file name, not a
  position in a list, so the id survives a benchmark update.

### 6.3 Generation (`generate.py`)

Phase one of the two-phase scheme. Planner: SymK, because the paper names top-k and
top-quality planners as the intended pool generators and SymK provides both.

- Prefer the binary that the `up-symk` wheel ships (it is already an optional Poetry
  group). If the wheel does not expose the search-string interface needed, build SymK
  from source at a pinned tag into `runs/<name>/symk/` and record the tag.
- Invoke SymK through `subprocess` with an explicit search string, one call per
  `(instance, mode, q, N)`. The search strings are, per the SymK README:
  top-quality with relative bound: `symq-bd(plan_selection=top_k(num_plans=N), quality=q)`;
  top-k: `symk-bd(plan_selection=top_k(num_plans=N))`.
  **Verify the exact syntax against the README of the pinned SymK version before use**
  and record the strings used in the manifest.
- Run under `ulimit`-style limits from `[run]`; a timeout is a recorded outcome, not an
  error. Keep partial output: SymK writes `sas_plan.1`, `sas_plan.2`, ... as it goes.
- Pool file: `runs/<name>/pools/<domain>/<instance>/<mode>-q<q>-N<N>.json`

```json
{
  "instance": "rovers/2002/p05", "domain_file": "...", "problem_file": "...",
  "planner": {"name": "symk", "version": "...", "search": "..."},
  "mode": "topq", "q": 2.0, "requested": 1000,
  "optimal_cost": 22, "optimal_cost_source": "symk first plan (optimal)",
  "wall_s": 123.4, "cpu_s": 120.1, "timed_out": false, "exhausted": true,
  "plans": [ {"actions": ["(navigate rover0 w0 w1)", "..."], "cost": 22}, ... ]
}
```

- `exhausted = true` when SymK proved fewer than `N` plans exist within the bound.
- Deduplicate plans by action sequence at write time and record the count dropped.
- `optimal_cost`: for `topq` the first plan is optimal; for `topk` run `symk-opt` once
  per instance or take the cheapest of the pool and say so in `optimal_cost_source`.

### 6.4 Pool loading and the behaviour cache (`pools.py`)

- `load_pool(path, task)`: parse every plan with unified-planning, replay it with
  `simulation.simulate`, drop and list plans that fail replay, drop and list plans above
  `q * optimal_cost` (should be none for `topq`), sort by `(cost, original index)`.
- Plan name normalisation: SymK/Fast Downward lowercases and replaces `-` with `_` and
  may suffix `_n`; resolve against the task's action and object names, and fail loudly
  on an ambiguous token rather than guessing.
- Behaviour dump: a plan's behaviour under a model depends only on the model and the
  plan, so the first task to map a pool under a model writes
  `runs/<name>/behaviours/<model_hash>/<domain>/<instance>/<pool-stem>.json` holding,
  for every plan index, its behaviour tuple and cost, plus the `b x b` dissimilarity
  matrix over the distinct behaviours (as a list of lists, with the behaviour order).
  Later tasks read it instead of replaying. This file is both the cache and the raw
  data that lets anyone recompute every indicator by hand. Miss means recompute;
  there is no invalidation logic, the hash in the path is the invalidation.
- Every loaded pool records `size`, `dropped_replay`, `dropped_cost`, `duplicates`,
  `optimal_cost`, and timing of parse and replay separately.

### 6.5 Tasks, manifests, runner (`runner.py`, `cli.py`)

- Task id: `<experiment>/<instance>/<pool-stem>` (+ model name where an experiment runs
  several). One task writes one result JSON under `runs/<name>/results/<experiment>/`.
- Result file skeleton:

```json
{"task_id": "...", "experiment": "e2", "config_hash": "...", "git": "...",
 "started": "...", "ended": "...", "error": null,
 "pool": {...}, "model": {...}, "rows": [ ... ], "extra": {...}}
```

- `bdcexp` subcommands, three only:
  - `bdcexp generate <config> [--instance ID] [--list]`: fetches the benchmark if
    missing, then phase one; resumable.
  - `bdcexp run <config> <experiment> [--task ID] [--list] [--force] [--jobs N]`: phase
    two; resumable, one result file per task, traceback captured; `--jobs` runs tasks in
    a local process pool.
  - `bdcexp report <config> <experiment|setup>`: CSVs, tables, figures, manifest, from
    dumps only.
  The smoke run is `bdcexp run configs/smoke.toml <experiment>` against the committed
  smoke pools; no separate subcommand.
- `scripts/slurm_array.sh`: pipes `bdcexp run --list` into one `sbatch --array` per
  experiment, one element per task, and a dependent report job. About 30 lines, no
  prompts, no chunking logic; if a site's array limit bites, split the list with `split`.

## 7. Phase 2: diversity models

The paper's thesis is that features are user-defined. So the evaluation uses
domain-specific models written by the authors acting as domain expert for a subset of
domains, plus one generic model on every domain as a control. All registered in
`models.py` as specs:

```python
@dataclass(frozen=True)
class FeatureSpec:  key: str; params: dict; weight: float | None
@dataclass(frozen=True)
class ModelSpec:    name: str; domains: tuple[str, ...] | None; features: tuple[FeatureSpec, ...]
def build_counter(spec: ModelSpec, task, instance_info, trace_cache) -> BehaviourDiversityCounter
def model_hash(spec) -> str
```

### 7.1 Per-domain models (the astronaut and their colleagues)

| Model | Domain | Feature 1 | Feature 2 | Weights |
|---|---|---|---|---|
| `rovers_astronaut` | rovers | number of rovers used (`rn`, discrete dissimilarity) | order in which the `communicated_*` goal atoms are first achieved (`go`, Hamming / count) | 1/2, 1/2 |
| `rovers_fine` | rovers | which rovers used (`ru`, Jaccard) | same as above | 1/2, 1/2 |
| `logistics_dispatcher` | logistics | vehicles used (`ru` over trucks and airplanes) | delivery order (`go` over `at package location` goals) | 1/2, 1/2 |
| `driverlog_dispatcher` | driverlog | drivers used (`ru` over drivers) | delivery order (`go`) | 1/2, 1/2 |
| `satellite_operator` | satellite | satellites used (`ru`) | image order (`go` over `have_image` goals) | 1/2, 1/2 |

The resource objects of each instance are read from the PDDL problem by type name
(`rover`, `truck`, `airplane`, `driver`, `satellite`), not from any external declaration
file. Write a small helper that emits the `(:resource ...)` declaration text the `ru`/`rn`
dimensions expect from the task's objects of the named types. Record the object list in
the result's `model` block.

### 7.2 Generic model (every domain)

`generic`: goal ordering (`go`, cap `goal_cap`) + cost bin (`cbin`, width
`cost_bin_width`, over `[1, q]`), equal weights. At `q = 1.0` the cost bin is constant;
say so in the report rather than dropping the feature silently.

### 7.3 Resolution knobs (E5)

`goal_cap` and `cost_bin_width` are the two resolution knobs. A model spec carries them,
so E5 builds several specs of the generic model and the cache keys them separately.

## 8. Phase 3: the experiments

Shared conventions:

- Selection input for `k` is the whole pool (up to `N` plans) unless the experiment says
  otherwise. Record `pool_size` and `b` (distinct behaviours of the pool) on every row.
- All four selections run on the same pool, same model, same `kappa`.
- Every row carries `instance`, `domain`, `q`, `N`, `model`, `k`, `kappa`, `pool_size`, `b`.
- Report aggregates two ways: **pooled** over all rows and **macro** (mean of per-domain
  means), because domains contribute unequal numbers of instances. Tables show both.
- Every paired comparison is paired on `(instance, q, N, model, k, kappa)`.
- **Raw dump contract.** Each `run_task` returns a dict that is written verbatim as
  the task's result JSON. Besides the rows, it always contains: the pool record; the
  model record (features, weights, resource objects, caps); for every selection made,
  the indices of the selected plans in pool order, their action strings, their
  behaviours and costs, and the wall and CPU time; and whatever the experiment
  enumerated or sampled (subsets, optima, prefixes). A row never holds a number that
  cannot be recomputed from the same file plus the behaviour dump. Reports read the
  result JSONs and the behaviour dumps and nothing else.

### E1 case study (C1)

- Instance: the smallest rovers instance whose `topq` pool at `q = 2.0` exhibits at
  least six behaviours under `rovers_astronaut` **and** at least two behaviours on the
  rover-count feature. If none exists, take the smallest with six behaviours and state
  that the rover-count feature is constant on it; then repeat under `rovers_fine`.
  Selection is automatic from the pools, and the report prints the selection rule and
  which instance it picked.
- For `k = 3`, `kappa = 1`: table of the pool's behaviours with counts and cheapest
  cost; table of the plans each of the four selections returns, their behaviours, and
  the four indicator values of each returned set.
- For each pair of returned plans, the list of features on which they differ. This is
  the componentwise reading of C1 and is the point of the experiment.
- Output: `e1_behaviours.csv`, `e1_selections.csv`, `e1_pairwise_diffs.csv`, the two
  LaTeX tables, and a `e1_note.md` stating which instance, why, and the tie rule.

### E2 separation of equal-count sets (C2)

Part A, selection-free. For every pool and model with `b > k`:
- Draw up to 200 random subsets of exactly `k` distinct behaviours (seeded).
- Score each under all four indicators (B-Novelty at each `kappa`). Dump every subset
  as its list of behaviour indices into the pool's behaviour order, with its scores.
- Report: fraction of subsets on which each indicator is constant (B-Coverage must be 1.0
  by construction; state it as a check, not a result); Kendall tau-b between every pair of
  the three dissimilarity-based indicators' rankings, median and IQR per `(k, kappa)`;
  fraction of pools with negative tau.

Part B, selection-based. On the same pools:
- Run the four selections; score each returned set under all four indicators.
- Cross table: rows = selecting indicator, columns = scoring indicator, entry = value
  divided by the best of the four selections on that pool, averaged pooled and macro.
- Restrict to pools with `b > k` in the main table; report the all-pools version in the
  appendix CSV with a note that on `b <= k` pools all selections return every behaviour.

Outputs: `e2_random_subsets.csv`, `e2_kendall.csv`, `e2_cross.csv`, `e2_cross_macro.csv`,
LaTeX tables, one figure (tau distributions per pair and `k`).

### E3 greedy against the optimum (C3)

The paper states that greedy selection is exact for B-Coverage and gives the other
three selection functions as heuristics without a bound. This experiment checks the
first and measures the second: how far, on real pools, each heuristic falls from the
optimum it targets. The numbers describe the heuristics; they neither prove nor assume
a bound, and the report says so.

- Pools and models with `b` in `[e3.behaviour_range]`, `k` in `[e3.k_values]`,
  `kappa` in `[selection.kappa_values]`. Enumerate the optimum with
  `reference.ref_optimum` over subsets of exactly `k` behaviours (binomial(24,5) = 42504,
  cheap); also `ref_optimum_at_most` for B-MaxMin.
- Ratio = greedy value / optimal value, per indicator. Record enumeration time. Dump
  the optimal subset (behaviour indices) and the greedy subset for every case, so a
  reader can see which behaviours the heuristic missed.
- Models: `generic` on every pool in range; the per-domain models where they apply.
- Check written to `e3_checks.json`: every B-Coverage ratio is exactly 1. A violation is
  a blocking issue: stop and report, it means the library or the pool loader is wrong.
- For the other three, report the distribution of the ratio (min, 5th percentile,
  median, fraction of cases at the optimum) per indicator, `k` and `kappa`, and the
  worst case with its pool named, so the author can look at it.
- Also record, for B-MaxMin, how often the at-most-`k` optimum exceeds the exactly-`k`
  optimum, and the ratio of the greedy set of `k` to the at-most-`k` optimum. This feeds
  the fixed-size reading of E4.
- Outputs: `e3_ratios.csv`, summary table, `e3_worst_cases.csv`, box-plot figure.

### E4 the fixed-size reading of B-MaxMin and B-Novelty (C4)

- For every pool and model, run each of the four selections for `k = 2..k_max`
  (`e4.k_range`, capped at `b`), on the same pool. Because the greedy procedures are
  prefix-consistent (step `k+1` extends step `k`), one run to `k_max` gives every prefix;
  verify this once in a test and then compute from the single run.
- Dump the full selection order (plan indices) once per selection; the indicator value
  of every prefix is computed from the behaviour dump at report time and also stored
  in the rows for convenience.
- Report: fraction of `k -> k+1` steps at which the value falls, per indicator; median
  relative fall; the `k` at which B-MaxMin first falls below its opening value, relative
  to `b`. B-Coverage and B-MaxSum must never fall: check.
- Outputs: `e4_prefix_values.csv`, summary table, one figure of value against `k` for a
  few representative pools plus the aggregate.

### E5 resolution (C5)

- Generic model on every pool at every `goal_cap` in `[e5.goal_caps]` and every
  `cost_bin_width` in `[e5.cost_bin_widths]` (the width only matters at `q > 1`).
- Record per spec: `|BS|` (product of dimension sizes as the paper defines it, where the
  goal-order dimension size is `cap!` and the cost dimension size is the bin count),
  `b` exposed by the pool, `b / pool_size`, the four indicator values of each selection at
  `k = 10`, and selection CPU time.
- Report: `b` against cap per domain (median with IQR); `b / |BS|`; selection time
  against `b` on log axes (this is the `b^2` term of C6 seen empirically); the point at
  which every plan becomes its own behaviour, if any pool reaches it.
- Outputs: `e5_resolution.csv`, tables, two figures.

### E6 cost of the second phase (C6)

- Pools of `N` in `[e6.pool_sizes]` (generate them in phase one; where SymK exhausts
  the instance before `N`, record the actual size).
- For each pool, model and `k` in `[selection.k_values]`: time (a) replay + mapping into
  the behaviour space, (b) each selection, `[e6.repeats]` times, on an otherwise idle
  core; report medians. Timing must use `time.process_time` and `perf_counter` both.
- Compare against the pool's generation time from the pool file.
- Dump every individual timing sample, not only the medians.
- Also vary `n`, the number of features, by running the generic model with one, two and
  (where a per-domain model exists) three features, to see the `n` factor.
- No curve fitting. The figure plots the samples against `pool_size` and against `b`
  on log axes with the two cost terms drawn as reference slopes; the reader judges.
- Outputs: `e6_timing.csv`, table of medians per `(N, k)`, figure of mapping time versus
  generation time and selection time versus `b`.

## 9. Phase 4: raw data layout, statistics and reports

### 9.1 Raw data layout

Everything lives under `runs/<name>/` and that directory is the artefact:

```
runs/<name>/
  config.toml                 the config as run, plus its hash in the manifest
  benchmark/                  the classical-domains checkout (git submodule or clone)
  pools/<domain>/<instance>/<mode>-q<q>-N<N>.json          phase one output, plans included
  behaviours/<model_hash>/<domain>/<instance>/<pool>.json  per-plan behaviour + cost, b x b matrix
  results/<experiment>/<task_id>.json                      one raw dump per task (Section 8)
  reports/<experiment>/                                    CSVs, tables/*.tex, figures/*.pdf, manifest.json
```

Rules: JSON everywhere for raw data (readable by anyone, diffable, no schema library);
CSV only in `reports/`; plan action strings stored verbatim so a plan can be re-run
with VAL or any planner; behaviour tuples stored as lists of strings; dissimilarity
matrices stored in full; floats stored unrounded; every file self-describing with a
`schema` field naming the experiment and a version integer. `reports/` must be fully
regenerable by `bdcexp report` from `results/` and `behaviours/` alone. Provide one
script-sized example in `docs/EXPERIMENTS.md` that recomputes one table with `pandas`
from the raw JSON, to prove the point.

### 9.2 Statistics and reports

- Use `scipy.stats.wilcoxon` (two-sided, zero differences dropped) and
  `scipy.stats.kendalltau` (tau-b) directly. Holm is five lines in `report.py`, tested
  against known values. No bootstrap intervals.
- Every table reports `n` (pairs or pools), median, IQR, and where a test is run the
  Holm-adjusted p-value and the fraction of pairs with a non-zero difference (a
  significant test on a zero median is common here and must be legible).
- `report.py`: CSV first, LaTeX `booktabs` second, figures third (matplotlib, PDF,
  colour-blind-safe palette, no title text inside the figure). One manifest per
  experiment with git revision, package versions, planner tag and search strings, config
  hash, timestamps, task counts (ok / failed / skipped), and every output path.
- Never round in CSVs; round only in LaTeX.
- `bdcexp report <config> setup` writes the two files the paper's Setup subsection
  consumes: `benchmark.csv` (domain, IPC year, instances used, pools per `q`, pools the
  planner exhausted, pools that timed out) and `models.csv` (model, domains, feature,
  dimension and its size rule, dissimilarity, weight), plus the LaTeX table of each.

## 10. Phase 5: tests

`tests/experiments/`, all runnable without SymK and without the benchmark checkout:

- `test_audit.py` (Phase 0).
- `test_models.py`: the resource declaration helper picks objects by type; every registered model builds a counter on the transport fixture
  or on the bundled smoke pools; `model_hash` is stable across processes.
- `test_pools.py`: pool file round-trip; replay failure is listed, not raised; cost
  filter; sort order; cache hit on second load.
- `test_experiments_smoke.py`: each experiment runs on `smoke.toml` end to end and its
  report writes every declared output; all checks in `e3_checks.json` pass on the smoke
  pools.
- `test_e4_prefix_consistency.py`: a run to `k_max` equals the runs to each `k`.
- `test_recompute.py`: for one smoke result, recompute every row's indicator values from
  the behaviour dump alone and compare; this is the guarantee behind ground rule 8.
- Holm against known values, inside `test_report.py`.

CI: `poetry run pytest` on 3.10 and 3.13, with `--extras analysis`. The smoke suite must
finish in under two minutes.

## 11. Phase 6: documentation

- `docs/EXPERIMENTS.md`: install, fetch, generate, run, report, slurm; the config
  reference; the output catalogue (every CSV column defined); how to add a model.
- `docs/AUDIT.md` from Phase 0.
- Update the top-level `README.md` with one paragraph pointing at both. Do not describe
  `paper-experiments/` there.

## 12. Order of work and definition of done

1. Phase 0 audit, with `docs/AUDIT.md`. Blocking: nothing else starts until the audit
   passes or the library fixes are in.
2. Phase 1 infrastructure with `smoke.toml` and bundled smoke pools; `bdcexp run
   configs/smoke.toml e3` green end to end.
3. Phase 2 models; tests green.
4. Phase 3 experiments in the order E3, E2, E4, E5, E6, E1 (E3 first because its
   enumeration against the reference code exercises the whole stack end to end; E1
   last because it depends on the pools existing).
5. Phase 4 reports; Phase 5 tests complete; Phase 6 docs.
6. Full sweep on the cluster; reports built; manifests present.

Done means: every experiment has a manifest with zero unexplained failures, every
check file has no blocking violation, `docs/EXPERIMENTS.md` reproduces the sweep from a
clean clone, `bdcexp report` regenerates `reports/` byte-for-byte from `results/` and
`behaviours/`, `experiments/` is within the line budget of ground rule 7, and `pytest`
is green on 3.10 and 3.13. Commit in small steps with messages naming the phase.

## 13. Mapping to the paper's Section 5

The paper's `sections/05-experiments.tex` is written to this structure. Each report
must produce the outputs its subsection names, with the column names below, so the
LaTeX tables can be generated and dropped in without renaming.

| Paper subsection | Label | Experiment | Outputs the subsection consumes |
|---|---|---|---|
| Setup | `sec:exp-setup` | all | `manifest.json` of every experiment (planner, search strings, limits, hardware, versions); `benchmark.csv` (domain, ipc year, instances, pools per q, exhausted pools); `models.csv` (model, domain, feature, dimension size rule, dissimilarity, weight) |
| Case Study: Reading the Differences | `sec:exp-case` | E1 | `e1_behaviours.csv`, `e1_selections.csv`, `e1_pairwise_diffs.csv`, `e1_note.md` |
| Separating Equal-Count Sets | `sec:exp-separation` | E2 | `e2_random_subsets.csv`, `e2_kendall.csv`, `e2_cross.csv`, `e2_cross_macro.csv` |
| Greedy Selection Against the Optimum | `sec:exp-greedy` | E3 | `e3_ratios.csv`, `e3_worst_cases.csv`, `e3_checks.json` |
| Reading B-MaxMin and B-Novelty at a Fixed Size | `sec:exp-fixed-size` | E4 | `e4_prefix_values.csv` and its summary |
| Feature Resolution | `sec:exp-resolution` | E5 | `e5_resolution.csv` |
| Cost of the Second Phase | `sec:exp-cost` | E6 | `e6_timing.csv` |
| Discussion | `sec:exp-discussion` | all | the check files and the failure counts of every manifest |

Table and figure files are written as `tables/<experiment>_<name>.tex` and
`figures/<experiment>_<name>.pdf`, with the label `tab:<experiment>-<name>` or
`fig:<experiment>-<name>` inside, so that the paper can `\input` a table and
`\includegraphics` a figure by a name that does not change between sweeps.

## 14. Decisions the author must confirm before the sweep (not before coding)

1. The domain list and `instances_per_domain = 15`.
2. Whether `topk` pools are generated as well as `topq` (cost: double generation time).
3. The `q` values; `[1.0, 2.0]` is the default written above.
4. Whether `k = 20` stays in the grid given that many pools have `b < 20`.
5. Time and memory limits for generation on the target cluster.

Code everything so that these are config changes, not code changes.
