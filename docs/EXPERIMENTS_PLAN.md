# Implementation plan: the empirical evaluation of "Behaviour Spaces for Diversity Planning"

Repository: `MFaisalZaki/BehaviourDiversityCounter`, branch `claude-exp-implementation`.
Written for the agent that will implement it. Read all of it before touching a file.
Revision of 2026-09-14: three experiments (a three-domain case study that also reads the
literature's plan-level distance next to the user's features, separation of equal-count
sets, cost of the second phase) and the selection grid `k in {3, 5, 10}`. The paper's `sections/05-experiments.tex`
of the same date is written to this plan.

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
   An experiment that does not trace to one of them is out of scope. In particular, the
   paper makes **no claim about the selection functions** (they are reused from the
   literature), so nothing measures how close a selection comes to an optimum, and no
   report ranks the selection functions against each other.
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
   behaviour-dissimilarity matrix of the pool under the model, every random subset
   drawn, every timing sample. Reports are pure functions of those dumps and never
   touch a pool or a counter. Section 9 gives the layout; the whole `runs/<name>/`
   directory is the artefact that ships with the paper.

## 1. What the paper claims and what each experiment tests

The paper (Sections 2 to 4 plus Table 1) makes the claims below. The experiment on the
right is the only place that claim is tested. Section numbers are given for the author's
benefit; do not rely on them, the paper is in revision.

| # | Claim | Where | Experiment |
|---|---|---|---|
| C1 | Two behaviours are compared componentwise, so the features on which two plans differ can be read off | Motivation, Sec. 4.1 | E1 case study |
| C2 | The formulation is solved the same way with the literature's model and the user's, and a plan-level distance collapses differences the user cares about | Sec. 1, Sec. 2, Def. 3.6 | E1 case study, stability reading |
| C3 | The three dissimilarity-based indicators separate plan sets that B-Coverage scores alike, for every kappa | Table 1 row "Separates equal-count sets" | E2 separation |
| C4 | All four indicators ignore a plan whose behaviour is already held (twinning) | indicator theorems, clause (ii) | plans-per-behaviour statistic inside E1 |
| C5 | The representation's cost is knowledge engineering, not computation; indicator cost is `O(k n c_ext + b^2 n c_dist)` | Abstract, Sec. 4 closing, appendix cost paragraphs | E3 cost |

Where an experiment also produces a check of a stated property (B-Coverage is constant
over equal-count subsets; the stability model gives one behaviour per distinct action
set), the report flags any violation prominently. A violation is a finding about the
library or the paper and must not be smoothed over.

## 2. The paper's definitions, restated for implementation

These are the facts the code must match. `M` is a diversity model, `Psi` a plan set.

- **Feature** `f = (Delta, extract, psi, w)`: a value set, an extracting function from
  plans to `Delta`, a dissimilarity `psi: Delta x Delta -> [0,1]` with `psi(x,y)=0` iff `x=y`,
  symmetry, and a weight `w in (0,1]` (Def. 3.3 requires definiteness of every
  dissimilarity).
- **Diversity model** `M = <f_1..f_n>`, weights sum to 1. Model dissimilarity
  `psi_M(a,b) = sum_i w_i psi_i(extract_i(a), extract_i(b))`.
- **Feature-based model**: every `Delta_i` is a finite set of values of a user criterion.
- **Stability model** (the literature's model, Section 7.3): one feature whose extractor
  returns the plan's action set, whose dimension is the set of action sets, and whose
  dissimilarity is the stability distance `1 - |A(a) & A(b)| / |A(a) | A(b)|`, weight 1.
  It is definite on action sets, so two plans share a behaviour iff they have the same
  action set, and `extract_BMaxSum` under it is the post-hoc greedy of Katz and Sohrabi
  (2020). It is not a metric-based model in the sense of Def. 3.6, because that would need
  a distance definite on plans and stability is zero on two orderings of one action set.
- **Behaviour** of a plan: the tuple `(extract_1(pi), ..., extract_n(pi))`.
  `B_M(Psi)` = set of distinct behaviours. `U_M(Psi)` = one plan per behaviour.
- **B-Coverage** `= |B_M(Psi)|`.
- **B-MaxSum** `= sum over unordered pairs of U_M(Psi) of psi_M`; 0 below two behaviours.
- **B-MaxMin** `= min over unordered pairs of U_M(Psi) of psi_M`; 0 below two behaviours.
- **B-Novelty(kappa)**: the mean over plans in `U_M(Psi)` of the mean dissimilarity to their
  neighbourhood, the `min(kappa, |U_M(Psi)| - 1)` closest plans of `U_M(Psi)`; 0 when
  `|U_M(Psi)| < 2`. Ties among neighbours broken arbitrarily.
- **Definiteness**: every `psi_i` is zero only on equal values (Def. 3.3), so
  `psi_M(a,b) = 0` iff same behaviour. The audit checks every model, including `stability`.
- **Two-phase scheme**: phase one, any planner produces a pool `C` of plans with cost
  `<= c`; phase two, `extract_lambda(M, C, k)` selects at most `k` plans.
- **extract_BCoverage** (MAP-Elites per-cell retention): one pass over `C`; keep the
  cheapest plan per behaviour, up to `k` behaviours; if fewer than `k` behaviours exist,
  fill with arbitrary plans.
- **extract_BMaxSum** (Katz and Sohrabi's greedy): if `k < 2` or `|C| < 2` return any
  `min(k,|C|)` plans. Open on the pair maximising `psi_M`. Then repeatedly add the
  candidate maximising B-MaxSum of the combined set (its gain: sum of `psi_M` to the
  selected plans if its behaviour is new, else 0). Ties arbitrary.
- **extract_BMaxMin** (Ravi's farthest-first): same opening; then add the candidate
  maximising `min over U_M(Psi) of psi_M(candidate, held)`. Duplicates score 0.
- **extract_BNovelty** (the Katz and Sohrabi greedy with B-Novelty as objective): same
  opening; while `k` not reached and some behaviour of `C` is not held, add the candidate
  with a new behaviour maximising B-Novelty of the combined set; then fill with arbitrary
  plans.
- All three dissimilarity-based selections return the `k` plans selected, never a
  shorter prefix, even when the indicator fell during selection.

Tie-breaking is "arbitrary" in the paper. The implementation must make it deterministic
and record the rule (lowest index in the cost-sorted pool). State this in the report.

## 3. What exists in the repository and how to treat it

| Path | Status | Treatment |
|---|---|---|
| `behaviour_diversity_counter/behaviour_diversity_counter.py` | indicators, `extract`, caches | Use. Audit in Phase 0. Note `DEFAULT_K_NN = 3`; the paper fixes no kappa, so **every experiment passes `k_nn` explicitly** and never relies on the default. |
| `behaviour_diversity_counter/dimensions/*.py` | `go`, `cbin`, `rn`, `ru`, `rc`, `uv`, `fn`, `cb` | Use `go`, `cbin`, `rn`, `ru`. Add `stability.py` (Section 7.3). Docstrings cite theorem names that no longer exist in the paper; leave them, the author will sync. |
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
  reference.py     brute-force reference implementations of the indicators and selections (Phase 0)
  runner.py        task ids, run one task with error capture, list tasks, load dumps
  e1_case_study.py       each: run_task(task, cfg) -> dict, report(dumps, cfg, out) -> None
  e2_separation.py
  e3_cost.py
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
ref_extract_bcoverage / ref_extract_bmaxsum / ref_extract_bmaxmin / ref_extract_bnovelty
    (plans as (index, cost, behaviour) triples, d, k, kappa) -> list of indices
    written literally from the Section 2 pseudo-code, with lowest-index tie-break
ref_stability(actions_a, actions_b) -> float          # 1 - Jaccard over action sets
```

There is no optimum enumerator: the paper makes no optimality claim, so nothing needs it.

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
7. Every model in `models.py`, `stability` included: each per-dimension dissimilarity
   returns 0 only on equal values, checked over every pair of values its dimension takes on the
   smoke pools.
8. Stability model (Section 7.3) on the transport fixture: `b` equals the number of
   distinct action sets; `psi_M` equals `ref_stability` on every pair  `extract_BMaxSum`
   under it equals `ref_extract_bmaxsum` with `ref_stability` as `d`.

Known things to look at closely during the audit:

- `_extract_greedy` ranks only candidates with a new behaviour ("fresh"); the paper's
  B-MaxSum ranks all candidates by the combined-set value. Under definiteness the two
  agree. Confirm with test 2 and note the equivalence in the audit note.
- `best_index` rounds scores to 3 decimals before argmax. Check this cannot merge two
  genuinely different scores in the spaces used here (dissimilarities are rationals with
  small denominators, so it should not). Under the stability model the stability
  values are rationals with denominators up to the union size, so verify on a smoke
  pool of a few hundred plans as well.
- `b_novelty` breaks neighbour ties by `np.partition`; the paper says arbitrary. Fine,
  but the reference must produce the same value for any tie order (it does, the mean
  over tied neighbours is the same); assert that.
- The counter's `b x b` matrix caching: under the stability model `b` is the number of
  distinct action sets, up to the pool size of 1000, so the matrix has a million entries. Measure the time and memory of
  one selection on a 1000-plan smoke pool. If it exceeds one minute or 200 MB, the
  stability selections use `reference.ref_extract_bmaxsum` with `ref_stability`
  directly (validated by test 8), and the manifest records which path was used.

Deliverable: `docs/AUDIT.md` listing each check, its outcome, any library fix made (with
the commit), the tie-breaking rule in one sentence, and the stability-model path decision.

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
modes = ["topq"]                       # top-quality with bound q * c*
pool_sizes = [1000]                    # N: plans requested per pool
q_values = [1.0, 2.0]

[selection]
k_values = [3, 5, 10]
kappa_values = [1, 2, 3]

[models]
generic = { features = ["goal_order", "cost_bin"], goal_cap = 6, cost_bin_width = 0.1 }
stability = { }
# per-domain models are declared in models.py; this section only sets their knobs

[e1]
domains = ["rovers", "logistics", "satellite"]
min_behaviours = 6

[e2]
random_subsets = 200
weight_settings = [[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]]   # rovers_astronaut only

[e3]
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
top-quality planners as the intended pool generators and SymK provides both. Only
top-quality pools are generated; no claim needs top-k pools.

- Prefer the binary that the `up-symk` wheel ships (it is already an optional Poetry
  group). If the wheel does not expose the search-string interface needed, build SymK
  from source at a pinned tag into `runs/<name>/symk/` and record the tag.
- Invoke SymK through `subprocess` with an explicit search string, one call per
  `(instance, mode, q, N)`. The search string is, per the SymK README, top-quality with
  relative bound: `symq-bd(plan_selection=top_k(num_plans=N), quality=q)`.
  **Verify the exact syntax against the README of the pinned SymK version before use**
  and record the string used in the manifest.
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
  This matters for the case study: two orderings of one action set are twins under the stability
  model by design, but an exact duplicate sequence would be counted twice in `pool_size`.
- `optimal_cost`: for `topq` the first plan is optimal.

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
- **Stability model exception.** Its `b x b` matrix has one row per distinct action set,
  up to `|C| x |C|`, a million floats per pool, so the dump for that model stores the behaviour (the action tuple's
  index, which is the plan index) and cost per plan, and **no matrix**. Every E1 stability result
  dumps the `k x k` stability matrix of each selected set and the action strings, which
  is enough to recompute every stability number from the result file alone.
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
domains, one generic model on every domain as a control, and the literature's model,
stated as one feature, which turns the formulation into its post-hoc selection. All registered in
`models.py` as specs:

```python
@dataclass(frozen=True)
class FeatureSpec:  key: str; params: dict; weight: float | None
@dataclass(frozen=True)
class ModelSpec:    name: str; domains: tuple[str, ...] | None; features: tuple[FeatureSpec, ...]
def build_counter(spec: ModelSpec, task, instance_info, trace_cache) -> BehaviourDiversityCounter
def model_hash(spec) -> str
def domain_model(domain) -> ModelSpec      # the per-domain model, else `generic`
```

### 7.1 Per-domain models (the astronaut and their colleagues)

| Model | Domain | Feature 1 | Feature 2 | Weights |
|---|---|---|---|---|
| `rovers_astronaut` | rovers | number of rovers used (`rn`, discrete dissimilarity) | order in which the `communicated_*` goal atoms are first achieved (`go`, Hamming / count) | 1/2, 1/2 |
| `logistics_dispatcher` | logistics | vehicles used (`ru` over trucks and airplanes) | delivery order (`go` over `at package location` goals) | 1/2, 1/2 |
| `driverlog_dispatcher` | driverlog | drivers used (`ru` over drivers) | delivery order (`go`) | 1/2, 1/2 |
| `satellite_operator` | satellite | satellites used (`ru`) | image order (`go` over `have_image` goals) | 1/2, 1/2 |

The `rovers_fine` variant (which rovers rather than how many) is not used.

`rovers_astronaut` is the one model that E2 also builds under the weight settings of
`[e2.weight_settings]`; the spec's weights are parameters, and each setting has its own
`model_hash`.

The resource objects of each instance are read from the PDDL problem by type name
(`rover`, `truck`, `airplane`, `driver`, `satellite`), not from any external declaration
file. Write a small helper that emits the `(:resource ...)` declaration text the `ru`/`rn`
dimensions expect from the task's objects of the named types. Record the object list in
the result's `model` block.

### 7.2 Generic model (every domain)

`generic`: goal ordering (`go`, cap `goal_cap`) + cost bin (`cbin`, width
`cost_bin_width`, over `[1, q]`), equal weights; both knobs are fixed in `[models]`. At `q = 1.0` the cost bin is constant;
say so in the report rather than dropping the feature silently.

### 7.3 Stability model (every domain)

`stability`: one feature from a new dimension module `dimensions/stability.py`. Its
extracting function returns the plan's action set as a frozenset, its dimension is the
set of action sets (finite, since the action set of the task is), and its dissimilarity is
`ref_stability`, one minus the Jaccard measure, which is the stability distance of
Srivastava et al. (2007) and the metric Katz and Sohrabi (2020) evaluate; weight 1. It is
definite on action sets, as Def. 3.3 requires, so two plans are twins under it exactly
when they have the same action set; the paper says so in the setup, and E2 reports how
often that happens in the case study. Only the stability distance is implemented;
`uniqueness` and the state distance are out of scope unless the author asks.

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
  sampled (subsets, timing samples). A row never holds a number that cannot
  be recomputed from the same file plus the behaviour dump. Reports read the result
  JSONs and the behaviour dumps and nothing else.

### E1 case study (C1, C2, C4)

- Instances: for each domain in `[e1.domains]`, the smallest instance whose `topq` pool at
  `q = 2.0` exhibits at least `[e1.min_behaviours]` behaviours under the domain model
  **and** at least two behaviours on the agents feature. If none exists, take the smallest
  with `min_behaviours` behaviours and state that the agents feature is constant on it.
  Selection is automatic from the pools, and the report prints the rule and which
  instance it picked per domain.
- For each instance, `k = 3`, `kappa = 1`, under the domain model: table of the pool's
  behaviours with counts and cheapest cost; table of the plans each of the four selections
  returns, their behaviours, and the four indicator values of each returned set; for each
  pair of returned plans, the list of features on which they differ (C1).
- The plans-per-behaviour counts are C4 on each instance: the report names how many of
  the pool's plans the model treats as the same, and shows two plans from the fullest
  cell with their action strings so a reader sees what the features do not record.
- Stability reading (C2), same three pools: run `extract_BMaxSum` under the `stability`
  model for `k = 3` (this is the Katz and Sohrabi selection). Read the returned set under
  the domain model: its behaviours, the number of distinct behaviours it covers out of
  `min(k, b)`, and which of its pairs share a behaviour. Read the domain-model selections
  under `stability`: their pairwise stability values. Dump both selections' indices,
  action strings and behaviours and the `k x k` stability matrices. The report states
  the numbers and does not rank the two models; the paper says diversity is the user's.
- Output: `e1_behaviours.csv`, `e1_selections.csv`, `e1_pairwise_diffs.csv`,
  `e1_stability.csv`, one LaTeX table per instance for the selections and one for the
  pairwise differences, and `e1_note.md` stating which instances, why, and the tie rule.

### E2 separation of equal-count sets (C3)

Part A, selection-free. For every pool and model with `b > k`:
- Draw up to `[e2.random_subsets]` random subsets of exactly `k` distinct behaviours
  (seeded).
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

Part C, weights, rovers only:
- Repeat Part B under `rovers_astronaut` at each setting in `[e2.weight_settings]`.
- Report, per pair of settings and per indicator: the Jaccard similarity of the two
  returned behaviour sets, and the indicator values. This says whether the weights
  change the set or only its value. Note that `bcoverage` never reads the weights, so
  its rows are a check (identical sets), not a result.

Outputs: `e2_random_subsets.csv`, `e2_kendall.csv`, `e2_cross.csv`, `e2_cross_macro.csv`,
`e2_weights.csv`, LaTeX tables, one figure (tau distributions per pair, `k` and `kappa`).

### E3 cost of the second phase (C5)

- Pools of `N` in `[e3.pool_sizes]` (generate them in phase one; where SymK exhausts
  the instance before `N`, record the actual size).
- For each pool, model and `k` in `[selection.k_values]`: time (a) replay + mapping into
  the behaviour space, (b) each selection, `[e3.repeats]` times, on an otherwise idle
  core; report medians. Timing must use `time.process_time` and `perf_counter` both.
- Compare against the pool's generation time from the pool file.
- Dump every individual timing sample, not only the medians.
- Vary `n`, the number of features, by running the generic model with one, two and
  (where a per-domain model exists) three features, to see the `n` factor.
- No curve fitting. The figure plots the samples against `pool_size` and against `b`
  on log axes with the two cost terms drawn as reference slopes; the reader judges.
- The report does not rank the four selections by speed; the paper makes no claim
  about the selection functions.
- Outputs: `e3_timing.csv`, table of medians per `(N, k)`, figure of mapping time versus
  generation time and selection time versus `b`.

## 9. Phase 4: raw data layout, statistics and reports

### 9.1 Raw data layout

Everything lives under `runs/<name>/` and that directory is the artefact:

```
runs/<name>/
  config.toml                 the config as run, plus its hash in the manifest
  benchmark/                  the classical-domains checkout (git submodule or clone)
  pools/<domain>/<instance>/<mode>-q<q>-N<N>.json          phase one output, plans included
  behaviours/<model_hash>/<domain>/<instance>/<pool>.json  per-plan behaviour + cost, b x b matrix (none for `stability`)
  results/<experiment>/<task_id>.json                      one raw dump per task (Section 8)
  reports/<experiment>/                                    CSVs, tables/*.tex, figures/*.pdf, manifest.json
```

Rules: JSON everywhere for raw data (readable by anyone, diffable, no schema library);
CSV only in `reports/`; plan action strings stored verbatim so a plan can be re-run
with VAL or any planner; behaviour tuples stored as lists of strings; dissimilarity
matrices stored in full except for the stability model (Section 6.4); floats stored
unrounded; every file self-describing with a `schema` field naming the experiment and a
version integer. `reports/` must be fully regenerable by `bdcexp report` from
`results/` and `behaviours/` alone. Provide one script-sized example in
`docs/EXPERIMENTS.md` that recomputes one table with `pandas` from the raw JSON, to
prove the point.

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
  The `stability` model appears in `models.csv` like any other, with dimension "action
  sets" and dissimilarity "stability".

## 10. Phase 5: tests

`tests/experiments/`, all runnable without SymK and without the benchmark checkout:

- `test_audit.py` (Phase 0, tests 1 to 8).
- `test_models.py`: the resource declaration helper picks objects by type; every
  registered model, including `stability` and each weight setting of `rovers_astronaut`,
  builds a counter on the transport fixture or on the bundled smoke pools;
  `model_hash` is stable across processes and differs across weight settings.
- `test_pools.py`: pool file round-trip; replay failure is listed, not raised; cost
  filter; sort order; deduplication count; cache hit on second load; no matrix in the
  `stability` dump.
- `test_experiments_smoke.py`: each experiment runs on `smoke.toml` end to end and its
  report writes every declared output; every check (B-Coverage constant on E2 random
  subsets, `b` = number of distinct action sets under `stability` in E1) passes on the
  smoke pools.
- `test_recompute.py`: for one smoke result of E1 and one of E2, recompute every row's
  values from the result file and the behaviour dump alone and compare; this is the
  guarantee behind ground rule 8, and for E1 it proves the `k x k` matrix dump suffices.
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
   passes or the library fixes are in. The stability-model path decision (Section 5, last
   bullet) is made here.
2. Phase 1 infrastructure with `smoke.toml` and bundled smoke pools; `bdcexp run
   configs/smoke.toml e2` green end to end.
3. Phase 2 models; tests green.
4. Phase 3 experiments in the order E2, E3, E1 (E2 first because its random subsets and
   four selections exercise the whole stack against the reference code; E1 last because
   it adds the stability model and depends on the pools existing).
5. Phase 4 reports; Phase 5 tests complete; Phase 6 docs.
6. Full sweep on the cluster; reports built; manifests present.

Done means: every experiment has a manifest with zero unexplained failures, every
check has no blocking violation, `docs/EXPERIMENTS.md` reproduces the sweep from a
clean clone, `bdcexp report` regenerates `reports/` byte-for-byte from `results/` and
`behaviours/`, `experiments/` is within the line budget of ground rule 7, and `pytest`
is green on 3.10 and 3.13. Commit in small steps with messages naming the phase.

## 13. Mapping to the paper's Section 5

The paper's `sections/05-experiments.tex` is written to this structure. Each report
must produce the outputs its subsection names, with the column names below, so the
LaTeX tables can be generated and dropped in without renaming.

| Paper subsection | Label | Experiment | Outputs the subsection consumes |
|---|---|---|---|
| Setup | `sec:exp-setup` | all | `manifest.json` of every experiment (planner, search string, limits, hardware, versions); `benchmark.csv`; `models.csv` |
| Case Study: Reading the Differences | `sec:exp-case` | E1 | `e1_behaviours.csv`, `e1_selections.csv`, `e1_pairwise_diffs.csv`, `e1_stability.csv`, `e1_note.md` |
| Separating Equal-Count Sets | `sec:exp-separation` | E2 | `e2_random_subsets.csv`, `e2_kendall.csv`, `e2_cross.csv`, `e2_cross_macro.csv`, `e2_weights.csv` |
| Cost of the Second Phase | `sec:exp-cost` | E3 | `e3_timing.csv` |
| Discussion | `sec:exp-discussion` | all | the check results and the failure counts of every manifest |

Table and figure files are written as `tables/<experiment>_<name>.tex` and
`figures/<experiment>_<name>.pdf`, with the label `tab:<experiment>-<name>` or
`fig:<experiment>-<name>` inside, so that the paper can `\input` a table and
`\includegraphics` a figure by a name that does not change between sweeps.

## 14. Decisions the author has made, and those still open

Made (2026-09-14): `k in {3, 5, 10}`; `kappa in {1, 2, 3}`; `q in {1.0, 2.0}`; top-quality
pools only; the literature's model is the stability distance, as a one-feature model; the
weight run is on rovers only; no fixed-size or resolution experiment.

Still open, to confirm before the sweep (not before coding):

1. The domain list and `instances_per_domain = 15`.
2. Time and memory limits for generation on the target cluster.
3. Whether `N = 10000` pools for E3 are affordable on the cluster; if not, `[100, 1000]`
   and say so in the paper.

Code everything so that these are config changes, not code changes.
