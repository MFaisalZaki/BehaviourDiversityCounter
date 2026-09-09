# The paper's empirical evaluation

The experiments of `EXPERIMENTS_BRIEF.md` (E0 to E6), run over pools of plans
stored on disk. The harness consumes pools; it does not produce them. Every
number in the tables is reproducible from `paperexps/exp-cfg-files/default.json`,
the seed it carries, and the scripts here.

## Layout

```
paperexps/
  runnner.py                 run one task, every task, list the tasks, or --report
  utils.py                   pool file -> (domain, problem, resource declarations)
  harness.py                 load a pool (replay, cost-bound, sort by cost), run the task list
  model.py                   the feature-based model per domain; the rovers case-study model
  baseline.py                stability / state / uniqueness and the plan-level greedy (E1)
  stats.py                   Wilcoxon signed-rank, Holm, Kendall's tau-b (stdlib only)
  report.py                  CSVs, booktabs tables, figures (matplotlib optional), manifests
  exp_e0_case_study.py       E0   ...   exp_e6_sensitivity.py  E6
  gen_pools.py               pool generation with forbid-iterative ("kind": "generate" entries)
  exp-cfg-files/default.json the grid, limits, paths, the generation sweep and per-experiment parameters
  sandbox/                   built by ../setup_benchmark.sh: pools, benchmark, ru-info, results
data/
  fi-generated-plans-dir.zip the forbid-iterative pools ({q}-{k}-{track}-{year}-{domain}-{inst}-fi-bc-results.json)
  ru-info-dir/               per-domain (:resource ...) declarations: the agent type of each domain
setup_benchmark.sh           venv, benchmark checkout, pool unpacking, slurm arrays or local runs
```

Result files land in each experiment's `dump-dir` (one JSON per pool); the
report stage writes `results/E?_*.csv`, `tables/E?_*.tex`, `figures/E?_*.pdf`
and `results/E?_manifest.json` (versions, seed, parameters, start and end).

## Reproducing

```bash
poetry install --extras analysis          # scipy for the tests, matplotlib for the figures
cd paper-experiments
./setup_benchmark.sh --planners --yes     # venv (+ forbid-iterative), classical-domains, unpack the pools, build the arrays
./setup_benchmark.sh --submit --yes       # ...and submit them (slurm)
./setup_benchmark.sh --local-jobs 4 --yes # ...or run them here, reports included
```

With `--submit` the whole sweep is queued from this one command, in the order
the dependencies need: the generation arrays (`GEN-fi-q2`) run first; a chain
job with `--dependency=afterany` on them re-runs the script for the evaluation
experiments once the pools exist, since their task lists are the pool files;
each evaluation array is followed by its report job. An experiment with no
pools on disk is skipped with a warning rather than aborting the sweep.
`--experiment E1,E2` narrows the selection; `--skip-existing` resumes.

```bash

cd paperexps
python runnner.py --config-file exp-cfg-files/default.json --experiment-name E1 --list-tasks
python runnner.py --config-file exp-cfg-files/default.json --experiment-name E1 --task-id 1.0-10-classical-2002-rovers-1-fi-bc
python runnner.py --config-file exp-cfg-files/default.json --experiment-name E1            # every task, resumable
python runnner.py --config-file exp-cfg-files/default.json --experiment-name E1 --report   # CSVs, tables, figures
```

Every task writes one result file and is skipped when that file exists
(`--force` reruns it), so a partial sweep resumes by resubmitting;
`setup_benchmark.sh --skip-existing` rebuilds the array of what is missing.
A task that raises writes its traceback into its result file and the report
counts it as a failure; nothing is fabricated for it.

## Generating pools for a quality bound

`GEN-fi-q2` in the configuration is a generation sweep (`"kind": "generate"`):
one task per (domain, year, instance, k) over the brief's domain list, each
running forbid-iterative once and writing a pool file in the evaluation's
own layout, `{q}-{k}-classical-{year}-{domain}-{inst}-{tag}-results.json`.

```bash
cd paperexps
python runnner.py --config-file exp-cfg-files/default.json --experiment-name GEN-fi-q2 --list-tasks
python runnner.py --config-file exp-cfg-files/default.json --experiment-name GEN-fi-q2 --task-id 2.0-10-classical-2002-rovers-1-fi-bc
```

| key | meaning |
|---|---|
| `generator` | `fi` (`--planner diverse`, N plans), `topk` (`--planner topk`, the N cheapest), `topq` (`--planner topq_via_topk --quality-bound q`) |
| `tag` | the generator tag in the file name; `fi-bc` by default so the evaluation's `generators` filter matches |
| `q` | the quality bound; the pool is filtered to cost ≤ q · c* |
| `pool-k-values`, `pool-factor`, `pool-cap` | one pool per k, asking the planner for min(factor · k, cap) plans |
| `time-limit`, `memory-limit` | passed as `--overall-time-limit` / `--overall-memory-limit` (30m, 8G per the brief); `subprocess-timeout-s` is the hard stop |
| `fi-command` | how to invoke the planner, `python3 -m forbiditerative.plan` by default (`setup_benchmark.sh --planners` installs it) |

`topq` knows the bound itself. For `fi` and `topk` the optimal cost c* comes
from a q = 1.0 pool of the instance already on disk (replayed, so exact) and
otherwise from one extra `--planner topk --number-of-plans 1` call; the pool's
`info` records the source, the bound, how many plans were found and filtered,
the planner's status and command, and the last lines of its output. Plans
are read from the Fast Downward plan files the planner leaves in its working
directory (`found_plans/done/sas_plan.*`, with their `; cost = N` comment);
the harness then validates every plan by replaying it when a pool is loaded.
A failed or timed-out call writes a pool with no plans and the failure in
`info`, so it is counted, never fabricated. Add a second entry with
`"generator": "topk"` or `"topq"` (and its own `tag`) to produce the pools
E4 and E5 compare.

## What each experiment reads

| | pools | selection k | model |
|---|---|---|---|
| E0 case study | rovers, q = 2.0 | 3 | rovers used (discrete) + subgoal ordering, 1/2 each |
| E1 metric vs feature | all FI pools | 5, 10, 100 | ordering + cost bin (+ agents used) |
| E2 cross-evaluation | all FI pools | 5, 10, 100 | same |
| E3 greedy vs optimal | pools with 8 to 20 behaviours | 3, 4, 5 | same |
| E4 generators | every generator's pools, cut to a shared size | 5, 10, 100 | same |
| E5 selection cost | the 20 largest pools | 5, 10, 100, 1000 | same |
| E6 sensitivity | FI pools, q = 2.0 | 10 | ordering + cost bin |

The agents-used feature is the `ru` dimension read from `data/ru-info-dir`,
which declares the agent type per domain (rovers in rovers, trucks and
airplanes in logistics, ...); domains without a declaration omit the feature,
and each result file records the objects used under `model.agents`. The
subgoal-ordering feature caps the goal atoms at 8 (`grid.max-goals`), taking
the first 8 in the canonical order; the cap and the count are in `model`.
`c*` is the cheapest plan of the pool, which is exact at q = 1.0 and the best
bound the pool gives above it; pools are deduplicated by action sequence,
plans that cannot be replayed are dropped and listed, and plans over q · c*
are filtered out and listed.

## Deviations from the brief, and why

- **The pools shipped in `data/` are forbid-iterative at q = 1.0** (k in 5,
  10, 100, 1000). The q = 2.0 pools come from the `GEN-fi-q2` sweep, which
  needs forbid-iterative installed (`setup_benchmark.sh --planners`); until it
  has run, E0 and E6 list no tasks, the cost-bin feature is constant, and E4
  reports one generator, all of which the manifests say. `topk` and `topq`
  pools are one more generation entry each.
- **Validation** is by replay against the task in unified-planning rather
  than by VAL; a plan that cannot be replayed is dropped and listed in the
  pool record.
- **E5's 10 000-plan pools** are cut to the largest pool on disk; the manifest
  records the cut.
- **Selection time limit** (10 min) is enforced by the slurm per-task limit
  rather than inside the harness; wall-clock and CPU time are recorded per
  selection.
- The metric-based selections use the same cost-sorted pool as the
  behaviour-space ones, so ties fall to the cheapest plan under every rule.

## Hardware and versions

Each `results/E?_manifest.json` records the Python, unified-planning, numpy,
scipy and matplotlib versions and the git revision of the run. Fill in the
node type and the planner versions that produced the pools when the sweep is
run; the pool files carry the planner tag and its total time.
