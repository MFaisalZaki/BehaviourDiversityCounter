# Paper experiments — Reshaping Diversity Planning

Empirical evaluation for the paper, over the ForbidIterative plan pools in
`data/fi-generated-plans-dir.zip`. Five experiments, all computed in a single
pass per pool:

| # | Question | Output |
|---|----------|--------|
| 1 | **Behaviour collapse**: how many of a pool's n plans are behaviourally distinct (b/n)? | `exp1_collapse.csv` |
| 2 | **Discrimination** (paper Q1): within groups of equal BDC, how far does B-MaxSum spread? | `exp2_discrimination.csv` |
| 3 | **Cross-evaluation**: do metric-based selections (greedy MaxSum-Stability) deliver behaviour diversity, and vice versa? | `exp3_cross.csv`, `exp3_summary.csv` |
| 4 | **Greedy gap**: how close is greedy Extract_BMaxSum to the exact optimum (B-MaxSum is not submodular), plus the Theorem `bdc-greedy` check | `exp4_greedy_gap.csv` |
| 5 | **Runtime** (paper Q2): BDC vs B-MaxSum vs plan-level MaxSum wall-clock | `exp5_runtime.csv` |

## Layout

```
paper-experiments/
  data/
    fi-generated-plans-dir.zip   the FI pools (one JSON per task x requested-k)
    ru-info-dir/                 per-domain (:resource ...) declarations
    classical-domains/           cloned by setup.sh (AI-Planning/classical-domains)
  paperexps/
    common.py          pool-file -> PDDL task + resources matching
    run_pool.py        one pool -> one result JSON (all five experiments)
    generate_slurm.py  aspbench-style job arrays (one array per requested-k)
    aggregate.py       result JSONs -> the CSVs above (+ headline stats)
    plots.py           paper figures from the CSVs
  setup_benchmark.sh   one-shot setup: prompts + flags, venv, fetch, generate
  sandbox/             generated: cmds/, slurm/, results/, analysis/
```

## Matching pools to tasks

A pool file `1.0-100-classical-2006-rovers-15-fi-bc-results.json` is
`{q}-{k}-{track}-{year}-{domain}-{inst}`. Its `info` block names the exact
files inside the classical-domains checkout (`info.domain` =
`rovers/domain.pddl` relative to `classical/`, `info.problem` = the problem
basename), so task matching never guesses. The filename's `{domain}-{year}`
and `{inst}` key the resource declarations:
`data/ru-info-dir/{domain}/{domain}-{year}.json` holds `instances[inst]`,
where `inst` is the 1-based position of the problem in the domain's problem
list sorted by file name (catalog-run-2's convention). Pools of 48 bytes are
planner timeouts and are skipped at generation time.

## Behaviour space

The paper's Section 7 space: goal-predicate ordering (`go`) plus, where the
instance declares resources, the set of resources used (`ru`). `ru` rather
than `rc` because B-MaxSum needs per-dimension distances, which only `go`,
`cb`, `ru` implement. Domains without resources measure diversity along `go`
alone.

## Running

```bash
./setup_benchmark.sh                # interactive: prompts with defaults
./setup_benchmark.sh --yes          # all defaults, no prompts
./setup_benchmark.sh --partition compute --k-filter 100 --yes

bash sandbox/slurm/submit_all.sh    # cluster
bash sandbox/run_local.sh 8         # or locally, 8 pools at a time

venv/bin/python paperexps/aggregate.py   # CSVs + headline numbers
venv/bin/python paperexps/plots.py       # figures (PDF + PNG)
```

`setup_benchmark.sh` mirrors ASPPlanners' `benchmarks/setup_benchmark.sh`:
every prompt has a matching flag, `--yes` accepts all defaults, and re-running
is safe (venv, clone and unpacked pools are reused; `--skip-existing` skips
pools that already have results). It creates the venv (needs Python 3.10-3.13;
pass `PYTHON_BIN=python3.12` if `python3` is newer), installs
`behaviour_diversity_counter` and `plandiversity` (from the sibling
DiverseScore checkout, or GitHub), fetches classical-domains, unpacks the
pools, and calls `generate_slurm.py` -- which emits one job array per
requested-k group, each array index pulling its line from a command file, a
`submit_all.sh`, and a no-slurm `run_local.sh` fallback. Defaults: 90 min /
8 GB per pool (the paper's stated limits), arrays throttled to 50 concurrent
jobs.

## Method notes

- **Exp 2 sampling.** B-MaxSum depends only on the *distinct* behaviours a
  subset exhibits, so a size-m plan subset with BDC = v is sampled as a
  v-subset of the pool's b behaviours that some m plans can realise (total
  multiplicity ≥ m). Min/max/spread per group are identical to plan-subset
  sampling (same support); means/stds are over behaviour combinations.
  Groups with C(b, v) ≤ `--samples` are enumerated exhaustively (flagged
  `exhaustive` in the CSV).
- **Exp 3 selectors**: greedy plan-level MaxSum over Stability (the
  ForbidIterative baseline) and Uniqueness (`--with-states` adds States at
  the cost of a second simulation pass), `extract` under both behaviour
  indicators, `random`, and `first-k` (the pool in FI generation order).
  Every selection is scored under every family.
- **Exp 4 exact optimum.** A size-m set maximising B-MaxSum exhibits
  min(m, b) distinct behaviours, so the optimum is the best pair-sum over
  C(b, min(m, b)) behaviour combinations — enumerated when that count is
  within `--exact-cap` (default 200k), recorded as skipped otherwise.
- **Exp 5 timings** separate behaviour extraction (simulation, shared by both
  indicators) from indicator evaluation: `t_bdc_warm` (min of 3, warm
  cache), `t_bmaxsum_cold` (first call, pays all pairwise distances) and
  `t_bmaxsum_warm`, and `t_maxsum_stability_cold` (plan-level baseline,
  features + O(n²) pairs).
- Every per-pool result records parse failures and inapplicable plans rather
  than dying on them; pools that lose every plan produce a `skipped` record,
  tallied in `skipped.csv`.
