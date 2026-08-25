# Experiment configurations

Each directory here is a *template* for an experiment: one `exp-details.json`
holding the resource limits, the pool selection and the knobs of the three
experiments. `setup_benchmark.sh` copies the one you pick into a working
experiment (`<work-dir>/experiments/<config>`) and rewrites its limits there, so
these stay as they were committed.

| configuration | pools | what it is for |
|---|---|---|
| `default` | every k = 1000 pool that holds plans (918) | the full sweep |
| `discriminating` | k = 1000 pools with at least 20 recorded behaviours and 100 plans (337) | the stratum where a selection rule can differ from another one at all — 61% of pools hold fewer than k = 20 distinct behaviours, and there no rule can differ from any other |
| `smoke` | three domains, one pool each | an end-to-end check in a couple of minutes |

Copy any of them, edit, and pass the directory to `--config`.

## The fields

`cfgs` — per-pool time and memory limits, the head-room slurm gets on top of
them, and the slurm knobs (partition, account, QOS, throttle, array size).

`pools` — which FI pools to sweep. `requested-k` picks the pool family by the
`k` in its filename; `min-behaviour-count` and `min-plans` filter on what the
pool file already records, so selection needs no simulation;
`max-pools-per-domain` and `max-pools` cap a sweep, and `selection: even`
spreads that cap across domains rather than taking the alphabetically earliest.

`experiments` — `seed` (every other seed is derived from it, so one number
reproduces a whole sweep), `k-nn` for B-Novelty, `multiset-stability` for the
plan-level baseline's reading of `A(p)`, and one block per experiment. Each can
be switched off; none can be added, because there are exactly three.
