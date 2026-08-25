# Experiment B, checkpoint run

The run that decides whether the paper has the result its framing predicts.
Committed rather than left in a sandbox because it is what the decision rests on.

**60 tasks**, one k = 1000 pool each, across 17 domains, all of them
"discriminating": at least 20 recorded behaviours and at least 100 plans, so a
k = 20 selection has something to choose between. `pools.txt` names them.
Reproduce with

```bash
bdcevalcli run --experiment b --pool <each pool> --out ...      # or:
./setup_benchmark.sh --config discriminating --yes
```

The seed is 2026 throughout, and every derived seed comes from it, so the CSVs
are byte-reproducible.

## Why these 60 and not a random sixty

Of the 918 k = 1000 pools that hold plans, **61% hold fewer than k = 20 distinct
behaviours** and 28% hold exactly one. On such a pool every selection rule
covers every behaviour there is, so no rule can differ from any other and the
medians over all pools measure the benchmark set rather than the rules. The
`discriminating` configuration is that filter, and the fraction it excludes is
itself a result worth reporting.

## What it says

| | `bcoverage` | `bmaxsum` | `bmaxmin` | `bnovelty` | `maxsum_stability` |
|---|---|---|---|---|---|
| plans returned (median) | 20 | 20 | 2 | 2 | 20 |
| behaviours covered (median) | 20 | 20 | 2 | 2 | 20 |
| `redundancy` (median / mean) | 0 / 0.00 | 0 / 0.00 | 18 / 17.2 | 18 / 17.2 | **0 / 0.65** |
| hidden preference, exact (median) | 0.510 | **0.534** | 0.501 | 0.501 | 0.525 |
| vs baseline, paired Wilcoxon | 10-43-7, p = 0.35 | **15-44-1, p = 0.0007** | 0-0-60, p < 1e-5 | 0-0-60, p < 1e-5 | — |

1. **The plan-level baseline is not returning redundant plans.** Its redundancy
   is zero on the median task and it covers every behaviour `bcoverage` does on
   **49 of 60** tasks. The framing that the baseline hands the user twenty plans
   of which only a handful are distinct in their own terms is not what these
   pools show.

2. **It does lose behaviours on 11 of 60**, by 1 to 9, and that is exactly where
   the behaviour-space rules win: openstacks-1 (baseline 13 of 24 goal
   orderings against B-MaxSum's 20), gripper-1 (16 against 20), woodworking-13
   (7 of 77 against 16).

3. **B-MaxSum beats the baseline on the hidden preference**, significantly but
   narrowly: 15 wins, 44 ties, 1 loss, p = 0.0007 (Holm 0.0014), mean difference
   +0.014 on a measure that runs 0 to 1.

4. **B-MaxMin and B-Novelty lose on all 60**, because they return two plans
   where the others return twenty — `thm:bmaxmin-degenerate` in the data, and
   the reason B-MaxMin is read as a certificate over the other selections rather
   than as a selector of its own.

## Read `hidden_pref_exact`, not `hidden_pref_rate`

Both are in the CSV. The rate is the fifty sampled draws the design asks for;
the exact column is the same quantity in closed form. Fifty draws carry a
standard error near 0.07 and every difference between the rules is smaller than
that, so the sampled column cannot resolve them — it puts B-MaxSum level with
the baseline (p = 0.77) and B-Coverage nearly below it (p = 0.05), both of which
are noise. `coverage_by_dimension` shows the working behind the exact number.

One thing the closed form makes visible: a dimension taking a single value
across the whole pool contributes 1.0 to every selection alive. On these pools
`ru` is constant far more often than not, so half of every draw is a free hit
for all five rules alike, which is why the rates sit near 0.5. In a
one-dimensional space the measure reduces to the fraction of behaviours covered
exactly — it registers *how many* a rule covers, never *which*.
