"""Experiment C: the effect of weights.

How sensitive is the selection to the user's weights, and does raising a weight
move the selection the way the paper claims? B-MaxSum only, over the tasks whose
behaviour space carries both dimensions, sweeping w_go from 0 to 1 in steps of
0.05 with w_ru = 1 - w_go and re-selecting k = 20 at every step.

Two things are recorded.

**Sensitivity.** The smallest w_go at which the returned set differs from the
one returned at w_go = 0, and how many distinct sets appear over the sweep.
"Differs" means the set of *distinct behaviours* differs: a weight change that
swaps one plan for another exhibiting the same behaviour has changed nothing
the user can see, and counting it would overstate the sensitivity. The
plan-level version is recorded beside it. Both tails are informative and
neither is a bug -- a set that changes at every step means the model is twitchy
and the user's weights must be exact, and a set that never changes means the
pool cannot act on the preference at all, which is the two-phase ceiling the
paper already concedes.

**Responsiveness.** spread_go and spread_ru at each step. The paper predicts the
first rises and the second falls as w_go rises. Nothing here smooths, filters or
drops a task to make the curves cross; if they do not, the claim in section 5.1
is falsified and that needs to be known before a reviewer finds it.

``ru_values_in_pool`` is in every row because it decides whether the question
can be asked at all: where every plan in a pool uses the same resources, the ru
distance is identically zero and no weighting of it can move anything.
"""

import hashlib
import os
import sys
import tempfile

if __package__ in (None, ''):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paperexps import harness

FIELDS = [
    'task_id', 'domain', 'w_go', 'w_ru', 'k', 'seed',
    'behaviour_set_hash', 'changed_from_baseline', 'n_distinct_behaviour_sets_so_far',
    'plan_set_hash', 'changed_from_baseline_plans',
    'spread_go', 'spread_ru', 'b_coverage', 'b_maxsum',
    # Beyond the specified shape:
    'step', 'n_distinct_plan_sets_so_far', 'n_selected',
    'n_pool', 'b_pool', 'ru_values_in_pool', 'go_values_in_pool', 'year',
]


def stable_hash(items):
    """A hash that is the same in every process.

    Python's own hash() is randomised per process, so a sweep run across a
    SLURM array would compare hashes that were never comparable.
    """
    joined = '\x00'.join(sorted(str(item) for item in items))
    return hashlib.sha1(joined.encode()).hexdigest()[:16]


def weight_steps(step_size):
    """w_go from 0 to 1 inclusive, exactly, without float drift."""
    count = round(1.0 / step_size)
    return [index / count for index in range(count + 1)]


def run_pool(pool, args):
    counter = pool.counter
    seed = harness.derive_seed(args.seed, pool.task_id, 'weights')
    position = pool.position()
    values_in_pool = {
        name: len({harness.dimension_value(behaviour, name) for behaviour in pool.distinct})
        for name in pool.dimensions
    }

    rows = []
    baseline_behaviours = baseline_plans = None
    behaviour_hashes, plan_hashes = [], []
    for step, w_go in enumerate(weight_steps(args.step)):
        counter.set_weights({'go': w_go, 'ru': 1.0 - w_go})
        selected = counter.extract(pool.plans, k=args.k, indicator='bmaxsum')
        behaviours = [plan.behaviour for plan in selected]

        behaviour_hash = stable_hash(set(behaviours))
        plan_hash = stable_hash(position[id(plan)] for plan in selected)
        if baseline_behaviours is None:
            baseline_behaviours, baseline_plans = behaviour_hash, plan_hash
        behaviour_hashes.append(behaviour_hash)
        plan_hashes.append(plan_hash)

        rows.append({
            'task_id': pool.task_id, 'domain': pool.domain,
            'w_go': w_go, 'w_ru': 1.0 - w_go, 'k': args.k, 'seed': seed,
            'behaviour_set_hash': behaviour_hash,
            'changed_from_baseline': behaviour_hash != baseline_behaviours,
            'n_distinct_behaviour_sets_so_far': len(set(behaviour_hashes)),
            'plan_set_hash': plan_hash,
            'changed_from_baseline_plans': plan_hash != baseline_plans,
            'spread_go': harness.dimension_spread(counter, behaviours, 'go'),
            'spread_ru': harness.dimension_spread(counter, behaviours, 'ru'),
            'b_coverage': counter.b_coverage(selected),
            'b_maxsum': counter.b_maxsum(selected),
            'step': step,
            'n_distinct_plan_sets_so_far': len(set(plan_hashes)),
            'n_selected': len(selected),
            'n_pool': pool.n, 'b_pool': pool.b,
            'ru_values_in_pool': values_in_pool.get('ru', ''),
            'go_values_in_pool': values_in_pool.get('go', ''),
            'year': pool.year,
        })
    counter.set_weights(None)  # leave the counter as it was found
    return rows


def main(argv=None):
    parser = harness.base_parser(__doc__.splitlines()[0])
    parser.add_argument('--k', type=int, default=20, help='Selection size.')
    parser.add_argument('--step', type=float, default=0.05,
                        help='Weight step; 0.05 gives the 21 points of the paper.')
    parser.add_argument('--require-varying-ru', action='store_true',
                        help='Skip tasks whose ru dimension takes one value across the '
                             'whole pool, where no weighting of it can move anything. '
                             'Off by default: those tasks are a finding, not noise.')
    args = parser.parse_args(argv)
    if args.dry_run:
        args.step = 0.25

    rows, skipped = [], []
    for path in harness.pools_from(args):
        task_id = harness.task_id_of(path)
        with tempfile.TemporaryDirectory(prefix='paperexps-') as workdir:
            try:
                pool = harness.load_pool(path, args, workdir)
            except harness.PoolSkipped as reason:
                skipped.append((task_id, str(reason)))
                print(f'{task_id}: skipped -- {reason}')
                continue
            if len(pool.dimensions) < 2:
                skipped.append((task_id, 'behaviour space has fewer than two dimensions'))
                print(f'{task_id}: skipped -- only {pool.dimensions}')
                continue
            pool_rows = run_pool(pool, args)
            if args.require_varying_ru and pool_rows[0]['ru_values_in_pool'] <= 1:
                skipped.append((task_id, 'ru takes one value across the pool'))
                print(f'{task_id}: skipped -- ru is constant over the pool')
                continue
            rows += pool_rows
            last = pool_rows[-1]
            changed = [r for r in pool_rows if r['changed_from_baseline']]
            first = f"{changed[0]['w_go']:.2f}" if changed else 'never'
            print(f"{task_id}: ru_values={last['ru_values_in_pool']} "
                  f"distinct behaviour sets={last['n_distinct_behaviour_sets_so_far']} "
                  f"first change at w_go={first}")

    harness.write_csv(args.out, FIELDS, rows)
    print(f'wrote {len(rows)} rows to {args.out}')
    if skipped:
        print(f'skipped {len(skipped)} pool(s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
