#!/usr/bin/env bash
# One sbatch array per experiment, one element per task, plus a dependent
# report job. No prompts and no chunking: if a site's array limit bites, split
# the task list with `split -n l/<parts>` and submit the parts.
#
#   scripts/slurm_array.sh <config> [experiment ...]
set -euo pipefail

CONFIG="${1:?usage: slurm_array.sh <config> [experiment ...]}"; shift
EXPERIMENTS=("${@:-e3 e2 e4 e5 e6 e1}")
LOGS="${SLURM_LOGS:-slurm-logs}"; mkdir -p "$LOGS"

for experiment in ${EXPERIMENTS[@]}; do
    list="$LOGS/$experiment.tasks"
    bdcexp run "$CONFIG" "$experiment" --list > "$list"
    count=$(wc -l < "$list")
    [ "$count" -gt 0 ] || { echo "$experiment: no tasks"; continue; }
    run=$(sbatch --parsable --job-name="bdc-$experiment" --array="1-$count" \
                 --output="$LOGS/$experiment-%a.out" --time="${SLURM_TIME:-04:00:00}" \
                 --mem="${SLURM_MEM:-8G}" --cpus-per-task=1 \
                 --wrap "bdcexp run $CONFIG $experiment --task \$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $list)")
    sbatch --parsable --job-name="bdc-$experiment-report" --dependency="afterany:$run" \
           --output="$LOGS/$experiment-report.out" --time="${SLURM_TIME:-04:00:00}" \
           --mem="${SLURM_MEM:-8G}" --cpus-per-task=1 \
           --wrap "bdcexp report $CONFIG $experiment"
    echo "$experiment: $count tasks submitted as job $run"
done
