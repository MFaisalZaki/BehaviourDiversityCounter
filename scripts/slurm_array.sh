#!/usr/bin/env bash
# One sbatch array per task kind, one element per task, and one report job
# that waits for both. No prompts and no chunking: if a site's array limit
# bites, split the task list with `split -n l/<parts>` and submit the parts.
#
#   scripts/slurm_array.sh <config> [kind ...]        kinds: select time
set -euo pipefail

CONFIG="${1:?usage: slurm_array.sh <config> [kind ...]}"; shift
KINDS=("${@:-select time}")
LOGS="${SLURM_LOGS:-slurm-logs}"; mkdir -p "$LOGS"
DEPENDS=""

for kind in ${KINDS[@]}; do
    list="$LOGS/$kind.tasks"
    bdcexp run "$CONFIG" "$kind" --list > "$list"
    count=$(wc -l < "$list")
    [ "$count" -gt 0 ] || { echo "$kind: no tasks"; continue; }
    run=$(sbatch --parsable --job-name="bdc-$kind" --array="1-$count" \
                 --output="$LOGS/$kind-%a.out" --time="${SLURM_TIME:-04:00:00}" \
                 --mem="${SLURM_MEM:-8G}" --cpus-per-task=1 \
                 --wrap "bdcexp run $CONFIG $kind --task \$(sed -n \"\${SLURM_ARRAY_TASK_ID}p\" $list)")
    DEPENDS="$DEPENDS:$run"
    echo "$kind: $count tasks submitted as job $run"
done

[ -n "$DEPENDS" ] && sbatch --parsable --job-name="bdc-report" --dependency="afterany$DEPENDS" \
       --output="$LOGS/report.out" --time="${SLURM_TIME:-04:00:00}" \
       --mem="${SLURM_MEM:-8G}" --cpus-per-task=1 --wrap "bdcexp report $CONFIG all"
