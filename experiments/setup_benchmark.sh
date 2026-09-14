#!/usr/bin/env bash
#
# One-shot setup for the paper's evaluation sweep:
#
#   1. create a virtualenv and install the library and the harness (bdcexp),
#   2. fetch classical-domains at the commit the config pins and unpack the
#      pools of the archive against it (phase one),
#   3. write the job arrays of phase two (`bdcexp jobs`) and submit them, or
#      run the same commands here.
#
# The config file is the single source of truth: the domains, the archive of
# pools, the ru-info tree, the selection grid and the slurm defaults are all
# read from it. Every slurm setting is prompted with the config's value as the
# default, and every prompt has a matching flag, so the same script drives an
# interactive setup and a scripted one (--yes). An answer that differs from
# the config becomes a `bdcexp --set` override, which travels with every
# command of the sweep and is recorded in every manifest. Re-running is safe:
# an existing venv, checkout, pool or result is reused; --skip-existing leaves
# finished tasks out of the arrays, which is how a partial sweep is resumed.
#
# Usage:
#   experiments/setup_benchmark.sh                          # interactive
#   experiments/setup_benchmark.sh --yes                    # every default, no questions
#   experiments/setup_benchmark.sh --submit --yes           # build and submit the sweep
#   experiments/setup_benchmark.sh --partition long --max-parallel 100 --submit
#   experiments/setup_benchmark.sh --skip-existing --submit --yes   # resume a partial sweep
#   experiments/setup_benchmark.sh --local-jobs 4 --yes     # run the sweep here, reports included
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # .../experiments
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../BehaviourDiversityCounter

CONFIG="default"                # a packaged config name, or a path to a TOML file
RESULTS_DIR=""                  # blank: the config's [run].results_dir
VENV_DIR=""                     # blank: prompted, default <repo>/venv
PYTHON_BIN="${PYTHON_BIN:-python3}"
TIME_LIMIT=""                   # blank: prompted, default the config's value
MEMORY_MB=""
CPUS=""
PARTITION=""
ACCOUNT=""
QOS=""
MAX_PARALLEL=""
MAX_ARRAY=""
SKIP_INSTALL="no"
SKIP_FETCH="no"
SKIP_EXISTING="no"
SUBMIT="no"
LOCAL_JOBS="0"
ASSUME_YES="no"

usage() {
    cat <<'EOF'
One-shot setup for the paper's evaluation sweep: venv, install, benchmark
checkout, pools, job arrays; then submit them or run them here. Every slurm
setting is prompted with the config's value as the default; a flag answers
the prompt, and --yes takes every default.

  --config NAME|PATH      a packaged config (default, smoke) or a TOML file (default: default)
  --results-dir DIR       where the run goes (default: the config's [run].results_dir)
  --venv-dir DIR          virtualenv location (default: <repo>/venv)
  --python BIN            interpreter to build the virtualenv with (default: python3)

  --time-limit S|H:M:S    per-task wall clock, seconds or HH:MM:SS   ([run].time_limit_selection_s)
  --memory-mb N           per-task memory in MB                       ([slurm].memory_mb)
  --cpus N                cpus per task                               ([slurm].cpus_per_task)
  --partition NAME        slurm partition, blank = site default       ([slurm].partition)
  --account NAME          slurm account,   blank = site default       ([slurm].account)
  --qos NAME              slurm QOS,       blank = site default       ([slurm].qos)
  --max-parallel N        array elements running at once              ([slurm].max_parallel_jobs)
  --max-array-size N      elements per array; the site's MaxArraySize - 1 where scontrol answers
                                                                      ([slurm].max_array_size)

  --skip-install          do not touch the virtualenv
  --skip-fetch            do not clone the benchmark
  --skip-existing         leave out the tasks that already have a result
  --submit                submit the arrays with sbatch (the tasks, then the report)
  --local-jobs N          run the same commands here, N at a time, then the reports
  -y, --yes               take every default, ask nothing
  -h, --help              this text
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config)         CONFIG="$2"; shift 2 ;;
        --results-dir)    RESULTS_DIR="$2"; shift 2 ;;
        --venv-dir)       VENV_DIR="$2"; shift 2 ;;
        --python)         PYTHON_BIN="$2"; shift 2 ;;
        --time-limit)     TIME_LIMIT="$2"; shift 2 ;;
        --memory-mb)      MEMORY_MB="$2"; shift 2 ;;
        --cpus)           CPUS="$2"; shift 2 ;;
        --partition)      PARTITION="$2"; shift 2 ;;
        --account)        ACCOUNT="$2"; shift 2 ;;
        --qos)            QOS="$2"; shift 2 ;;
        --max-parallel)   MAX_PARALLEL="$2"; shift 2 ;;
        --max-array-size) MAX_ARRAY="$2"; shift 2 ;;
        --skip-install)   SKIP_INSTALL="yes"; shift ;;
        --skip-fetch)     SKIP_FETCH="yes"; shift ;;
        --skip-existing)  SKIP_EXISTING="yes"; shift ;;
        --submit)         SUBMIT="yes"; shift ;;
        --local-jobs)     LOCAL_JOBS="$2"; shift 2 ;;
        -y|--yes)         ASSUME_YES="yes"; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# ask <prompt> <default> <variable-name>: a flag's value stands and is not
# asked about; the prompt is skipped under --yes or without a terminal.
ask() {
    local prompt="$1" default="$2" varname="$3" answer=""
    [ -z "${!varname}" ] || return 0
    if [ "$ASSUME_YES" = "yes" ] || [ ! -t 0 ]; then
        answer="$default"
    else
        read -r -p "$(printf '%s [%s]: ' "$prompt" "$default")" answer || answer=""
        answer="${answer:-$default}"
    fi
    printf -v "$varname" '%s' "$answer"
}

# seconds <S or MM:SS or HH:MM:SS>
seconds() {
    local h=0 m=0 s
    case "$1" in
        *:*:*) IFS=: read -r h m s <<< "$1" ;;
        *:*)   IFS=: read -r m s <<< "$1" ;;
        *)     s="$1" ;;
    esac
    echo $((10#$h * 3600 + 10#$m * 60 + 10#$s))
}

# ---------------------------------------------------------------- install --
ask "Virtualenv directory" "${REPO_DIR}/venv" VENV_DIR
if [ "$SKIP_INSTALL" = "yes" ]; then
    say "skipping installation (--skip-install)"
else
    if [ ! -d "$VENV_DIR" ]; then
        say "creating virtualenv at ${VENV_DIR}"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        say "reusing virtualenv at ${VENV_DIR}"
    fi
    say "installing the library and the harness, with the report extras"
    "${VENV_DIR}/bin/python" -m pip install --quiet --upgrade pip
    "${VENV_DIR}/bin/python" -m pip install --quiet -e "${REPO_DIR}[analysis]"
fi
BDCEXP="${VENV_DIR}/bin/bdcexp"
[ -x "$BDCEXP" ] || die "no bdcexp at ${BDCEXP}: run without --skip-install, or point --venv-dir at a venv that has it"

# The run directory, whether there is an archive to unpack, and the slurm
# defaults, read through the harness so that this script never parses the
# TOML itself. '|'-separated: an empty field survives it, a blank would not.
IFS='|' read -r RUN_DIR ARCHIVE D_TIME D_MEMORY D_CPUS D_PARTITION D_ACCOUNT D_QOS D_PARALLEL D_ARRAY \
    < <("${VENV_DIR}/bin/python" - "$CONFIG" "$RESULTS_DIR" <<'PY'
import sys
from bdc_experiments.config import load, results_root
cfg = load(sys.argv[1], results_dir=sys.argv[2] or None)
s = cfg['slurm']
print('|'.join(map(str, [results_root(cfg).resolve(), 'yes' if cfg['generation']['archive'] else 'no',
                        cfg['run']['time_limit_selection_s'], s['memory_mb'], s['cpus_per_task'],
                        s['partition'], s['account'], s['qos'], s['max_parallel_jobs'], s['max_array_size']])))
PY
)
say "config ${CONFIG}, run directory ${RUN_DIR}"

# ------------------------------------------------------------ the prompts --
# The site's MaxArraySize caps the highest array index, so the default chunk
# is one below it where scontrol can say; the config's value otherwise.
if command -v scontrol >/dev/null 2>&1; then
    site_max="$(scontrol show config 2>/dev/null | awk -F'= *' '/^MaxArraySize/ {print $2}' | tr -d ' ')"
    if [ -n "${site_max:-}" ] && [ "$site_max" -gt 1 ] 2>/dev/null && [ $((site_max - 1)) -lt "$D_ARRAY" ]; then
        D_ARRAY=$((site_max - 1))
    fi
fi
ask "Per-task time limit (seconds, or HH:MM:SS)"  "$D_TIME"      TIME_LIMIT
ask "Per-task memory (MB)"                         "$D_MEMORY"    MEMORY_MB
ask "CPUs per task"                                "$D_CPUS"      CPUS
ask "Slurm partition (blank = site default)"       "$D_PARTITION" PARTITION
ask "Slurm account (blank = site default)"         "$D_ACCOUNT"   ACCOUNT
ask "Slurm QOS (blank = site default)"             "$D_QOS"       QOS
ask "Max array elements running at once"           "$D_PARALLEL"  MAX_PARALLEL
ask "Max elements per array"                       "$D_ARRAY"     MAX_ARRAY
TIME_LIMIT="$(seconds "$TIME_LIMIT")"

# Only what differs from the config becomes an override, so a run with every
# default keeps the config's own hash.
SET_ARGS=()
override() { [ "$2" = "$3" ] || SET_ARGS+=(--set "$1=$2"); }
override run.time_limit_selection_s "$TIME_LIMIT"   "$D_TIME"
override slurm.memory_mb            "$MEMORY_MB"    "$D_MEMORY"
override slurm.cpus_per_task        "$CPUS"         "$D_CPUS"
override slurm.partition            "$PARTITION"    "$D_PARTITION"
override slurm.account              "$ACCOUNT"      "$D_ACCOUNT"
override slurm.qos                  "$QOS"          "$D_QOS"
override slurm.max_parallel_jobs    "$MAX_PARALLEL" "$D_PARALLEL"
override slurm.max_array_size       "$MAX_ARRAY"    "$D_ARRAY"
[ ${#SET_ARGS[@]} -eq 0 ] || say "overriding the config with:${SET_ARGS[*]/--set/}"

# `bdcexp <verb> <config> [--results-dir DIR] [--set ...] ...` on every
# subcommand. The `${a[@]+"${a[@]}"}` form is what bash 3.2 (macOS) needs for
# an empty array under `set -u`.
RESULTS_ARGS=()
[ -n "$RESULTS_DIR" ] && RESULTS_ARGS=(--results-dir "$RESULTS_DIR")
bdcexp() {
    "$BDCEXP" "$1" "$CONFIG" ${RESULTS_ARGS[@]+"${RESULTS_ARGS[@]}"} ${SET_ARGS[@]+"${SET_ARGS[@]}"} "${@:2}"
}

# ------------------------------------------------------------------ fetch --
if [ "$ARCHIVE" = "no" ]; then
    say "the config names no archive: the pools are committed, no benchmark checkout is needed"
elif [ "$SKIP_FETCH" = "yes" ]; then
    say "skipping the benchmark checkout (--skip-fetch)"
elif [ -d "${RUN_DIR}/benchmark/.git" ]; then
    say "reusing the benchmark checkout at ${RUN_DIR}/benchmark"
else
    say "cloning classical-domains into ${RUN_DIR}/benchmark"
    bdcexp generate --list > /dev/null
fi

# ------------------------------------------------------------ the pools --
# Phase one is unpacking the archive: seconds, so it runs here and not as a
# job, and the arrays below can list one task per pool and model.
if [ "$ARCHIVE" = "yes" ]; then
    say "unpacking the pools of the configured domains (existing ones are kept)"
    bdcexp generate | tail -n 3 | sed 's/^/      /'
fi

# ------------------------------------------------------------- the arrays --
say "writing the job arrays"
JOBS_FLAG=""
[ "$SKIP_EXISTING" = "yes" ] && JOBS_FLAG="--skip-existing"
bdcexp jobs $JOBS_FLAG | sed 's/^/      /'
SLURM_DIR="${RUN_DIR}/slurm"

# --------------------------------------------------------- submit or run --
if [ "$LOCAL_JOBS" -gt 0 ] 2>/dev/null; then
    say "running the sweep here, ${LOCAL_JOBS} at a time"
    bash "${SLURM_DIR}/run_local.sh" "$LOCAL_JOBS"
    say "building the reports"
    bdcexp report all | sed 's/^/      /'
    say "done: ${RUN_DIR}/reports/"
    echo "E3 measures time: with several workers, or on a shared machine, its numbers are of the load." >&2
elif [ "$SUBMIT" = "yes" ]; then
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found on this machine"
    say "submitting: the tasks, then the report"
    bash "${SLURM_DIR}/submit_all.sh"
    say "submitted; watch with: squeue -u \$USER; logs under ${SLURM_DIR}/logs/"
else
    cat <<EOF

Next steps
  submit the sweep      bash ${SLURM_DIR}/submit_all.sh
                        (or re-run this script with --submit and the same answers)
  watch it              squeue -u \$USER
  resume a partial run  $0 --skip-existing --submit --yes
  run it here instead   $0 --local-jobs 4 --yes
  the reports           ${BDCEXP} report ${CONFIG}${RESULTS_DIR:+ --results-dir $RESULTS_DIR}${SET_ARGS[@]+ ${SET_ARGS[*]}} all
                        (submit_all.sh queues this as the last job)
EOF
fi
