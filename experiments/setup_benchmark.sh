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
# pools, the ru-info tree, the selection grid and the slurm settings
# (partition, account, QOS, throttle, headroom) are all read from it, so edit
# `[slurm]` in the config rather than passing flags here. Re-running is safe:
# an existing venv, checkout, pool or result is reused; --skip-existing leaves
# finished tasks out of the arrays, which is how a partial sweep is resumed.
#
# Usage:
#   experiments/setup_benchmark.sh                          # venv, install, clone, write the arrays
#   experiments/setup_benchmark.sh --submit                 # ...and submit them (slurm)
#   experiments/setup_benchmark.sh --local-jobs 4           # ...or run the sweep here, reports included
#   experiments/setup_benchmark.sh --skip-existing --submit # resume a partial sweep
#   experiments/setup_benchmark.sh --config my.toml --results-dir /scratch/bdc --submit
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # .../experiments
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                    # .../BehaviourDiversityCounter

CONFIG="default"                # a packaged config name, or a path to a TOML file
RESULTS_DIR=""                  # blank: the config's [run].results_dir
VENV_DIR="${REPO_DIR}/venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_INSTALL="no"
SKIP_FETCH="no"
SKIP_EXISTING="no"
SUBMIT="no"
LOCAL_JOBS="0"

usage() {
    cat <<'EOF'
One-shot setup for the paper's evaluation sweep: venv, install, benchmark
checkout, job arrays; then submit them or run them here.

  --config NAME|PATH      a packaged config (default, smoke) or a TOML file (default: default)
  --results-dir DIR       where the run goes (default: the config's [run].results_dir)
  --venv-dir DIR          virtualenv location (default: <repo>/venv)
  --python BIN            interpreter to build the virtualenv with (default: python3)
  --skip-install          do not touch the virtualenv
  --skip-fetch            do not clone the benchmark
  --skip-existing         leave out pools and tasks that already have a file
  --submit                submit the arrays with sbatch (the tasks, then the report)
  --local-jobs N          run the same commands here, N at a time, then the reports
  -h, --help              this text

Slurm settings live in the config's [slurm] section.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --config)        CONFIG="$2"; shift 2 ;;
        --results-dir)   RESULTS_DIR="$2"; shift 2 ;;
        --venv-dir)      VENV_DIR="$2"; shift 2 ;;
        --python)        PYTHON_BIN="$2"; shift 2 ;;
        --skip-install)  SKIP_INSTALL="yes"; shift ;;
        --skip-fetch)    SKIP_FETCH="yes"; shift ;;
        --skip-existing) SKIP_EXISTING="yes"; shift ;;
        --submit)        SUBMIT="yes"; shift ;;
        --local-jobs)    LOCAL_JOBS="$2"; shift 2 ;;
        -h|--help)       usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
die()  { printf '\033[31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- install --
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

# `bdcexp <verb> <config> [--results-dir DIR] ...` on every subcommand. The
# `${a[@]+"${a[@]}"}` form is what bash 3.2 (macOS) needs for an empty array
# under `set -u`.
RESULTS_ARGS=()
[ -n "$RESULTS_DIR" ] && RESULTS_ARGS=(--results-dir "$RESULTS_DIR")
bdcexp() { "$BDCEXP" "$1" "$CONFIG" ${RESULTS_ARGS[@]+"${RESULTS_ARGS[@]}"} "${@:2}"; }

# The run directory and whether the config has an archive to unpack, read
# through the harness so that this script never parses the TOML itself.
read -r RUN_DIR ARCHIVE < <("${VENV_DIR}/bin/python" - "$CONFIG" "$RESULTS_DIR" <<'PY'
import sys
from bdc_experiments.config import load, results_root
cfg = load(sys.argv[1], results_dir=sys.argv[2] or None)
print(results_root(cfg).resolve(), 'yes' if cfg['generation']['archive'] else 'no')
PY
)
say "config ${CONFIG}, run directory ${RUN_DIR}"

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
                        (or: $0 --submit${RESULTS_DIR:+ --results-dir $RESULTS_DIR})
  watch it              squeue -u \$USER
  resume a partial run  $0 --skip-existing --submit
  run it here instead   $0 --local-jobs 4
  the reports           ${BDCEXP} report ${CONFIG}${RESULTS_DIR:+ --results-dir $RESULTS_DIR} all
                        (submit_all.sh queues this as the last job)
EOF
fi
