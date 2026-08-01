#!/usr/bin/env bash
#
# One-shot setup for the paper-experiments sweep:
#
#   1. ask for the per-pool time and memory limits (and a few other knobs),
#   2. create a virtualenv and install behaviour_diversity_counter,
#      plandiversity and matplotlib into it,
#   3. fetch the classical-domains benchmark repository and unpack the FI
#      plan pools,
#   4. generate one run_pool command per pool and the slurm job arrays.
#
# Everything is prompted with a default, and every prompt has a matching flag,
# so the same script drives an interactive setup and a scripted one (--yes).
# Re-running it is safe: an existing venv, clone or unpacked pool dir is
# reused, and --skip-existing skips pools that already have results.
#
# Usage:
#   ./setup_benchmark.sh                      # interactive
#   ./setup_benchmark.sh --yes                # all defaults, no prompts
#   ./setup_benchmark.sh --partition compute --k-filter 100 --yes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../paper-experiments
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # .../BehaviourDiversityCounter

# ---------------------------------------------------------------- defaults --
VENV_DIR="${SCRIPT_DIR}/venv"
SANDBOX_DIR="${SCRIPT_DIR}/sandbox"
TASKS_DIR="${SCRIPT_DIR}/data/classical-domains"
PLANS_DIR="${SCRIPT_DIR}/data/fi-generated-plans-dir"
DIVERSESCORE_DIR="${DIVERSESCORE_DIR:-$(dirname "${REPO_DIR}")/DiverseScore}"
TIME_LIMIT="01:30:00"
MEMORY_LIMIT="8G"
K_FILTER=""
PARTITION=""
ACCOUNT=""
QOS=""
MAX_PARALLEL="50"
SAMPLES="1000"
SUBSET_SIZES="5,10,20"
SELECT_K="5,10"
EXACT_SIZES="5,10"
EXACT_CAP="200000"
SEED="2026"
WITH_STATES="no"
SKIP_FETCH="no"
SKIP_INSTALL="no"
SKIP_EXISTING="no"
ASSUME_YES="no"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CLASSICAL_REPO="https://github.com/AI-Planning/classical-domains.git"
PLANDIVERSITY_REPO="https://github.com/MFaisalZaki/plandiversity.git"

usage() {
    cat <<'EOF'
One-shot setup for the paper-experiments sweep:

  1. ask for the per-pool time and memory limits (and a few other knobs),
  2. create a virtualenv and install the two libraries into it,
  3. fetch classical-domains and unpack the FI plan pools,
  4. generate one run_pool command per pool and the slurm job arrays.

Re-running it is safe: an existing venv, clone or unpacked pool dir is reused.

Usage:
  ./setup_benchmark.sh                      # interactive
  ./setup_benchmark.sh --yes                # all defaults, no prompts
  ./setup_benchmark.sh --partition compute --k-filter 100 --yes

Options:
  --venv-dir DIR          virtualenv location            (default: <here>/venv)
  --sandbox-dir DIR       commands, results, logs        (default: <here>/sandbox)
  --tasks-dir DIR         classical-domains checkout     (default: <here>/data/classical-domains)
  --plans-dir DIR         unpacked FI pools              (default: <here>/data/fi-generated-plans-dir)
  --diversescore-dir DIR  DiverseScore checkout for plandiversity
                          (default: sibling of this repo; GitHub as fallback)
  --time-limit VALUE      per pool, e.g. 01:30:00
  --memory-limit VALUE    per pool, e.g. 8G
  --k-filter LIST         only pools with these requested k, e.g. "100" or "5,100"
                          (blank = every pool)
  --partition NAME        slurm partition
  --account NAME          slurm account
  --qos NAME              slurm QOS
  --max-parallel N        cap on concurrently running array jobs
  --samples N             exp 2: samples per (m, BDC value) group
  --subset-sizes LIST     exp 2: subset sizes m               (default: 5,10,20)
  --select-k LIST         exp 3: selection sizes k            (default: 5,10)
  --exact-sizes LIST      exp 4: exact/greedy subset sizes    (default: 5,10)
  --exact-cap N           exp 4: enumeration budget on C(b,m) (default: 200000)
  --seed N                sampling seed                       (default: 2026)
  --with-states           exp 3: add the States metric (a second simulation pass)
  --skip-fetch            do not clone/update classical-domains or unzip the pools
  --skip-install          do not create the venv or install anything
  --skip-existing         do not regenerate commands for pools that have results
  -y, --yes               accept every default, never prompt
  -h, --help              this message
EOF
}

# ------------------------------------------------------------------- args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --venv-dir)          VENV_DIR="$2"; shift 2 ;;
        --sandbox-dir)       SANDBOX_DIR="$2"; shift 2 ;;
        --tasks-dir)         TASKS_DIR="$2"; shift 2 ;;
        --plans-dir)         PLANS_DIR="$2"; shift 2 ;;
        --diversescore-dir)  DIVERSESCORE_DIR="$2"; shift 2 ;;
        --time-limit)        TIME_LIMIT="$2"; shift 2 ;;
        --memory-limit)      MEMORY_LIMIT="$2"; shift 2 ;;
        --k-filter)          K_FILTER="$2"; shift 2 ;;
        --partition)         PARTITION="$2"; shift 2 ;;
        --account)           ACCOUNT="$2"; shift 2 ;;
        --qos)               QOS="$2"; shift 2 ;;
        --max-parallel)      MAX_PARALLEL="$2"; shift 2 ;;
        --samples)           SAMPLES="$2"; shift 2 ;;
        --subset-sizes)      SUBSET_SIZES="$2"; shift 2 ;;
        --select-k)          SELECT_K="$2"; shift 2 ;;
        --exact-sizes)       EXACT_SIZES="$2"; shift 2 ;;
        --exact-cap)         EXACT_CAP="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --with-states)       WITH_STATES="yes"; shift ;;
        --skip-fetch)        SKIP_FETCH="yes"; shift ;;
        --skip-install)      SKIP_INSTALL="yes"; shift ;;
        --skip-existing)     SKIP_EXISTING="yes"; shift ;;
        -y|--yes)            ASSUME_YES="yes"; shift ;;
        -h|--help)           usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- helpers ---
say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m warning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m error:\033[0m %s\n' "$*" >&2; exit 1; }

# ask <prompt> <default> <variable-name>
ask() {
    local prompt="$1" default="$2" varname="$3" answer=""
    if [ "$ASSUME_YES" = "yes" ] || [ ! -t 0 ]; then
        answer="$default"
    else
        read -r -p "$(printf '%s [%s]: ' "$prompt" "$default")" answer || answer=""
        answer="${answer:-$default}"
    fi
    printf -v "$varname" '%s' "$answer"
}

ask_yes_no() {
    local prompt="$1" default="$2" varname="$3" answer=""
    ask "$prompt (yes/no)" "$default" answer
    # tr rather than ${answer,,}: macOS still ships bash 3.2.
    case "$(printf '%s' "$answer" | tr '[:upper:]' '[:lower:]')" in
        y|yes|true|1) printf -v "$varname" '%s' 'yes' ;;
        *)            printf -v "$varname" '%s' 'no' ;;
    esac
}

clone_or_update() {
    local url="$1" dest="$2"
    if [ -d "${dest}/.git" ]; then
        say "updating $(basename "$dest")"
        git -C "$dest" fetch --depth 1 origin HEAD --quiet || warn "could not update ${dest}"
        git -C "$dest" reset --hard FETCH_HEAD --quiet || warn "could not fast-forward ${dest}"
    elif [ -d "$dest" ]; then
        say "reusing $(basename "$dest") (not a git checkout)"
    else
        say "cloning $(basename "$dest")"
        git clone --depth 1 --quiet "$url" "$dest"
    fi
}

# --------------------------------------------------------------- prompting --
if [ "$ASSUME_YES" != "yes" ] && [ -t 0 ]; then
    echo
    echo "paper-experiments benchmark setup -- press enter to accept a default."
    echo
fi

ask "Per-pool time limit (HH:MM:SS)"              "$TIME_LIMIT"    TIME_LIMIT
ask "Per-pool memory limit (e.g. 8G)"             "$MEMORY_LIMIT"  MEMORY_LIMIT
ask "Requested-k filter (blank = every pool)"     "$K_FILTER"      K_FILTER
ask "Slurm partition (blank = site default)"      "$PARTITION"     PARTITION
ask "Slurm account (blank = site default)"        "$ACCOUNT"       ACCOUNT
ask "Slurm QOS (blank = site default)"            "$QOS"           QOS
ask "Max array jobs running at once"              "$MAX_PARALLEL"  MAX_PARALLEL
ask "Exp 2 samples per (m, BDC value) group"      "$SAMPLES"       SAMPLES
ask_yes_no "Exp 3: include the States metric"     "$WITH_STATES"   WITH_STATES

echo
say "venv         ${VENV_DIR}"
say "sandbox      ${SANDBOX_DIR}"
say "tasks        ${TASKS_DIR}"
say "plan pools   ${PLANS_DIR}"
say "limits       ${TIME_LIMIT} / ${MEMORY_LIMIT} per pool"
say "pools        requested-k filter: ${K_FILTER:-all}"
echo

# ------------------------------------------------------------- install -----
if [ "$SKIP_INSTALL" = "yes" ]; then
    say "skipping installation (--skip-install)"
    [ -x "${VENV_DIR}/bin/python" ] || warn "no python in ${VENV_DIR}; generation will fail"
else
    if [ ! -d "$VENV_DIR" ]; then
        command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python interpreter not found: ${PYTHON_BIN}"
        # unified-planning has no wheels on 3.14+ yet; catch it before pip does.
        "$PYTHON_BIN" -c 'import sys; sys.exit(0 if sys.version_info < (3, 14) else 1)' \
            || die "${PYTHON_BIN} is 3.14+; pass PYTHON_BIN=python3.12 (or 3.10-3.13)"
        say "creating virtualenv at ${VENV_DIR}"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        say "reusing virtualenv at ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    . "${VENV_DIR}/bin/activate"
    say "installing behaviour_diversity_counter, plandiversity and matplotlib"
    python -m pip install --quiet --upgrade pip
    python -m pip install --quiet "$REPO_DIR"
    if [ -d "$DIVERSESCORE_DIR" ]; then
        python -m pip install --quiet "$DIVERSESCORE_DIR"
    else
        say "no DiverseScore checkout at ${DIVERSESCORE_DIR}; installing plandiversity from GitHub"
        python -m pip install --quiet "git+${PLANDIVERSITY_REPO}"
    fi
    python -m pip install --quiet matplotlib
    deactivate
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
[ -x "$VENV_PYTHON" ] || die "no python in ${VENV_DIR}"
"$VENV_PYTHON" -c 'import behaviour_diversity_counter, plandiversity' \
    || die "the libraries are not importable from ${VENV_DIR}"

# ------------------------------------------------------------- benchmarks --
if [ "$SKIP_FETCH" = "yes" ]; then
    say "skipping benchmark fetch (--skip-fetch)"
else
    clone_or_update "$CLASSICAL_REPO" "$TASKS_DIR"
    if [ ! -d "$PLANS_DIR" ]; then
        ZIP_FILE="${SCRIPT_DIR}/data/fi-generated-plans-dir.zip"
        [ -f "$ZIP_FILE" ] || die "neither ${PLANS_DIR} nor ${ZIP_FILE} exists"
        say "unpacking $(basename "$ZIP_FILE")"
        unzip -q "$ZIP_FILE" -d "$(dirname "$PLANS_DIR")"
    else
        say "reusing unpacked pools at ${PLANS_DIR}"
    fi
fi
[ -d "$TASKS_DIR" ] || die "no classical-domains checkout at ${TASKS_DIR}"
[ -d "$PLANS_DIR" ] || die "no plan pools at ${PLANS_DIR}"

# --------------------------------------------------------------- generate --
GENERATE_ARGS=(
    "${SCRIPT_DIR}/paperexps/generate_slurm.py"
    --plans-dir "$PLANS_DIR"
    --sandbox-dir "$SANDBOX_DIR"
    --classical-domains "$TASKS_DIR"
    --python "$VENV_PYTHON"
    --time "$TIME_LIMIT"
    --mem "$MEMORY_LIMIT"
    --max-parallel-jobs "$MAX_PARALLEL"
    --samples "$SAMPLES"
    --subset-sizes "$SUBSET_SIZES"
    --select-k "$SELECT_K"
    --exact-sizes "$EXACT_SIZES"
    --exact-cap "$EXACT_CAP"
    --seed "$SEED"
)
[ -n "$K_FILTER" ]           && GENERATE_ARGS+=(--k-filter "$K_FILTER")
[ -n "$PARTITION" ]          && GENERATE_ARGS+=(--partition "$PARTITION")
[ -n "$ACCOUNT" ]            && GENERATE_ARGS+=(--account "$ACCOUNT")
[ -n "$QOS" ]                && GENERATE_ARGS+=(--qos "$QOS")
[ "$WITH_STATES" = "yes" ]   && GENERATE_ARGS+=(--with-states)
[ "$SKIP_EXISTING" = "yes" ] && GENERATE_ARGS+=(--skip-existing)

echo
say "generating run commands"
"$VENV_PYTHON" "${GENERATE_ARGS[@]}"

cat <<EOF

Next steps
  submit the sweep      bash ${SANDBOX_DIR}/slurm/submit_all.sh
  or run it locally     bash ${SANDBOX_DIR}/run_local.sh 8
  watch it              squeue -u \$USER | grep paperexps
  collect the results   ${VENV_PYTHON} ${SCRIPT_DIR}/paperexps/aggregate.py --results-dir ${SANDBOX_DIR}/results --out-dir ${SANDBOX_DIR}/analysis
  draw the figures      ${VENV_PYTHON} ${SCRIPT_DIR}/paperexps/plots.py --analysis-dir ${SANDBOX_DIR}/analysis

  re-generate after a partial run (skips finished pools):
    ./setup_benchmark.sh --skip-install --skip-fetch --skip-existing --yes \\
        --partition '${PARTITION}' --k-filter '${K_FILTER}'
EOF
