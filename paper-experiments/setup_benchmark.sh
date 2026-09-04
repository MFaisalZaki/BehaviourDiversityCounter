#!/usr/bin/env bash
#
# One-shot setup for the paper's evaluation sweep:
#
#   1. create a virtualenv and install the library and the harness,
#   2. fetch classical-domains and unpack the FI plan pools into the sandbox
#      the experiment configuration points at,
#   3. build the slurm job array for one experiment -- one element per task --
#      and submit it, or run the same task list locally.
#
# The configuration file is the single source of truth for where things live:
# this script reads the chosen experiment's `plansdir`, `benchmark`, `ru-info`
# and `dump-dir` out of it and prepares exactly those directories, so the setup
# and the run cannot drift apart.
#
# Everything is prompted with a default, and every prompt has a matching flag,
# so the same script drives an interactive setup and a scripted one (--yes).
# Re-running it is safe: an existing venv, clone or unpacked pool dir is reused.
#
# Usage:
#   ./setup_benchmark.sh                                  # interactive
#   ./setup_benchmark.sh --yes                            # all defaults
#   ./setup_benchmark.sh --experiment runtime --submit --yes
#   ./setup_benchmark.sh --experiment runtime --skip-existing --submit --yes
#   ./setup_benchmark.sh --experiment weights --local-jobs 4 --yes
#   ./setup_benchmark.sh --list-experiments
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../paper-experiments
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # .../BehaviourDiversityCounter
PAPEREXPS="${SCRIPT_DIR}/paperexps"

# ---------------------------------------------------------------- defaults --
VENV_DIR="${REPO_DIR}/venv"
CONFIG_FILE="${PAPEREXPS}/exp-cfg-files/default.json"
EXPERIMENT=""
DEFAULT_EXPERIMENT="runtime"
TIME_LIMIT="01:30:00"
MEMORY_LIMIT="8GB"
CPUS="1"
PARTITION=""
ACCOUNT=""
QOS=""
MAX_PARALLEL="50"
EXTRAS=""
PACKAGES=""
LOCAL_JOBS="0"                 # >0 runs the manifest here instead of on slurm
SKIP_FETCH="no"
SKIP_INSTALL="no"
SKIP_EXISTING="no"
LIST_EXPERIMENTS="no"
SUBMIT="no"
ASSUME_YES="no"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CLASSICAL_REPO="https://github.com/AI-Planning/classical-domains.git"
ZIP_FILE="${SCRIPT_DIR}/data/fi-generated-plans-dir.zip"
RU_INFO_SOURCE="${SCRIPT_DIR}/data/ru-info-dir"

usage() {
    cat <<'EOF'
One-shot setup for the paper's evaluation sweep.

  --experiment NAME       which experiment from the configuration to build
  --config-file PATH      experiment configuration (default: paperexps/exp-cfg-files/default.json)
  --list-experiments      print the experiments the configuration declares and exit
  --venv-dir DIR          virtualenv location          (default: <repo>/venv)

  --time-limit HH:MM:SS   per-task wall clock          (default: 01:30:00)
  --memory-limit SIZE     per-task memory              (default: 8GB)
  --cpus N                cpus per task                (default: 1)
  --partition NAME        slurm partition (blank = site default)
  --account NAME          slurm account   (blank = site default)
  --qos NAME              slurm QOS       (blank = site default)
  --max-parallel N        cap on concurrently running array elements

  --skip-existing         drop tasks whose result file is already written
  --submit                sbatch the array; without this the command is printed
  --local-jobs N          run the task list here with N workers instead of slurm

  --extras LIST           comma-separated package extras to install
  --packages "A B"        extra pip packages
  --skip-fetch            do not clone or unpack anything
  --skip-install          do not touch the virtualenv
  -y, --yes               take every default, ask nothing
  -h, --help              this text
EOF
}

# ------------------------------------------------------------------- args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --experiment)       EXPERIMENT="$2"; shift 2 ;;
        --config-file)      CONFIG_FILE="$2"; shift 2 ;;
        --list-experiments) LIST_EXPERIMENTS="yes"; shift ;;
        --venv-dir)         VENV_DIR="$2"; shift 2 ;;
        --time-limit)       TIME_LIMIT="$2"; shift 2 ;;
        --memory-limit)     MEMORY_LIMIT="$2"; shift 2 ;;
        --cpus)             CPUS="$2"; shift 2 ;;
        --partition)        PARTITION="$2"; shift 2 ;;
        --account)          ACCOUNT="$2"; shift 2 ;;
        --qos)              QOS="$2"; shift 2 ;;
        --max-parallel)     MAX_PARALLEL="$2"; shift 2 ;;
        --skip-existing)    SKIP_EXISTING="yes"; shift ;;
        --submit)           SUBMIT="yes"; shift ;;
        --local-jobs)       LOCAL_JOBS="$2"; shift 2 ;;
        --extras)           EXTRAS="$2"; shift 2 ;;
        --packages)         PACKAGES="$2"; shift 2 ;;
        --skip-fetch)       SKIP_FETCH="yes"; shift ;;
        --skip-install)     SKIP_INSTALL="yes"; shift ;;
        -y|--yes)           ASSUME_YES="yes"; shift ;;
        -h|--help)          usage; exit 0 ;;
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

clone_or_update() {
    local url="$1" dest="$2"
    if [ -d "${dest}/.git" ]; then
        say "reusing the checkout at ${dest}"
    else
        say "cloning ${url}"
        git clone --depth 1 "$url" "$dest"
    fi
}

# `python -` reading the configuration: the config is JSON, and bash should not
# be the thing that parses it.
config_query() {
    "$PYTHON_BIN" - "$CONFIG_FILE" "$@" <<'PY'
import json, os, sys
config_path = sys.argv[1]
mode = sys.argv[2]
with open(config_path) as handle:
    experiments = json.load(handle)['experiments']
if mode == 'names':
    for entry in experiments:
        print(entry['name'])
    raise SystemExit(0)
name = sys.argv[3]
entry = next((e for e in experiments if e['name'] == name), None)
if entry is None:
    raise SystemExit(f"no experiment '{name}' in {config_path}")
here = os.path.dirname(os.path.abspath(config_path))
here = os.path.dirname(here)                      # .../paperexps, what utils resolves against
if mode == 'paths':                               # everything the sweep reads and writes
    for key in ('plansdir', 'benchmark', 'ru-info', 'dump-dir'):
        print(os.path.join(here, entry[key]))
PY
}

[ -f "$CONFIG_FILE" ] || die "no configuration file at ${CONFIG_FILE}"

if [ "$LIST_EXPERIMENTS" = "yes" ]; then
    echo "Experiments in ${CONFIG_FILE}:"
    config_query names | sed 's/^/  /'
    exit 0
fi

# --------------------------------------------------------------- prompting --
if [ -z "$EXPERIMENT" ]; then
    ask "Experiment to build ($(config_query names | paste -sd'/' -))" \
        "$DEFAULT_EXPERIMENT" EXPERIMENT
fi
EXPERIMENT_NAMES="$(config_query names)"
case "
${EXPERIMENT_NAMES}
" in
    *"
${EXPERIMENT}
"*) ;;
    *) die "no experiment '${EXPERIMENT}' in ${CONFIG_FILE}" ;;
esac

ask "Virtualenv directory"                       "$VENV_DIR"      VENV_DIR
ask "Per-task time limit"                        "$TIME_LIMIT"    TIME_LIMIT
ask "Per-task memory limit"                      "$MEMORY_LIMIT"  MEMORY_LIMIT
ask "Slurm partition (blank = site default)"     "$PARTITION"     PARTITION
ask "Slurm account (blank = site default)"       "$ACCOUNT"       ACCOUNT
ask "Slurm QOS (blank = site default)"           "$QOS"           QOS
ask "Max array elements running at once"         "$MAX_PARALLEL"  MAX_PARALLEL

# The configuration says where everything lives; read it rather than guess.
# Read line by line rather than with mapfile: bash 3.2, which is what macOS
# ships, does not have it.
PLANS_DIR=""; TASKS_DIR=""; RU_INFO_DIR=""; DUMP_DIR=""
path_index=0
while IFS= read -r path_line; do
    case "$path_index" in
        0) PLANS_DIR="$path_line" ;;
        1) TASKS_DIR="$path_line" ;;
        2) RU_INFO_DIR="$path_line" ;;
        3) DUMP_DIR="$path_line" ;;
    esac
    path_index=$((path_index + 1))
done < <(config_query paths "$EXPERIMENT")
[ -n "$DUMP_DIR" ] || die "could not read the paths for '${EXPERIMENT}'"
JOBS_DIR="$(dirname "$DUMP_DIR")/slurm"
LOG_DIR="${JOBS_DIR}/logs"

say "experiment '${EXPERIMENT}' reads and writes"
printf '      pools     %s\n      benchmark %s\n      ru-info   %s\n      results   %s\n' \
    "$PLANS_DIR" "$TASKS_DIR" "$RU_INFO_DIR" "$DUMP_DIR"

# --------------------------------------------------------------- install ---
if [ "$SKIP_INSTALL" = "yes" ]; then
    say "skipping installation (--skip-install)"
else
    if [ ! -d "$VENV_DIR" ]; then
        say "creating virtualenv at ${VENV_DIR}"
        "$PYTHON_BIN" -m venv "$VENV_DIR"
    else
        say "reusing virtualenv at ${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    . "${VENV_DIR}/bin/activate"
    python -m pip install --quiet --upgrade pip setuptools wheel
    if [ -n "$EXTRAS" ]; then
        say "installing the library and the harness with extras: ${EXTRAS}"
        python -m pip install --quiet -e "${REPO_DIR}[${EXTRAS}]"
    else
        say "installing the library and the harness"
        python -m pip install --quiet -e "${REPO_DIR}"
    fi
    if [ -n "$PACKAGES" ]; then
        say "installing extra packages: ${PACKAGES}"
        # word-split deliberately: --packages takes a space-separated list
        # shellcheck disable=SC2086
        python -m pip install --quiet $PACKAGES
    fi
    deactivate
fi

VENV_PYTHON="${VENV_DIR}/bin/python"
[ -x "$VENV_PYTHON" ] || die "no python at ${VENV_PYTHON}"

# ------------------------------------------------------------- benchmarks --
if [ "$SKIP_FETCH" != "yes" ]; then
    mkdir -p "$(dirname "$TASKS_DIR")"
    clone_or_update "$CLASSICAL_REPO" "$TASKS_DIR"

    if [ ! -d "$PLANS_DIR" ]; then
        [ -f "$ZIP_FILE" ] || die "no plan pools: neither ${PLANS_DIR} nor ${ZIP_FILE}"
        say "unpacking the FI plan pools"
        command -v unzip >/dev/null 2>&1 || die "unzip is needed to unpack ${ZIP_FILE}"
        mkdir -p "$(dirname "$PLANS_DIR")"
        unzip -q "$ZIP_FILE" -d "$(dirname "$PLANS_DIR")"
    else
        say "reusing the unpacked pools at ${PLANS_DIR}"
    fi

    if [ ! -d "$RU_INFO_DIR" ] || [ -z "$(ls -A "$RU_INFO_DIR" 2>/dev/null)" ]; then
        [ -d "$RU_INFO_SOURCE" ] || die "no ru-info to copy from ${RU_INFO_SOURCE}"
        say "copying ru-info into ${RU_INFO_DIR}"
        mkdir -p "$RU_INFO_DIR"
        cp -R "${RU_INFO_SOURCE}/." "$RU_INFO_DIR/"
    else
        say "reusing ru-info at ${RU_INFO_DIR}"
    fi
fi
[ -d "$TASKS_DIR" ]   || die "no classical-domains checkout at ${TASKS_DIR}"
[ -d "$PLANS_DIR" ]   || die "no unpacked plan pools at ${PLANS_DIR}"
[ -d "$RU_INFO_DIR" ] || die "no ru-info directory at ${RU_INFO_DIR}"
mkdir -p "$DUMP_DIR" "$JOBS_DIR" "$LOG_DIR"

# ------------------------------------------------------- the task manifest --
# One array element runs one task, because a task is the unit that writes one
# result file: an element that dies takes its own task down and no other, and
# --skip-existing then rebuilds a manifest of exactly what is still missing.
MANIFEST="${JOBS_DIR}/${EXPERIMENT}.manifest"
ALL_TASKS="${JOBS_DIR}/${EXPERIMENT}.all.tsv"

say "listing the tasks for '${EXPERIMENT}'"
( cd "$PAPEREXPS" && "$VENV_PYTHON" runnner.py \
      --config-file "$CONFIG_FILE" --experiment-name "$EXPERIMENT" --list-tasks ) > "$ALL_TASKS"
TOTAL="$(wc -l < "$ALL_TASKS" | tr -d ' ')"
[ "$TOTAL" -gt 0 ] || die "the configuration matched no tasks"

if [ "$SKIP_EXISTING" = "yes" ]; then
    : > "$MANIFEST"
    while IFS=$'\t' read -r task dumpfile; do
        [ -s "${DUMP_DIR}/${dumpfile}" ] || printf '%s\t%s\n' "$task" "$dumpfile" >> "$MANIFEST"
    done < "$ALL_TASKS"
else
    cp "$ALL_TASKS" "$MANIFEST"
fi

COUNT="$(wc -l < "$MANIFEST" | tr -d ' ')"
say "${COUNT} of ${TOTAL} tasks to run"
if [ "$COUNT" -eq 0 ]; then
    say "nothing to do"
    exit 0
fi

# ------------------------------------------------------------ run locally --
if [ "$LOCAL_JOBS" -gt 0 ] 2>/dev/null; then
    say "running ${COUNT} tasks locally with ${LOCAL_JOBS} workers"
    warn "the runtime experiment measures wall clock: use --local-jobs 1, or a node"
    warn "that is not sharing its CPUs, or its numbers are of the machine's load."
    cut -f1 "$MANIFEST" | ( cd "$PAPEREXPS" && xargs -P "$LOCAL_JOBS" -I{} \
        "$VENV_PYTHON" runnner.py --config-file "$CONFIG_FILE" \
            --experiment-name "$EXPERIMENT" --task-id {} )
    say "done; results in ${DUMP_DIR}"
    exit 0
fi

# A site's MaxArraySize caps the highest index, not the element count. Check it
# here rather than letting sbatch reject the whole submission.
if command -v scontrol >/dev/null 2>&1; then
    MAX_ARRAY="$(scontrol show config 2>/dev/null | awk -F'= *' '/^MaxArraySize/ {print $2}' | tr -d ' ')"
    if [ -n "${MAX_ARRAY:-}" ] && [ "$COUNT" -ge "$MAX_ARRAY" ]; then
        die "${COUNT} elements exceeds this site's MaxArraySize of ${MAX_ARRAY}; split the manifest"
    fi
fi

# ------------------------------------------------------------ the job file --
SBATCH_FILE="${JOBS_DIR}/${EXPERIMENT}.sbatch"
{
    echo '#!/usr/bin/env bash'
    echo "#SBATCH --job-name=bdc-${EXPERIMENT}"
    echo "#SBATCH --array=1-${COUNT}%${MAX_PARALLEL}"
    echo "#SBATCH --time=${TIME_LIMIT}"
    echo "#SBATCH --mem=${MEMORY_LIMIT}"
    echo "#SBATCH --cpus-per-task=${CPUS}"
    echo "#SBATCH --output=${LOG_DIR}/%x-%A_%a.out"
    echo "#SBATCH --error=${LOG_DIR}/%x-%A_%a.err"
    if [ -n "$PARTITION" ]; then echo "#SBATCH --partition=${PARTITION}"; fi
    if [ -n "$ACCOUNT" ];   then echo "#SBATCH --account=${ACCOUNT}";     fi
    if [ -n "$QOS" ];       then echo "#SBATCH --qos=${QOS}";             fi
    cat <<EOF
#
# Generated by setup_benchmark.sh -- re-run it rather than editing this.
# One element, one task, one result file.
#
set -euo pipefail

MANIFEST="${MANIFEST}"
PAPEREXPS="${PAPEREXPS}"
PYTHON="${VENV_PYTHON}"
CONFIG_FILE="${CONFIG_FILE}"
EXPERIMENT="${EXPERIMENT}"

line="\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "\$MANIFEST")"
[ -n "\$line" ] || { echo "no task at manifest line \${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
task_id="\${line%%\$'\t'*}"

echo "[\$(date -Is)] \${SLURM_ARRAY_TASK_ID}/${COUNT}  \${task_id}"
cd "\$PAPEREXPS"
# exec so slurm's signals reach python rather than this wrapper: a timeout then
# kills the task itself, and the element's exit status is python's own.
exec "\$PYTHON" runnner.py \\
    --config-file "\$CONFIG_FILE" \\
    --experiment-name "\$EXPERIMENT" \\
    --task-id "\$task_id"
EOF
} > "$SBATCH_FILE"
chmod +x "$SBATCH_FILE"

say "manifest ${MANIFEST}"
say "job file ${SBATCH_FILE}"
say "logs     ${LOG_DIR}"

if [ "$SUBMIT" = "yes" ]; then
    command -v sbatch >/dev/null 2>&1 || die "sbatch not found on this machine"
    sbatch "$SBATCH_FILE"
else
    cat <<EOF

Next steps
  submit the sweep    sbatch ${SBATCH_FILE}
  watch it            squeue -u \$USER
  resume a partial run
                      ${BASH_SOURCE[0]} --experiment ${EXPERIMENT} --skip-existing --submit --yes
  run it here instead ${BASH_SOURCE[0]} --experiment ${EXPERIMENT} --local-jobs 4 --yes

  The runtime experiment measures wall clock: give it a node that is not
  sharing its CPUs, or its numbers are of the machine's load.
EOF
fi
