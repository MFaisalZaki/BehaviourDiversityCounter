#!/usr/bin/env bash
#
# One-shot setup for the paper's evaluation sweep:
#
#   1. create a virtualenv and install the library and the harness,
#   2. fetch classical-domains and unpack the FI plan pools into the sandbox
#      the experiment configuration points at,
#   3. build a slurm job array per experiment -- one element per task -- and
#      submit them, or run the same task lists locally.
#
# Every experiment the configuration declares is built unless --experiment
# narrows it, because the sweep is the whole set: building one and forgetting
# the other is the failure this defaults away from.
#
# The configuration file is the single source of truth for where things live:
# this script reads each experiment's `plansdir`, `benchmark`, `ru-info` and
# `dump-dir` out of it and prepares exactly those directories, so the setup and
# the run cannot drift apart.
#
# Everything is prompted with a default, and every prompt has a matching flag,
# so the same script drives an interactive setup and a scripted one (--yes).
# Re-running it is safe: an existing venv, clone or unpacked pool dir is reused.
#
# Usage:
#   ./setup_benchmark.sh                                  # interactive, all experiments
#   ./setup_benchmark.sh --yes                            # all defaults, all experiments
#   ./setup_benchmark.sh --submit --yes                   # build and submit every experiment
#   ./setup_benchmark.sh --experiment runtime --submit --yes
#   ./setup_benchmark.sh --skip-existing --submit --yes   # resume a partial sweep
#   ./setup_benchmark.sh --local-jobs 4 --yes
#   ./setup_benchmark.sh --list-experiments
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../paper-experiments
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # .../BehaviourDiversityCounter
PAPEREXPS="${SCRIPT_DIR}/paperexps"

# ---------------------------------------------------------------- defaults --
VENV_DIR="${REPO_DIR}/venv"
CONFIG_FILE="${PAPEREXPS}/exp-cfg-files/default.json"
EXPERIMENT=""                  # blank or "all" builds every declared experiment
TIME_LIMIT="01:30:00"
MEMORY_LIMIT="8GB"
CPUS="1"
PARTITION=""
ACCOUNT=""
QOS=""
MAX_PARALLEL="50"
EXTRAS=""
PACKAGES=""
LOCAL_JOBS="0"                 # >0 runs the manifests here instead of on slurm
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
One-shot setup for the paper's evaluation sweep. Builds every experiment the
configuration declares unless --experiment narrows it.

  --experiment NAME       build only this one ("all" is the default)
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
  --submit                sbatch each array; without this the commands are printed
  --local-jobs N          run the task lists here with N workers instead of slurm

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
here = os.path.dirname(os.path.dirname(os.path.abspath(config_path)))   # .../paperexps
if mode == 'paths':                               # everything the sweep reads and writes
    for key in ('plansdir', 'benchmark', 'ru-info', 'dump-dir'):
        print(os.path.join(here, entry[key]))
PY
}

# declared <name> -- is this one of the configuration's experiments?
declared() {
    case "
${EXPERIMENT_NAMES}
" in
        *"
$1
"*) return 0 ;;
        *) return 1 ;;
    esac
}

[ -f "$CONFIG_FILE" ] || die "no configuration file at ${CONFIG_FILE}"
EXPERIMENT_NAMES="$(config_query names)"
[ -n "$EXPERIMENT_NAMES" ] || die "${CONFIG_FILE} declares no experiments"

if [ "$LIST_EXPERIMENTS" = "yes" ]; then
    echo "Experiments in ${CONFIG_FILE}:"
    printf '%s\n' "$EXPERIMENT_NAMES" | sed 's/^/  /'
    exit 0
fi

# --------------------------------------------------------------- prompting --
if [ -z "$EXPERIMENT" ]; then
    ask "Experiments to build (all, or one of: $(printf '%s' "$EXPERIMENT_NAMES" | paste -sd'/' -))" \
        "all" EXPERIMENT
fi
if [ "$EXPERIMENT" = "all" ]; then
    SELECTED="$EXPERIMENT_NAMES"
else
    declared "$EXPERIMENT" || die "no experiment '${EXPERIMENT}' in ${CONFIG_FILE}"
    SELECTED="$EXPERIMENT"
fi

ask "Virtualenv directory"                       "$VENV_DIR"      VENV_DIR
ask "Per-task time limit"                        "$TIME_LIMIT"    TIME_LIMIT
ask "Per-task memory limit"                      "$MEMORY_LIMIT"  MEMORY_LIMIT
ask "Slurm partition (blank = site default)"     "$PARTITION"     PARTITION
ask "Slurm account (blank = site default)"       "$ACCOUNT"       ACCOUNT
ask "Slurm QOS (blank = site default)"           "$QOS"           QOS
ask "Max array elements running at once"         "$MAX_PARALLEL"  MAX_PARALLEL

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

# ------------------------------------------------------ one experiment --
# Everything below is per experiment: each writes its own manifest and its own
# array, so one can be resubmitted without disturbing the other.
GENERATED=""

build_experiment() {
    local name="$1"
    local plans_dir tasks_dir ru_info_dir dump_dir jobs_dir log_dir
    local manifest all_tasks total count sbatch_file max_array path_line path_index

    echo
    say "=== ${name} ==="

    # Read line by line rather than with mapfile: bash 3.2, which is what macOS
    # ships, does not have it.
    plans_dir=""; tasks_dir=""; ru_info_dir=""; dump_dir=""
    path_index=0
    while IFS= read -r path_line; do
        case "$path_index" in
            0) plans_dir="$path_line" ;;
            1) tasks_dir="$path_line" ;;
            2) ru_info_dir="$path_line" ;;
            3) dump_dir="$path_line" ;;
        esac
        path_index=$((path_index + 1))
    done < <(config_query paths "$name")
    [ -n "$dump_dir" ] || die "could not read the paths for '${name}'"

    jobs_dir="$(dirname "$dump_dir")/slurm"
    log_dir="${jobs_dir}/logs"
    printf '      pools     %s\n      benchmark %s\n      ru-info   %s\n      results   %s\n' \
        "$plans_dir" "$tasks_dir" "$ru_info_dir" "$dump_dir"

    # Idempotent, so two experiments sharing a sandbox cost one fetch.
    if [ "$SKIP_FETCH" != "yes" ]; then
        mkdir -p "$(dirname "$tasks_dir")"
        clone_or_update "$CLASSICAL_REPO" "$tasks_dir"
        if [ ! -d "$plans_dir" ]; then
            [ -f "$ZIP_FILE" ] || die "no plan pools: neither ${plans_dir} nor ${ZIP_FILE}"
            say "unpacking the FI plan pools"
            command -v unzip >/dev/null 2>&1 || die "unzip is needed to unpack ${ZIP_FILE}"
            mkdir -p "$(dirname "$plans_dir")"
            unzip -q "$ZIP_FILE" -d "$(dirname "$plans_dir")"
        fi
        if [ ! -d "$ru_info_dir" ] || [ -z "$(ls -A "$ru_info_dir" 2>/dev/null)" ]; then
            [ -d "$RU_INFO_SOURCE" ] || die "no ru-info to copy from ${RU_INFO_SOURCE}"
            say "copying ru-info into ${ru_info_dir}"
            mkdir -p "$ru_info_dir"
            cp -R "${RU_INFO_SOURCE}/." "$ru_info_dir/"
        fi
    fi
    [ -d "$tasks_dir" ]   || die "no classical-domains checkout at ${tasks_dir}"
    [ -d "$plans_dir" ]   || die "no unpacked plan pools at ${plans_dir}"
    [ -d "$ru_info_dir" ] || die "no ru-info directory at ${ru_info_dir}"
    mkdir -p "$dump_dir" "$jobs_dir" "$log_dir"

    # --------------------------------------------------- the task manifest --
    # One array element runs one task, because a task is the unit that writes
    # one result file: an element that dies takes its own task down and no
    # other, and --skip-existing then rebuilds a manifest of what is missing.
    manifest="${jobs_dir}/${name}.manifest"
    all_tasks="${jobs_dir}/${name}.all.tsv"

    say "listing the tasks"
    ( cd "$PAPEREXPS" && "$VENV_PYTHON" runnner.py \
          --config-file "$CONFIG_FILE" --experiment-name "$name" --list-tasks ) > "$all_tasks"
    total="$(wc -l < "$all_tasks" | tr -d ' ')"
    [ "$total" -gt 0 ] || die "'${name}' matched no tasks"

    if [ "$SKIP_EXISTING" = "yes" ]; then
        : > "$manifest"
        while IFS=$'\t' read -r task dumpfile; do
            [ -s "${dump_dir}/${dumpfile}" ] || printf '%s\t%s\n' "$task" "$dumpfile" >> "$manifest"
        done < "$all_tasks"
    else
        cp "$all_tasks" "$manifest"
    fi

    count="$(wc -l < "$manifest" | tr -d ' ')"
    say "${count} of ${total} tasks to run"
    if [ "$count" -eq 0 ]; then
        say "nothing to do for '${name}'"
        return 0
    fi

    # ---------------------------------------------------------- run here --
    if [ "$LOCAL_JOBS" -gt 0 ] 2>/dev/null; then
        say "running ${count} tasks locally with ${LOCAL_JOBS} workers"
        cut -f1 "$manifest" | ( cd "$PAPEREXPS" && xargs -P "$LOCAL_JOBS" -I{} \
            "$VENV_PYTHON" runnner.py --config-file "$CONFIG_FILE" \
                --experiment-name "$name" --task-id {} )
        say "done; results in ${dump_dir}"
        return 0
    fi

    # A site's MaxArraySize caps the highest index, not the element count.
    # Check it here rather than letting sbatch reject the whole submission.
    if command -v scontrol >/dev/null 2>&1; then
        max_array="$(scontrol show config 2>/dev/null | awk -F'= *' '/^MaxArraySize/ {print $2}' | tr -d ' ')"
        if [ -n "${max_array:-}" ] && [ "$count" -ge "$max_array" ]; then
            die "${count} elements exceeds this site's MaxArraySize of ${max_array}; split the manifest"
        fi
    fi

    # -------------------------------------------------------- the job file --
    sbatch_file="${jobs_dir}/${name}.sbatch"
    {
        echo '#!/usr/bin/env bash'
        echo "#SBATCH --job-name=bdc-${name}"
        echo "#SBATCH --array=1-${count}%${MAX_PARALLEL}"
        echo "#SBATCH --time=${TIME_LIMIT}"
        echo "#SBATCH --mem=${MEMORY_LIMIT}"
        echo "#SBATCH --cpus-per-task=${CPUS}"
        echo "#SBATCH --output=${log_dir}/%x-%A_%a.out"
        echo "#SBATCH --error=${log_dir}/%x-%A_%a.err"
        if [ -n "$PARTITION" ]; then echo "#SBATCH --partition=${PARTITION}"; fi
        if [ -n "$ACCOUNT" ];   then echo "#SBATCH --account=${ACCOUNT}";     fi
        if [ -n "$QOS" ];       then echo "#SBATCH --qos=${QOS}";             fi
        cat <<EOF
#
# Generated by setup_benchmark.sh -- re-run it rather than editing this.
# One element, one task, one result file.
#
set -euo pipefail

MANIFEST="${manifest}"
PAPEREXPS="${PAPEREXPS}"
PYTHON="${VENV_PYTHON}"
CONFIG_FILE="${CONFIG_FILE}"
EXPERIMENT="${name}"

line="\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "\$MANIFEST")"
[ -n "\$line" ] || { echo "no task at manifest line \${SLURM_ARRAY_TASK_ID}" >&2; exit 1; }
task_id="\${line%%\$'\t'*}"

echo "[\$(date -Is)] \${SLURM_ARRAY_TASK_ID}/${count}  \${task_id}"
cd "\$PAPEREXPS"
# exec so slurm's signals reach python rather than this wrapper: a timeout then
# kills the task itself, and the element's exit status is python's own.
exec "\$PYTHON" runnner.py \\
    --config-file "\$CONFIG_FILE" \\
    --experiment-name "\$EXPERIMENT" \\
    --task-id "\$task_id"
EOF
    } > "$sbatch_file"
    chmod +x "$sbatch_file"

    say "manifest ${manifest}"
    say "job file ${sbatch_file}"
    GENERATED="${GENERATED}${sbatch_file}
"

    if [ "$SUBMIT" = "yes" ]; then
        command -v sbatch >/dev/null 2>&1 || die "sbatch not found on this machine"
        sbatch "$sbatch_file"
    fi
}

for experiment_name in $SELECTED; do
    build_experiment "$experiment_name"
done

# ------------------------------------------------------------ next steps --
if [ "$LOCAL_JOBS" -gt 0 ] 2>/dev/null; then
    echo
    say "all requested experiments have been run locally"
    warn "the runtime experiment measures wall clock: with more than one worker,"
    warn "or on a node sharing its CPUs, its numbers are of the machine's load."
    exit 0
fi

if [ -z "$GENERATED" ]; then
    echo
    say "nothing to submit"
    exit 0
fi

if [ "$SUBMIT" != "yes" ]; then
    echo
    echo "Next steps"
    echo "  submit the sweep"
    printf '%s' "$GENERATED" | sed '/^$/d; s/^/    sbatch /'
    cat <<EOF
  watch it            squeue -u \$USER
  resume a partial run
                      ${BASH_SOURCE[0]} --skip-existing --submit --yes
  run it here instead ${BASH_SOURCE[0]} --local-jobs 4 --yes

  The runtime experiment measures wall clock: give it a node that is not
  sharing its CPUs, or its numbers are of the machine's load.
EOF
fi
