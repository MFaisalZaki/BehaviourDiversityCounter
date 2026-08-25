#!/usr/bin/env bash
#
# One-shot setup for the paper's evaluation sweep:
#
#   1. ask which experiment configuration to run,
#   2. ask for the per-pool time and memory limits (and a few other knobs),
#   3. create a virtualenv and install the library, the harness and any extra
#      python packages passed with --packages,
#   4. fetch classical-domains and unpack the FI plan pools,
#   5. write the experiment configuration with those limits,
#   6. generate one run command per (experiment, pool) and the slurm job arrays.
#
# Everything is prompted with a default, and every prompt has a matching flag,
# so the same script drives an interactive setup and a scripted one (--yes).
# Re-running it is safe: an existing venv, clone, unpacked pool dir or
# experiment is reused.
#
# Usage:
#   ./setup_benchmark.sh                      # interactive
#   ./setup_benchmark.sh --yes                # all defaults, no prompts
#   ./setup_benchmark.sh --config smoke --yes
#   ./setup_benchmark.sh --config discriminating --time-limit 30m --yes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../paper-experiments
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # .../BehaviourDiversityCounter

# ---------------------------------------------------------------- defaults --
WORK_DIR="${SCRIPT_DIR}/benchmark-run"
VENV_DIR=""
TASKS_DIR=""
PLANS_DIR=""
SANDBOX_DIR=""
EXP_DIR=""
CONFIG_ROOT="${SCRIPT_DIR}/exp-configurations"
CONFIG=""                      # name under exp-configurations/, or a path
DEFAULT_CONFIG="default"
TIME_LIMIT="01:30:00"
MEMORY_LIMIT="8GB"
PARTITION=""
ACCOUNT=""
QOS=""
MAX_PARALLEL="50"
EXTRAS=""
PACKAGES=""
LOCAL_JOBS="4"
SKIP_FETCH="no"
SKIP_INSTALL="no"
SKIP_EXISTING="no"
LIST_CONFIGS="no"
ASSUME_YES="no"
PYTHON_BIN="${PYTHON_BIN:-python3}"

CLASSICAL_REPO="https://github.com/AI-Planning/classical-domains.git"
ZIP_FILE="${SCRIPT_DIR}/data/fi-generated-plans-dir.zip"

usage() {
    cat <<'EOF'
One-shot setup for the paper's evaluation sweep.

Usage:
  ./setup_benchmark.sh                      # interactive; asks which configuration to run
  ./setup_benchmark.sh --yes                # all defaults, no prompts
  ./setup_benchmark.sh --config smoke --yes
  ./setup_benchmark.sh --config discriminating --time-limit 30m --yes

Options:
  --config NAME|DIR       which exp-configurations/ entry to run; a bare name is
                          looked up under exp-configurations/, a path is used as
                          given. Asked interactively when omitted (default: default)
  --list-configs          print the available configurations and exit
  --work-dir DIR          root for venv/tasks/sandbox   (default: <here>/benchmark-run)
  --venv-dir DIR          virtualenv location           (default: <work-dir>/venv)
  --tasks-dir DIR         classical-domains checkout    (default: <work-dir>/classical-domains)
  --plans-dir DIR         unpacked FI pools             (default: <here>/data/fi-generated-plans-dir)
  --sandbox-dir DIR       commands, results, logs       (default: <work-dir>/sandbox/<config>)
  --exp-dir DIR           the experiment to run, seeded from the configuration
                          (default: <work-dir>/experiments/<config>)
  --time-limit VALUE      per pool, e.g. 01:30:00 or 90m
  --memory-limit VALUE    per pool, e.g. 8GB
  --partition NAME        slurm partition
  --account NAME          slurm account
  --qos NAME              slurm QOS
  --max-parallel N        cap on concurrently running array jobs
  --local-jobs N          parallelism baked into run_local.sh (default: 4).
                          Use 1 for an Experiment A timing run: anything else on
                          the machine inflates what it is measuring
  --extras "a,b"          extras to install on top of the defaults
                          (available: plots, stats, analysis)
  --packages "a b"        extra python packages to pip install into the venv,
                          space or comma separated. Anything pip accepts works,
                          and a local checkout given as a directory is installed
                          editable (-e) so a library under development stays live
  --skip-existing         do not regenerate commands for pools that have results
  --skip-fetch            do not clone classical-domains or unzip the pools
  --skip-install          do not create the venv or install anything
  -y, --yes               accept every default, never prompt
  -h, --help              this message
EOF
}

# ------------------------------------------------------------------- args ---
while [ $# -gt 0 ]; do
    case "$1" in
        --config)          CONFIG="$2"; shift 2 ;;
        --list-configs)    LIST_CONFIGS="yes"; shift ;;
        --work-dir)        WORK_DIR="$2"; shift 2 ;;
        --venv-dir)        VENV_DIR="$2"; shift 2 ;;
        --tasks-dir)       TASKS_DIR="$2"; shift 2 ;;
        --plans-dir)       PLANS_DIR="$2"; shift 2 ;;
        --sandbox-dir)     SANDBOX_DIR="$2"; shift 2 ;;
        --exp-dir)         EXP_DIR="$2"; shift 2 ;;
        --time-limit)      TIME_LIMIT="$2"; shift 2 ;;
        --memory-limit)    MEMORY_LIMIT="$2"; shift 2 ;;
        --partition)       PARTITION="$2"; shift 2 ;;
        --account)         ACCOUNT="$2"; shift 2 ;;
        --qos)             QOS="$2"; shift 2 ;;
        --max-parallel)    MAX_PARALLEL="$2"; shift 2 ;;
        --local-jobs)      LOCAL_JOBS="$2"; shift 2 ;;
        --extras)          EXTRAS="$2"; shift 2 ;;
        --packages)        PACKAGES="$2"; shift 2 ;;
        --skip-existing)   SKIP_EXISTING="yes"; shift ;;
        --skip-fetch)      SKIP_FETCH="yes"; shift ;;
        --skip-install)    SKIP_INSTALL="yes"; shift ;;
        -y|--yes)          ASSUME_YES="yes"; shift ;;
        -h|--help)         usage; exit 0 ;;
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

# ------------------------------------------------------------ configurations --
# A configuration is a directory holding exp-details.json -- the same shape the
# harness runs. The ones shipped in exp-configurations/ are offered by name; any
# other directory works too, given as a path.

config_names() {
    local dir
    for dir in "${CONFIG_ROOT}"/*/; do
        [ -f "${dir}exp-details.json" ] || continue
        basename "${dir%/}"
    done
}

# "a+b+c over 918 pools" for a configuration directory.
config_summary() {
    "$PYTHON_BIN" - "$1" <<'PY' 2>/dev/null || echo "(unreadable)"
import json, sys
details = json.load(open(sys.argv[1] + '/exp-details.json'))
experiments = details.get('experiments', {})
enabled = [name for name in ('a', 'b', 'c')
           if experiments.get(name, {}).get('enabled', False)]
pools = details.get('pools', {})
bits = [f"experiments {'+'.join(enabled) or 'none'}",
        f"k = {experiments.get('b', {}).get('k', 20)}"]
if pools.get('min-behaviour-count'):
    bits.append(f"behaviour-count >= {pools['min-behaviour-count']}")
if pools.get('include-domains'):
    bits.append('domains: ' + ','.join(pools['include-domains']))
if pools.get('max-pools'):
    bits.append(f"at most {pools['max-pools']} pools")
print(', '.join(bits))
PY
}

config_dir() {
    case "$1" in
        */*|.|..) printf '%s' "$1" ;;
        *)        printf '%s' "${CONFIG_ROOT}/$1" ;;
    esac
}

list_configs() {
    local name
    echo "Configurations in ${CONFIG_ROOT}:"
    for name in $(config_names); do
        printf '  %-16s %s\n' "$name" "$(config_summary "${CONFIG_ROOT}/${name}")"
    done
}

# Ask which configuration to run and set CONFIG. Accepts its number, its name,
# or a path. Sets the global rather than printing it: the menu goes to stdout,
# so a $(...) capture would swallow it.
choose_config() {
    # `reply`, not `answer`: `ask` has a local of that name, and dynamic scoping
    # would have it assign to its own copy instead of this one.
    local names name count reply index=1
    names="$(config_names)"
    count=$(printf '%s\n' "$names" | sed '/^$/d' | wc -l | tr -d ' ')
    [ "$count" -gt 0 ] || die "no configuration in ${CONFIG_ROOT}"

    echo "Available experiment configurations:"
    for name in $names; do
        printf '  %d) %-16s %s\n' "$index" "$name" "$(config_summary "${CONFIG_ROOT}/${name}")"
        index=$((index + 1))
    done
    echo "  (or the path to a configuration directory of your own)"
    echo

    ask "Experiment configuration" "$DEFAULT_CONFIG" reply
    case "$reply" in
        ''|*[!0-9]*) CONFIG="$reply" ;;
        *)           CONFIG="$(printf '%s\n' "$names" | sed -n "${reply}p")"
                     [ -n "$CONFIG" ] || die "no configuration number ${reply}" ;;
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

if [ "$LIST_CONFIGS" = "yes" ]; then
    list_configs
    exit 0
fi

# --------------------------------------------------------------- prompting --
if [ "$ASSUME_YES" != "yes" ] && [ -t 0 ]; then
    echo
    echo "Behaviour-diversity evaluation setup -- press enter to accept a default."
    echo
fi

# The configuration comes first: it decides which pools are swept and which of
# the three experiments run.
if [ -z "$CONFIG" ]; then
    if [ "$ASSUME_YES" = "yes" ] || [ ! -t 0 ]; then
        CONFIG="$DEFAULT_CONFIG"
    else
        choose_config
    fi
fi

TEMPLATE_DIR="$(config_dir "$CONFIG")"
CONFIG_NAME="$(basename "${TEMPLATE_DIR%/}")"
[ -d "$TEMPLATE_DIR" ] || die "no such configuration: ${CONFIG} (tried ${TEMPLATE_DIR}); \
run with --list-configs to see the shipped ones"
[ -f "${TEMPLATE_DIR}/exp-details.json" ] || \
    die "${TEMPLATE_DIR} holds no exp-details.json, so it is not a configuration"

ask "Experiment root (venv, tasks, commands, results)" "$WORK_DIR" WORK_DIR
VENV_DIR="${VENV_DIR:-${WORK_DIR}/venv}"
TASKS_DIR="${TASKS_DIR:-${WORK_DIR}/classical-domains}"
PLANS_DIR="${PLANS_DIR:-${SCRIPT_DIR}/data/fi-generated-plans-dir}"

# One experiment and one sandbox per configuration: pointing a second
# configuration at the same directory would mix their results under one name.
SANDBOX_DIR="${SANDBOX_DIR:-${WORK_DIR}/sandbox/${CONFIG_NAME}}"
EXP_DIR="${EXP_DIR:-${WORK_DIR}/experiments/${CONFIG_NAME}}"

ask "Per-pool time limit (HH:MM:SS or 90m)"      "$TIME_LIMIT"    TIME_LIMIT
ask "Per-pool memory limit (e.g. 8GB)"           "$MEMORY_LIMIT"  MEMORY_LIMIT
ask "Slurm partition (blank = site default)"     "$PARTITION"     PARTITION
ask "Slurm account (blank = site default)"       "$ACCOUNT"       ACCOUNT
ask "Slurm QOS (blank = site default)"           "$QOS"           QOS
ask "Max array jobs running at once"             "$MAX_PARALLEL"  MAX_PARALLEL
ask "Parallelism for run_local.sh (1 for timing)" "$LOCAL_JOBS"   LOCAL_JOBS
ask "Extra python packages for the venv (blank = none)" "$PACKAGES" PACKAGES

# Commas and spaces both separate packages.
PACKAGE_LIST="$(printf '%s' "$PACKAGES" | tr ',' ' ')"

# An entry that is a directory is a local checkout under development, so it is
# installed editable; everything else goes to pip as given.
PACKAGE_ARGS=()
NEXT_IS_EDITABLE="no"
for pkg in $PACKAGE_LIST; do
    case "$pkg" in
        -e|--editable) NEXT_IS_EDITABLE="yes"; continue ;;
        "~/"*) pkg="${HOME}/${pkg#\~/}" ;;
    esac
    if [ "$NEXT_IS_EDITABLE" = "yes" ] || [ -d "$pkg" ]; then
        PACKAGE_ARGS+=(-e "$pkg")
    else
        PACKAGE_ARGS+=("$pkg")
    fi
    NEXT_IS_EDITABLE="no"
done

echo
say "config       ${CONFIG_NAME} (${TEMPLATE_DIR})"
say "             $(config_summary "$TEMPLATE_DIR")"
say "work dir     ${WORK_DIR}"
say "venv         ${VENV_DIR}"
say "tasks        ${TASKS_DIR}"
say "pools        ${PLANS_DIR}"
say "sandbox      ${SANDBOX_DIR}"
say "experiment   ${EXP_DIR}"
say "limits       ${TIME_LIMIT} / ${MEMORY_LIMIT} per pool"
[ ${#PACKAGE_ARGS[@]} -gt 0 ] && say "packages     ${PACKAGE_ARGS[*]}"
echo

mkdir -p "$WORK_DIR"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "python interpreter not found: ${PYTHON_BIN}"

# ------------------------------------------------------------- experiment --
mkdir -p "$EXP_DIR"
if [ ! -f "${EXP_DIR}/exp-details.json" ]; then
    say "creating experiment at ${EXP_DIR} from configuration ${CONFIG_NAME}"
    cp "${TEMPLATE_DIR}/exp-details.json" "${EXP_DIR}/"
else
    say "reusing experiment at ${EXP_DIR}"
    if [ -f "${EXP_DIR}/.configuration" ] && \
       [ "$(cat "${EXP_DIR}/.configuration")" != "$CONFIG_NAME" ]; then
        warn "${EXP_DIR} was created from configuration '$(cat "${EXP_DIR}/.configuration")';"
        warn "running '${CONFIG_NAME}' against it keeps that one's pool selection."
    fi
fi
printf '%s\n' "$CONFIG_NAME" > "${EXP_DIR}/.configuration"

# --------------------------------------------------------------- install ---
# The extras follow from what the sweep will do: analysis and figures are read
# off the CSVs afterwards, so `plots` is installed by default while a compute
# node that only runs `bdcevalcli run` needs none of it.
ALL_EXTRAS="$(printf 'plots,%s' "$EXTRAS" | tr ',' '\n' | sed '/^$/d' | sort -u | paste -sd, -)"

if [ "$SKIP_INSTALL" = "yes" ]; then
    say "skipping installation (--skip-install)"
    [ ${#PACKAGE_ARGS[@]} -gt 0 ] && warn "--skip-install also skips the extra packages: ${PACKAGE_ARGS[*]}"
    [ -x "${VENV_DIR}/bin/bdcevalcli" ] || warn "no bdcevalcli in ${VENV_DIR}; generation will fail"
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
    say "installing the library and the harness with extras: ${ALL_EXTRAS}"
    python -m pip install --quiet -e "${REPO_DIR}[${ALL_EXTRAS}]"
    if [ ${#PACKAGE_ARGS[@]} -gt 0 ]; then
        say "installing extra packages: ${PACKAGE_ARGS[*]}"
        python -m pip install --quiet "${PACKAGE_ARGS[@]}"
    fi
    deactivate
fi

CLI="${VENV_DIR}/bin/bdcevalcli"
VENV_PYTHON="${VENV_DIR}/bin/python"
[ -x "$CLI" ] || die "bdcevalcli was not installed into ${VENV_DIR}"

# ------------------------------------------------------------- benchmarks --
if [ "$SKIP_FETCH" != "yes" ]; then
    clone_or_update "$CLASSICAL_REPO" "$TASKS_DIR"
    if [ ! -d "$PLANS_DIR" ]; then
        [ -f "$ZIP_FILE" ] || die "no plan pools: neither ${PLANS_DIR} nor ${ZIP_FILE}"
        say "unpacking the FI plan pools"
        command -v unzip >/dev/null 2>&1 || die "unzip is needed to unpack ${ZIP_FILE}"
        unzip -q "$ZIP_FILE" -d "$(dirname "$PLANS_DIR")"
    else
        say "reusing the unpacked pools at ${PLANS_DIR}"
    fi
fi
[ -d "$TASKS_DIR" ] || die "no classical-domains checkout at ${TASKS_DIR}"
[ -d "$PLANS_DIR" ] || die "no unpacked plan pools at ${PLANS_DIR}"

# ------------------------------------------------------- experiment limits --
say "writing the limits into ${EXP_DIR}/exp-details.json"
BDCEVAL_TIME="$TIME_LIMIT" BDCEVAL_MEM="$MEMORY_LIMIT" BDCEVAL_PARTITION="$PARTITION" \
BDCEVAL_ACCOUNT="$ACCOUNT" BDCEVAL_QOS="$QOS" BDCEVAL_PARALLEL="$MAX_PARALLEL" \
"$VENV_PYTHON" - "$EXP_DIR" <<'PY'
import json, os, sys

exp_dir = sys.argv[1]
path = os.path.join(exp_dir, 'exp-details.json')
with open(path) as handle:
    details = json.load(handle)

cfgs = details.setdefault('cfgs', {})
cfgs['timelimit'] = os.environ['BDCEVAL_TIME']
cfgs['memorylimit'] = os.environ['BDCEVAL_MEM']
slurm = cfgs.setdefault('slurm', {})
slurm['partition'] = os.environ['BDCEVAL_PARTITION'] or None
slurm['account'] = os.environ['BDCEVAL_ACCOUNT'] or None
slurm['qos'] = os.environ['BDCEVAL_QOS'] or None
slurm['max-parallel-jobs'] = int(os.environ['BDCEVAL_PARALLEL'] or 0)
details['name'] = os.path.basename(os.path.abspath(exp_dir))

with open(path, 'w') as handle:
    json.dump(details, handle, indent=4)
    handle.write('\n')
print(f'  time={cfgs["timelimit"]} memory={cfgs["memorylimit"]}')
PY

# --------------------------------------------------------------- generate --
GENERATE_ARGS=(
    generate
    --exp-dir "$EXP_DIR"
    --sandbox-dir "$SANDBOX_DIR"
    --plans-dir "$PLANS_DIR"
    --ru-info-dir "${SCRIPT_DIR}/data/ru-info-dir"
    --classical-domains "$TASKS_DIR"
    --venv-dir "$VENV_DIR"
    --local-jobs "$LOCAL_JOBS"
)
[ "$SKIP_EXISTING" = "yes" ] && GENERATE_ARGS+=(--skip-existing)

echo
say "generating run commands"
"$CLI" "${GENERATE_ARGS[@]}"

cat <<EOF

Next steps
  submit the sweep      bash ${SANDBOX_DIR}/slurm/submit_all.sh
  or run it locally     bash ${SANDBOX_DIR}/run_local.sh ${LOCAL_JOBS}
  watch it              squeue -u \$USER
  summary tables        ${CLI} analyze --sandbox-dir ${SANDBOX_DIR}
  paper figures         ${CLI} report  --sandbox-dir ${SANDBOX_DIR}

  re-generate after changing the experiment (skips finished pools):
    ${CLI} generate --exp-dir ${EXP_DIR} --sandbox-dir ${SANDBOX_DIR} \\
        --plans-dir ${PLANS_DIR} --venv-dir ${VENV_DIR} --skip-existing

  Experiment A measures wall-clock: run it with run_local.sh 1, or on a node
  that is not sharing its CPUs, or its numbers are of the machine's load.
EOF
