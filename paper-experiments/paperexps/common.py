"""Matching FI result files to their PDDL tasks, plans and resource files.

The pool files live in ``data/fi-generated-plans-dir`` and are named

    {q}-{k}-{track}-{year}-{domain}-{inst}-fi-bc-results.json
    e.g. 1.0-100-classical-2006-rovers-15-fi-bc-results.json

where ``domain`` is the api.py ``name`` field of a classical-domains domain,
``year`` its ``ipc`` field ('None' for non-IPC domains) and ``inst`` the
1-based position of the problem in the domain's problem list sorted by file
name -- the convention of catalog-run-2's ``utilities.parse_planning_tasks``.

Matching does not rely on that filename alone: each non-timeout pool carries an
``info`` block whose ``domain`` is the path of the domain file relative to the
classical-domains ``classical/`` directory (e.g. ``rovers-02/domain.pddl``) and
whose ``problem`` is the problem file's basename. The filename supplies what
``info`` lacks: the ipc year and the sorted-position instance number, which
together key the resource declarations in ``data/ru-info-dir`` --
``ru-info-dir/{domain}/{domain}-{year}.json`` holding ``instances[inst]`` as a
``(:resource ...)`` string.
"""

import json
import os
import re
import sys


def _ensure_libs_importable():
    """Make behaviour_diversity_counter and plandiversity importable.

    Installed packages win; otherwise fall back to the sibling checkouts this
    repository is developed against (BehaviourDiversityCounter is this repo's
    root, DiverseScore its sibling).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    bdc_root = os.path.dirname(os.path.dirname(here))  # BehaviourDiversityCounter/
    candidates = [
        bdc_root,
        os.path.join(os.path.dirname(bdc_root), 'DiverseScore'),
    ]
    for root in candidates:
        if os.path.isdir(root) and root not in sys.path:
            sys.path.append(root)


_ensure_libs_importable()

PAPER_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PLANS_DIR = os.path.join(PAPER_EXPERIMENTS_DIR, 'data', 'fi-generated-plans-dir')
DEFAULT_RU_INFO_DIR = os.path.join(PAPER_EXPERIMENTS_DIR, 'data', 'ru-info-dir')
DEFAULT_CLASSICAL_DOMAINS = os.path.join(PAPER_EXPERIMENTS_DIR, 'data', 'classical-domains')

POOL_FILENAME_RE = re.compile(
    r'^(?P<q>[\d.]+)-(?P<k>\d+)-(?P<track>[a-z]+)-(?P<year>[A-Za-z0-9]+)-'
    r'(?P<domain>.+)-(?P<inst>\d+)-fi-bc-results\.json$'
)


def parse_pool_filename(name):
    """The q/k/track/year/domain/inst fields of a pool file name, or None."""
    match = POOL_FILENAME_RE.match(os.path.basename(name))
    if not match:
        return None
    fields = match.groupdict()
    fields['k'] = int(fields['k'])
    fields['q'] = float(fields['q'])
    return fields


def classical_dir(classical_domains_root):
    """The directory holding one folder per domain.

    Accepts either the classical-domains repository root (which contains a
    ``classical/`` directory) or that directory itself.
    """
    root = os.path.abspath(os.path.expanduser(classical_domains_root))
    nested = os.path.join(root, 'classical')
    return nested if os.path.isdir(nested) else root


def task_files(pool_info, classical_domains_root):
    """(domain_file, problem_file) for a pool's ``info`` block."""
    base = classical_dir(classical_domains_root)
    domain_file = os.path.join(base, pool_info['domain'])
    problem_file = os.path.join(base, os.path.dirname(pool_info['domain']),
                                pool_info['problem'])
    return domain_file, problem_file


def _resource_index(ru_info_dir, _cache={}):
    """Every ru-info file's instances, keyed by the (domain, year) it declares.

    Indexed by the ``info`` block *inside* each JSON, not by file name --
    catalog-run-2's convention (``_get_resources_details``). Most files happen
    to be named ``{domain}-{year}.json``, but not all (``agricola-opt18.json``
    declares domain agricola, year 2018), so the file name is never trusted.
    """
    root = os.path.abspath(ru_info_dir)
    if root not in _cache:
        index = {}
        for folder, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(folder, name)) as handle:
                        declared = json.load(handle)
                except (json.JSONDecodeError, OSError):
                    continue
                info = declared.get('info') or {}
                key = (str(info.get('domain')), str(info.get('year')))
                index[key] = declared.get('instances') or {}
        _cache[root] = index
    return _cache[root]


def load_resources(ru_info_dir, domain, year, inst):
    """The ``(:resource ...)`` declaration string for one instance, or None."""
    if not os.path.isdir(ru_info_dir):
        return None
    instances = _resource_index(ru_info_dir).get((str(domain), str(year)), {})
    return instances.get(str(inst))


def load_pool(path):
    with open(path) as handle:
        return json.load(handle)


def pool_has_plans(path, head_bytes=4096):
    """Cheap test that a pool file holds plans (timeout stubs do not).

    Pool files are dumped with ``plans`` as the first key, so reading the head
    is enough; the fallback full parse covers files written differently.
    """
    size = os.path.getsize(path)
    if size < 200:
        return False
    with open(path, 'rb') as handle:
        head = handle.read(head_bytes)
    if b'"plans"' in head:
        return True
    try:
        return bool(load_pool(path).get('plans'))
    except (json.JSONDecodeError, OSError):
        return False


def parse_task(domain_file, problem_file):
    """Parse a PDDL task, tolerating names the benchmarks reuse across kinds.

    unified-planning refuses by default to give two elements of a problem the
    same name, which is stricter than PDDL itself: floortile declares both a
    predicate and an action ``up`` (also ``down``, ``left``, ``right``) and
    tidybot declares an object ``cart`` of type ``cart``. Both are standard IPC
    benchmarks, and rejecting them costs 95 pools of the sweep, so the check is
    disabled -- the exception message itself names this flag as the remedy.
    """
    from unified_planning.io import PDDLReader
    import unified_planning.shortcuts as ups
    environment = ups.get_environment()
    environment.credits_stream = None
    environment.error_used_name = False
    return PDDLReader().parse_problem(domain_file, problem_file)


def _fd_name_map(names):
    """original names keyed by their Fast Downward normalisation.

    Fast Downward lowercases names and turns ``-`` into ``_``; the FI pools
    store plans with those normalised names, while the PDDL keeps the
    originals (e.g. plan ``jump_new_move pos_3_2`` for domain action
    ``jump-new-move`` and object ``pos-3-2``). Colliding normalisations are
    dropped so an ambiguous token can never be silently mispaired.
    """
    mapping, ambiguous = {}, set()
    for name in names:
        key = name.lower().replace('-', '_')
        if key in mapping and mapping[key] != name:
            ambiguous.add(key)
        mapping[key] = name
    return {key: name for key, name in mapping.items() if key not in ambiguous}


FD_SUFFIX_RE = re.compile(r'_\d+$')


def _resolve_name(name, known):
    """The task's name for a plan token, or the token unchanged.

    Beyond lowercasing and ``-``/``_``, Fast Downward appends ``_<n>`` to a
    name for two unrelated reasons, and the pools store the suffixed names:

    - to disambiguate a name that collides with another kind's -- floortile's
      action ``up`` clashes with its predicate ``up`` and becomes ``up_0``,
      tidybot's object ``cart`` clashes with its type ``cart`` and becomes
      ``cart_0``;
    - to name the pieces of an action it *split* because the precondition was
      disjunctive (trucks' ``load`` becomes ``load_1``, ``load_2``, ...;
      likewise openstacks and pathways).

    Both collapse back to the one PDDL name safely. The split is sound by
    construction -- Fast Downward splits precondition ``A or B`` into variants
    requiring ``A`` and requiring ``B``, over the same parameters and effects,
    so anything a variant permits the original permits too. The suffix is only
    stripped when the direct lookup misses *and* the stripped name is itself
    known, so a name genuinely ending in ``_1`` (``pos_3_2``, ``load_1`` as a
    real action) always wins, and a typo is left alone to fail loudly.
    """
    if name in known:
        return known[name]
    return known.get(FD_SUFFIX_RE.sub('', name), name)


def _denormalise_plan_text(task, text, _cache={}):
    """Rewrite a Fast Downward-normalised plan back to the task's own names."""
    if id(task) not in _cache:
        _cache.clear()  # one task per process; keep the previous maps from leaking
        _cache[id(task)] = (
            _fd_name_map(action.name for action in task.actions),
            _fd_name_map(obj.name for obj in task.all_objects),
        )
    actions, objects = _cache[id(task)]
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith('('):
            lines.append(line)
            continue
        tokens = stripped.strip('()').split()
        rewritten = [_resolve_name(tokens[0], actions)] + \
                    [_resolve_name(token, objects) for token in tokens[1:]]
        lines.append('(' + ' '.join(rewritten) + ')')
    return '\n'.join(lines)


def parse_plans(task, plan_strings):
    """Parse plan strings against the task; failures are recorded, not fatal.

    A plan that fails to parse verbatim is retried with Fast Downward's name
    normalisation undone (see :func:`_fd_name_map`).
    """
    from unified_planning.io import PDDLReader
    reader = PDDLReader()
    plans, failures = [], []
    for index, text in enumerate(plan_strings):
        try:
            plans.append(reader.parse_plan_string(task, text))
        except Exception:
            try:
                plans.append(reader.parse_plan_string(task, _denormalise_plan_text(task, text)))
            except Exception as error:  # a malformed plan must not kill the pool
                failures.append({'index': index, 'error': f'{type(error).__name__}: {error}'})
    return plans, failures


def build_counter(task, resources_string, workdir):
    """A BehaviourDiversityCounter over the paper's behaviour space.

    Dimensions are the goal-predicate ordering ('go') plus, when the instance
    declares resources, the set of resources used ('ru') -- the two dimensions
    of the paper's Section 7 setup. 'ru' rather than 'rc' because B-MaxSum
    needs a per-dimension distance, which only 'go', 'cb' and 'ru' implement.
    """
    from behaviour_diversity_counter import BehaviourDiversityCounter
    dimensions = [('go', None)]
    if resources_string:
        os.makedirs(workdir, exist_ok=True)
        resource_file = os.path.join(workdir, 'resources.txt')
        with open(resource_file, 'w') as handle:
            handle.write(resources_string)
        dimensions.append(('ru', resource_file))
    return BehaviourDiversityCounter(task, dimensions), [name for name, _ in dimensions]
