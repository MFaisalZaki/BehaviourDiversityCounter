"""Matching ForbidIterative pool files to the PDDL tasks they were planned for.

Pool files in the plans directory are named

    {q}-{k}-{track}-{year}-{domain}-{inst}-fi-bc-results.json
    e.g. 1.0-100-classical-2006-rovers-15-fi-bc-results.json

where ``domain`` is the ``name`` field of a classical-domains ``api.py`` entry,
``year`` its ``ipc`` field ('None' for the non-IPC domains) and ``inst`` the
1-based position of the problem in that entry's problem list sorted by file
name.

(domain, year) does not name a directory.  1998/logistics lives in
``logistics98`` and 2002/rovers in ``rovers-02``; where an IPC year ran both an
optimal and a satisficing track the pools sometimes come from one and sometimes
from the other -- 2011/barman is ``barman-opt11-strips`` but 2011/floortile is
``floortile-sat11-strips`` -- and those two editions hold *different* instances.
The directory is therefore resolved from the ``api.py`` index, and where that
leaves more than one candidate from the ``info`` block that every non-timeout
pool file carries (``info.domain`` is the domain file's path relative to the
``classical/`` root).  Timed-out pools are 48-byte files with no ``info``, so a
group made up entirely of timeouts can stay genuinely ambiguous; those are
reported rather than guessed at.

34 domain directories ship one domain file per problem (``airport``'s
``p01-domain.pddl``, ``p02-domain.pddl``, ...), so the domain file is taken from
the same ``api.py`` pair as the problem rather than assumed to be ``domain.pddl``.
"""

import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

RESULTS_SUFFIX = '-fi-bc-results.json'

# Fast Downward suffixes a name with ``_<n>`` where it disambiguated a clash
# (floortile's action ``up`` against its predicate ``up``) or split an action on
# a disjunctive precondition (trucks' ``load`` into ``load_1``, ``load_2``).
# Both collapse back to the one PDDL name safely.
FD_SUFFIX_RE = re.compile(r'_\d+$')

POOL_RE = re.compile(
    r'^(?P<q>[\d.]+)-(?P<k>\d+)-(?P<track>[a-z]+)-(?P<year>[A-Za-z0-9]+)-'
    r'(?P<domain>.+)-(?P<inst>\d+)' + re.escape(RESULTS_SUFFIX) + r'$'
)


def create_dump_dir(dump_dir):
    dump_dir_path = os.path.join(HERE, dump_dir)
    os.makedirs(dump_dir_path, exist_ok=True)
    return dump_dir_path


def _domain_index(root):
    """``(domain, year) -> {directory: [(domain_path, problem_path), ...]}``.

    Paths are relative to ``root``.  Each directory's list is its ``api.py``
    entries in file order, the problems within an entry sorted by path -- the
    ordering ``inst`` counts through.  ``api.py`` is parsed rather than imported
    so that reading the benchmark never executes it.
    """
    index = {}
    for entry in sorted(os.listdir(root)):
        api_path = os.path.join(root, entry, 'api.py')
        if not os.path.isfile(api_path):
            continue
        with open(api_path, encoding='utf-8') as handle:
            tree = ast.parse(handle.read(), filename=api_path)
        for node in tree.body:
            if not (isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == 'domains'
                    for target in node.targets)):
                continue
            for domain in ast.literal_eval(node.value):
                key = (domain.get('name'), str(domain.get('ipc')))
                problems = sorted((tuple(pair) for pair in domain.get('problems') or ()),
                                  key=lambda pair: pair[1])
                index.setdefault(key, {}).setdefault(entry, []).extend(problems)
    return index


def match_plans_with_problems(plans_dir, problems_dir, ru_info):
    """Every pool file under ``plans_dir`` paired with its domain and problem.

    ``problems_dir`` is the classical-domains checkout, either its root or its
    ``classical/`` directory.  ``ru_info`` is the ru-info tree, whose
    ``instances[inst]`` holds a task's ``(:resource ...)`` declarations.  Relative
    paths are taken against this package, matching :func:`create_dump_dir`.

    Returns a list of dicts sorted by pool file name::

        {'task_id': '1.0-10-classical-1998-gripper-1',
         'pool_file': '/.../1.0-10-classical-1998-gripper-1-fi-bc-results.json',
         'domain': 'gripper', 'year': '1998', 'inst': 1,
         'q': 1.0, 'k': 10, 'track': 'classical',
         'domain_dir': 'gripper',
         'domain_file': '/.../classical/gripper/domain.pddl',
         'problem_file': '/.../classical/gripper/prob01.pddl',
         'resolved_by': 'only api.py candidate',
         'resources': '(:resource left 100 0 5)\n(:resource right 100 0 5)'}

    ``resources`` is None where ru-info declares none: it covers 35 of the 69
    (domain, year) pairs the pools span.  Pools whose domain directory cannot be
    pinned down are dropped, with a summary on stderr.
    """
    ru_files_dir = os.path.join(HERE, ru_info)
    os.makedirs(ru_files_dir, exist_ok=True)

    plans_root = os.path.join(HERE, plans_dir)
    if not os.path.isdir(plans_root):
        raise FileNotFoundError(f'no plans directory at {plans_root}')

    problems_root = os.path.join(HERE, problems_dir)
    if os.path.isdir(os.path.join(problems_root, 'classical')):
        problems_root = os.path.join(problems_root, 'classical')
    if not os.path.isdir(problems_root):
        raise FileNotFoundError(f'no benchmark directory at {problems_root}')
    index = _domain_index(problems_root)

    ru_root = os.path.join(HERE, ru_info)
    if not os.path.isdir(ru_root):
        raise FileNotFoundError(f'no ru-info directory at {ru_root}')
    resources = {}                    # (domain, year) -> {instance: declarations}
    for dirpath, _, names in os.walk(ru_root):
        for name in sorted(names):
            if not name.endswith('.json'):
                continue
            with open(os.path.join(dirpath, name), encoding='utf-8') as handle:
                declared = json.load(handle)
            # Keyed by each file's own info block, not its name: agricola-opt18
            # declares agricola/2018.
            info = declared.get('info') or {}
            resources[(info.get('domain'), str(info.get('year')))] = (
                declared.get('instances') or {})

    groups, unparsed = {}, []
    for name in sorted(os.listdir(plans_root)):
        match = POOL_RE.match(name)
        if match is None:
            if name.endswith('.json'):
                unparsed.append(name)
            continue
        pool = match.groupdict()
        pool['q'] = float(pool['q'])
        pool['k'] = int(pool['k'])
        pool['inst'] = int(pool['inst'])
        pool['path'] = os.path.join(plans_root, name)
        groups.setdefault((pool['domain'], pool['year']), []).append(pool)

    tasks, unresolved = [], []
    for key, pools in sorted(groups.items()):
        candidates = index.get(key, {})
        directory = how = None
        if not candidates:
            how = 'no api.py entry for this (domain, year)'
        elif len(candidates) == 1:
            directory, how = next(iter(candidates)), 'only api.py candidate'
        else:
            for pool in pools:                      # ask the pools themselves
                with open(pool['path'], encoding='utf-8') as handle:
                    info = json.load(handle).get('info') or {}
                named = str(info.get('domain', '')).split('/')[0]
                if named in candidates:
                    directory, how = named, 'info block'
                    break
            else:
                # Every pool in the group timed out.  An edition too short to
                # hold the instances asked for cannot be the one they came from.
                largest = max(pool['inst'] for pool in pools)
                fits = [name for name, problems in candidates.items()
                        if largest <= len(problems)]
                if len(fits) == 1:
                    directory, how = fits[0], 'instance count'
                else:
                    how = 'ambiguous: ' + ', '.join(sorted(fits or candidates))
        if directory is None:
            unresolved.append((key, len(pools), how))
            continue

        problems = candidates[directory]
        for pool in pools:
            inst = pool['inst']
            if not 1 <= inst <= len(problems):
                unresolved.append((key, 1, f'instance {inst} outside {directory} '
                                           f'({len(problems)} problems)'))
                continue
            domain_path, problem_path = problems[inst - 1]

            # dump ru-info to file.
            ru_file = os.path.join(ru_files_dir, f"{os.path.basename(pool['path'])[:-len(RESULTS_SUFFIX)]}.txt")
            with open(ru_file, 'w') as f:
                _details = resources.get(key, {}).get(str(inst))
                if _details is None: continue
                for line in _details.splitlines():
                    f.write(line + '\n')

            tasks.append({
                'task_id': os.path.basename(pool['path'])[:-len(RESULTS_SUFFIX)],
                'pool_file': pool['path'],
                'domain': pool['domain'],
                'year': pool['year'],
                'inst': inst,
                'q': pool['q'],
                'k': pool['k'],
                'track': pool['track'],
                'domain_dir': directory,
                'domain_file': os.path.join(problems_root, domain_path),
                'problem_file': os.path.join(problems_root, problem_path),
                'resolved_by': how,
                'resources': ru_file,
            })

    tasks.sort(key=lambda task: os.path.basename(task['pool_file']))
    if unresolved or unparsed:
        dropped = sum(count for _, count, _ in unresolved) + len(unparsed)
        print(f'match_plans_with_problems: {dropped} of {len(tasks) + dropped} '
              f'pool files unmatched', file=sys.stderr)
        for (domain, year), count, why in sorted(unresolved):
            print(f'  {year}/{domain:<24} {count:>4} files  {why}', file=sys.stderr)
        if unparsed:
            print(f'  unparsable file names    {len(unparsed):>4} files  '
                  f'e.g. {unparsed[0]}', file=sys.stderr)
    return tasks

def construct_task(taskdetails):
    """``(task, plans, info)`` for one matched task: the parsed PDDL problem,
    its pool's plans, and the pool's own metadata.

    ``info['parse-failures']`` lists any plan that could not be parsed; a
    malformed plan is recorded rather than allowed to kill the whole pool.
    """
    from unified_planning.io import PDDLReader
    import unified_planning.shortcuts as ups

    # unified-planning refuses by default to give two elements of a problem the
    # same name, which is stricter than PDDL itself: floortile declares both a
    # predicate and an action ``up``, tidybot an object ``cart`` of type ``cart``.
    # Both are standard IPC benchmarks, so the check is disabled -- the exception
    # message itself names this flag as the remedy.
    environment = ups.get_environment()
    environment.credits_stream = None
    environment.error_used_name = False

    reader = PDDLReader()
    up_task = reader.parse_problem(taskdetails['domain_file'], taskdetails['problem_file'])

    with open(taskdetails['pool_file'], 'r') as f:
        pool_data = json.load(f)

    def fd_names(names):
        # Original names keyed by their Fast Downward normalisation.  Colliding
        # normalisations are dropped, so an ambiguous token is never mispaired.
        mapping, ambiguous = {}, set()
        for name in names:
            key = name.lower().replace('-', '_')
            if key in mapping and mapping[key] != name:
                ambiguous.add(key)
            mapping[key] = name
        return {key: name for key, name in mapping.items() if key not in ambiguous}

    actions = fd_names(action.name for action in up_task.actions)
    objects = fd_names(obj.name for obj in up_task.all_objects)

    def resolve(token, known):
        # The suffix is only stripped when the direct lookup misses and the
        # stripped name is itself known, so a name genuinely ending in ``_1``
        # wins and a typo is left alone to fail loudly.
        if token in known:
            return known[token]
        return known.get(FD_SUFFIX_RE.sub('', token), token)

    def denormalise(text):
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith('('):
                lines.append(line)            # ``;cost``/``;behaviour`` comments
                continue
            tokens = stripped.strip('()').split()
            lines.append('(' + ' '.join([resolve(tokens[0], actions)]
                                        + [resolve(t, objects) for t in tokens[1:]]) + ')')
        return '\n'.join(lines)

    # The pools store plans under Fast Downward's normalised names (lowercased,
    # ``-`` -> ``_``) while the PDDL keeps the originals -- logistics plans say
    # ``load_truck`` for the domain's ``LOAD-TRUCK``.  Parse verbatim first and
    # rewrite only on failure, so a task whose names already match is untouched.
    up_plans, failures = [], []
    for index, text in enumerate(pool_data.get('plans') or []):
        try:
            up_plans.append(reader.parse_plan_string(up_task, text))
            setattr(up_plans[-1], 'plan_str', text)
        except Exception:
            try:
                up_plans.append(reader.parse_plan_string(up_task, denormalise(text)))
                setattr(up_plans[-1], 'plan_str', text)
            except Exception as error:
                failures.append({'index': index, 'error': f'{type(error).__name__}: {error}'})

    info = {
        'planning-time': pool_data.get('total-time-seconds', None),
        'resources-file': taskdetails['resources'],
        'domain': taskdetails['domain'],
        'year': taskdetails['year'],
        'inst': taskdetails['inst'],
        'q': taskdetails['q'],
        'k': taskdetails['k'],
        'track': taskdetails['track'],
        'parse-failures': failures,
        'dumpfile-name': f"{taskdetails['track']}-{taskdetails['year']}-{taskdetails['domain']}-{taskdetails['inst']}-{taskdetails['q']}-{taskdetails['k']}.json"
    }

    return up_task, up_plans, info
