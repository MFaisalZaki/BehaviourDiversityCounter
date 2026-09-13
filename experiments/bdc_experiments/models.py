"""The diversity models, as data.

The paper's thesis is that features are user-defined, so the evaluation uses
domain-specific models written by the authors acting as the domain expert, plus
one generic model on every domain as a control. A model is a ``ModelSpec``; a
counter is built from one against a particular task.

The resource objects of an instance are read from the PDDL problem by type
name. Two encodings occur in the benchmark and both are handled: rovers
declares a `rover` type, while the STRIPS encodings of logistics, driverlog and
satellite carry the type as a unary predicate true in the initial state.
"""

import hashlib
import json
from dataclasses import dataclass

from behaviour_diversity_counter import BehaviourDiversityCounter

#: The friendly feature names the config may use for the generic model.
ALIASES = {'goal_order': 'go', 'cost_bin': 'cbin', 'resources_used': 'ru',
           'resource_number': 'rn', 'resource_count': 'rc'}


@dataclass(frozen=True)
class FeatureSpec:
    key: str            # a key of the library's dimensions_map
    params: dict        # the knobs of this feature, and nothing instance-specific
    weight: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    domains: tuple | None       # None means every domain
    features: tuple


#: The per-domain models. `go` is the order in which the problem's goal atoms
#: are first achieved: in rovers every goal atom is a `communicated_*` atom, in
#: logistics and satellite every goal atom is a delivery or an image, and in
#: driverlog the goals also fix where one truck and one driver end up, so there
#: the feature reads "goal order" a little more widely than "delivery order".
PER_DOMAIN = (
    ModelSpec('rovers_astronaut', ('rovers',),
              (FeatureSpec('rn', {'types': ('rover',)}, 0.5), FeatureSpec('go', {}, 0.5))),
    ModelSpec('rovers_fine', ('rovers',),
              (FeatureSpec('ru', {'types': ('rover',)}, 0.5), FeatureSpec('go', {}, 0.5))),
    ModelSpec('logistics_dispatcher', ('logistics98', 'logistics00'),
              (FeatureSpec('ru', {'types': ('truck', 'airplane')}, 0.5), FeatureSpec('go', {}, 0.5))),
    ModelSpec('driverlog_dispatcher', ('driverlog',),
              (FeatureSpec('ru', {'types': ('driver',)}, 0.5), FeatureSpec('go', {}, 0.5))),
    ModelSpec('satellite_operator', ('satellite',),
              (FeatureSpec('ru', {'types': ('satellite',)}, 0.5), FeatureSpec('go', {}, 0.5))),
)


def generic_spec(cfg, goal_cap=None, cost_bin_width=None, features=None, name=None):
    """The control model, and the two resolution knobs E5 varies."""
    settings = cfg['models']['generic']
    cap = settings['goal_cap'] if goal_cap is None else goal_cap
    width = settings['cost_bin_width'] if cost_bin_width is None else cost_bin_width
    keys = [ALIASES.get(f, f) for f in (features if features is not None else settings['features'])]
    knobs = {'go': {'max-goals': cap}, 'cbin': {'width': width}}
    return ModelSpec(name or 'generic', None,
                     tuple(FeatureSpec(key, knobs.get(key, {})) for key in keys))


def registry(cfg):
    """``name -> ModelSpec`` for everything the config enables."""
    models = {'generic': generic_spec(cfg)}
    models.update({spec.name: spec for spec in PER_DOMAIN})
    enabled = cfg['models'].get('enabled')
    return {name: spec for name, spec in models.items() if enabled is None or name in enabled}


def models_for(cfg, domain):
    """The models that apply to a domain, generic first."""
    return [spec for spec in registry(cfg).values()
            if spec.domains is None or domain in spec.domains]


def model_hash(spec):
    """A stable digest of the spec as declared, and so of the behaviour space.

    Instance-specific values (the optimal cost, the quality bound, the resource
    file) are not part of it: they vary with the pool, which the dump path
    already separates.
    """
    canonical = json.dumps({
        'name': spec.name, 'domains': list(spec.domains) if spec.domains else None,
        'features': [{'key': f.key, 'params': _sortable(f.params), 'weight': f.weight}
                     for f in spec.features],
    }, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def _sortable(params):
    return {key: (list(value) if isinstance(value, tuple) else value)
            for key, value in sorted(params.items())}


def resource_objects(task, type_names):
    """The instance's objects of the named types, by user type or, where the
    encoding is untyped STRIPS, by the unary predicate of that name."""
    declared = {user_type.name for user_type in task.user_types}
    chosen = set()
    for name in type_names:
        if name in declared:
            chosen |= {obj.name for obj in task.all_objects if obj.type.name == name}
            continue
        for fluent, value in task.initial_values.items():
            if (value.is_true() and len(fluent.args) == 1
                    and fluent.fluent().name == name and fluent.args[0].is_object_exp()):
                chosen.add(fluent.args[0].object().name)
    return sorted(chosen)


def write_resource_file(path, objects):
    """The ``(:resource ...)`` declaration the ru/rn/rc dimensions read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(f'(:resource {name} 0 1 1)\n' for name in objects))
    return path


def model_record(spec, task, instance_info):
    """What every result file and every behaviour dump says about the model."""
    features = []
    for feature in spec.features:
        entry = {'key': feature.key, 'params': _sortable(feature.params),
                 'weight': feature.weight}
        if 'types' in feature.params:
            entry['objects'] = resource_objects(task, feature.params['types'])
        if feature.key == 'go':
            entry['goal_atoms'] = _goal_atoms(task)
        features.append(entry)
    return {'name': spec.name, 'hash': model_hash(spec),
            'domains': list(spec.domains) if spec.domains else None,
            'features': features,
            'weight_convention': ('declared' if any(f.weight is not None for f in spec.features)
                                  else 'uniform 1/n'),
            'instance': instance_info.get('id')}


def _goal_atoms(task):
    from unified_planning.model.walkers.free_vars import FreeVarsExtractor
    atoms = []
    for goal in task.goals:
        atoms.extend(sorted(FreeVarsExtractor().get(goal), key=lambda atom: atom.node_id))
    return [str(atom) for atom in atoms]


def build_counter(spec, task, instance_info, trace_cache=None):
    """The library counter for a spec over one instance.

    ``instance_info`` carries what the spec deliberately leaves out: the
    instance's optimal cost and quality bound (for the cost bin) and the
    directory the resource declarations are written into.
    """
    dimensions = []
    for feature in spec.features:
        addinfo = dict(feature.params)
        if feature.weight is not None:
            addinfo['weight'] = feature.weight
        types = addinfo.pop('types', None)
        if types is not None:
            objects = resource_objects(task, types)
            if not objects:
                raise ValueError(f"model {spec.name}: instance {instance_info.get('id')} "
                                 f'declares no objects of type(s) {list(types)}')
            path = (instance_info['resource_dir']
                    / f"{instance_info['id'].replace('/', '__')}-{'_'.join(types)}.txt")
            addinfo['file'] = str(write_resource_file(path, objects))
        if feature.key == 'cbin':
            addinfo['optimal-cost'] = instance_info['optimal_cost']
            addinfo['q'] = instance_info['q']
        dimensions.append((feature.key, addinfo))
    return BehaviourDiversityCounter(task, dimensions, trace_cache=trace_cache)


#: What each dimension is, how the paper counts |Delta_i|, and its dissimilarity.
#: The setup report prints this table and E5 multiplies the sizes out.
DIMENSION_DOC = {
    'go': ('order in which the goal atoms are first achieved',
           'm! orderings of the m capped goal atoms',
           'Hamming distance over the ordering, divided by m'),
    'cbin': ('bin of cost / c* in a grid of width w over [1, q]',
             'max(1, round((q-1)/w)) bins',
             "|bin - bin'| / (bins - 1)"),
    'ru': ('the set of declared resources the plan uses at all',
           '2^|R| subsets of the R declared resources',
           'Jaccard distance over the used sets'),
    'rn': ('how many of the declared resources the plan uses',
           '|R| + 1 values',
           'discrete: 0 when equal, 1 otherwise'),
    'rc': ('how many times each declared resource appears',
           'unbounded: a count vector over the R resources',
           'weighted Jaccard (Ruzicka) over the count vectors'),
}


def dimension_size(counter, key):
    """|Delta_i| for one built dimension, or None where it is unbounded."""
    dimension = counter.dimensions[key]
    if key == 'go':
        return _factorial(len(dimension.vars))
    if key == 'cbin':
        return dimension.bins
    if key == 'ru':
        return 2 ** len(dimension.addinfo['objects'])
    if key == 'rn':
        return len(dimension.addinfo['objects']) + 1
    return None


def space_size(counter):
    """|BS| = the product of the dimension sizes, or None if any is unbounded."""
    sizes = [dimension_size(counter, key) for key in counter.dimensions]
    if any(size is None for size in sizes):
        return None
    product = 1
    for size in sizes:
        product *= size
    return product


def _factorial(n):
    result = 1
    for value in range(2, n + 1):
        result *= value
    return result
