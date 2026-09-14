"""The diversity models, as data.

The paper's thesis is that features are user-defined, so the evaluation uses
domain-specific models written by the authors acting as the domain expert, one
generic model on every domain as a control, and the literature's model stated
as one feature, the stability distance over action sets, which turns the
formulation into the post-hoc selection of Katz and Sohrabi (2020). E2 varies
the astronaut's weights and E3 the number of features; each variant is a spec
of its own, with its own hash and its own behaviour dump.

The agents of an instance are not guessed from the PDDL: they are the
``(:resource ...)`` declarations the authors wrote per domain, year and
instance under ``experiments/data/ru-info-dir``, which phase one copies into
the pool record and ``build_counter`` hands to the ru/rn/rc dimensions.
"""

import hashlib
import json
import math
from dataclasses import dataclass

from behaviour_diversity_counter import BehaviourDiversityCounter


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
#: driverlog the goals also fix where one truck and one driver end up.
#: The agents (`rn`, `ru`) are the instance's declared resources: rovers in
#: rovers, trucks in logistics and driverlog, satellites in satellite.
PER_DOMAIN = (
    ModelSpec('rovers_astronaut', ('rovers',),
              (FeatureSpec('rn', {}, 0.5), FeatureSpec('go', {}, 0.5))),
    ModelSpec('logistics_dispatcher', ('logistics98', 'logistics00'),
              (FeatureSpec('ru', {}, 0.5), FeatureSpec('go', {}, 0.5))),
    ModelSpec('driverlog_dispatcher', ('driverlog',),
              (FeatureSpec('ru', {}, 0.5), FeatureSpec('go', {}, 0.5))),
    ModelSpec('satellite_operator', ('satellite',),
              (FeatureSpec('ru', {}, 0.5), FeatureSpec('go', {}, 0.5))),
)

#: The dimensions that read a ``(:resource ...)`` declaration file.
RESOURCE_KEYS = ('ru', 'rn', 'rc')

#: The literature's model as one feature: the plan's action set, compared by
#: the stability distance.
STABILITY = ModelSpec('stability', None, (FeatureSpec('stability', {}),))


def generic_spec(cfg, features=None, name='generic'):
    """The control model; both its knobs are fixed in the config."""
    settings = cfg['models']['generic']
    knobs = {'go': {'max-goals': settings['goal_cap']}, 'cbin': {'width': settings['cost_bin_width']}}
    return ModelSpec(name, None, tuple(FeatureSpec(key, knobs.get(key, {}))
                                       for key in (features or settings['features'])))


def domain_model(cfg, domain):
    """The feature-based model of a domain: its own where one exists, else
    the generic control. E2's F."""
    return next((spec for spec in PER_DOMAIN if domain in spec.domains), generic_spec(cfg))


def weight_variants(cfg):
    """The astronaut's model under each further weight setting of E2."""
    base = PER_DOMAIN[0]
    return [ModelSpec(f"{base.name}-w{'-'.join(f'{w:g}' for w in weights)}", base.domains,
                      tuple(FeatureSpec(f.key, f.params, w) for f, w in zip(base.features, weights)))
            for weights in cfg['e2']['weight_settings']
            if list(weights) != [f.weight for f in base.features]]


def timing_specs(cfg, domain):
    """E3's models by feature count: goal order alone, the generic pair, and,
    where the domain has a model of its own, that model plus the cost bin."""
    own = next((spec for spec in PER_DOMAIN if domain in spec.domains), None)
    specs = [generic_spec(cfg, features=['go'], name='generic-n1'), generic_spec(cfg)]
    if own:
        specs.append(ModelSpec(f'{own.name}-cbin', own.domains,
                               tuple(FeatureSpec(f.key, f.params) for f in own.features)
                               + (FeatureSpec('cbin', {'width': cfg['models']['generic']['cost_bin_width']}),)))
    return specs


def selection_specs(cfg, domain):
    """Every model the selection sweep runs on a pool of this domain."""
    return ([generic_spec(cfg), STABILITY] + [s for s in PER_DOMAIN if domain in s.domains]
            + [s for s in weight_variants(cfg) if domain in s.domains])


def registry(cfg):
    """``name -> ModelSpec`` for every model any task can name."""
    specs = [generic_spec(cfg), STABILITY, *PER_DOMAIN, *weight_variants(cfg)]
    specs += [s for d in cfg['benchmark']['domains'] for s in timing_specs(cfg, d)]
    return {spec.name: spec for spec in specs}


def is_primary(name):
    """The models the setup describes: the generic control and the per-domain
    ones. E2 reads these alone; the variants belong to E2-C and E3."""
    return name == 'generic' or name in {spec.name for spec in PER_DOMAIN}


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


def write_resource_file(path, declarations):
    """The instance's ``(:resource ...)`` declarations, verbatim, where the
    ru/rn/rc dimensions read them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(declarations.strip() + '\n')
    return path


def build_counter(spec, task, instance_info):
    """The library counter for a spec over one instance.

    ``instance_info`` carries what the spec deliberately leaves out: the
    instance's optimal cost and quality bound (for the cost bin), its resource
    declarations, and the directory those are written into.
    """
    dimensions = []
    for feature in spec.features:
        addinfo = dict(feature.params)
        if feature.weight is not None:
            addinfo['weight'] = feature.weight
        if feature.key in RESOURCE_KEYS:
            if not instance_info.get('resources'):
                raise ValueError(f"model {spec.name}: instance {instance_info.get('id')} "
                                 'declares no resources (nothing in the ru-info tree for it)')
            path = instance_info['resource_dir'] / f"{instance_info['id'].replace('/', '__')}.txt"
            addinfo['file'] = str(write_resource_file(path, instance_info['resources']))
        if feature.key == 'cbin':
            if instance_info['optimal_cost'] is None:
                raise ValueError(f"model {spec.name}: instance {instance_info.get('id')} has no "
                                 f'optimal cost recorded, so the cost bin has no c* to divide by')
            addinfo['optimal-cost'] = instance_info['optimal_cost']
            addinfo['q'] = instance_info['q']
        dimensions.append((feature.key, addinfo))
    return BehaviourDiversityCounter(task, dimensions)


#: What each dimension is, how the paper counts |Delta_i|, and its dissimilarity.
#: The setup report prints this table.
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
    'stability': ("the plan's set of actions",
                  'the set of action sets over the task\'s actions',
                  'stability: 1 - Jaccard over the two action sets (Srivastava et al. 2007)'),
}


def dimension_size(counter, key):
    """|Delta_i| for one built dimension, or None where it is unbounded."""
    dimension = counter.dimensions[key]
    if key == 'go':
        return math.factorial(len(dimension.vars))
    if key == 'cbin':
        return dimension.bins
    if key == 'ru':
        return 2 ** len(dimension.addinfo['objects'])
    if key == 'rn':
        return len(dimension.addinfo['objects']) + 1
    return None


def model_record(spec, counter, task, instance_info):
    """What every result file and every behaviour dump says about the model:
    the features as declared, the objects and goal atoms they resolved to on
    this instance, and the dimension sizes whose product is |BS| (Def. bspace)."""
    features = []
    for feature in spec.features:
        entry = {'key': feature.key, 'params': _sortable(feature.params),
                 'weight': feature.weight, 'size': dimension_size(counter, feature.key)}
        if feature.key in RESOURCE_KEYS:
            entry['objects'] = sorted(counter.dimensions[feature.key].addinfo['objects'])
        if feature.key == 'go':
            entry['goal_atoms'] = [str(atom) for atom in counter.dimensions['go'].vars]
        features.append(entry)
    sizes = [f['size'] for f in features]
    return {'name': spec.name, 'hash': model_hash(spec),
            'domains': list(spec.domains) if spec.domains else None,
            'features': features,
            'space_size': None if None in sizes else math.prod(sizes),
            'weight_convention': ('declared' if any(f.weight is not None for f in spec.features)
                                  else 'uniform 1/n'),
            'instance': instance_info.get('id')}
