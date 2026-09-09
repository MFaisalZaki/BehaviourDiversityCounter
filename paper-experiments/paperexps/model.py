"""The feature-based diversity model an experiment scores a pool under.

The brief's model per domain is subgoal ordering + cost bin, plus agents used
where the domain declares an agent type (the ru-info declarations), with equal
weights.  The rovers case-study model is the paper's running example: number of
rovers used (discrete distance) + subgoal ordering, weights 1/2 each.
"""

from harness import make_counter


def default_dimensions(pool, params, *, bin_width=None, with_resources=True):
    """``[(key, addinfo), ...]`` of the brief's per-domain model."""
    dimensions = [
        ('go', {'max-goals': params.get('max-goals')}),
        ('cbin', {'optimal-cost': pool.optimal_cost or 1, 'q': pool.info['q'],
                  'width': params.get('cost-bin-width') if bin_width is None else bin_width}),
    ]
    if with_resources and pool.info['resources-file']:
        dimensions.append(('ru', {'file': pool.info['resources-file']}))
    return dimensions


def rovers_dimensions(pool, params):
    """The paper's running example, for the case study."""
    if not pool.info['resources-file']:
        raise ValueError('the rovers model needs the instance\'s (:resource ...) declarations')
    return [('rn', {'file': pool.info['resources-file']}),
            ('go', {'max-goals': params.get('max-goals')})]


def with_weights(dimensions, weights):
    """The same dimensions with weights declared; ``None`` leaves the uniform
    default.  ``weights`` is a list aligned with the dimensions."""
    if weights is None:
        return dimensions
    if len(weights) != len(dimensions):
        raise ValueError(f'{len(weights)} weights for {len(dimensions)} dimensions')
    return [(key, {**addinfo, 'weight': weight}) for (key, addinfo), weight in zip(dimensions, weights)]


def build(pool, dimensions):
    """A counter over the pool and a record of the model it carries."""
    counter = make_counter(pool, dimensions)
    record = {
        'features': [key for key, _ in dimensions],
        'weights': {name: dimension.weight for name, dimension in counter.dimensions.items()},
    }
    go = counter.dimensions.get('go')
    if go is not None:
        record['goal-atoms'] = go.goal_atoms
        record['max-goals'] = go.max_goals
        record['goal-atoms-used'] = len(go.vars)
    cbin = counter.dimensions.get('cbin')
    if cbin is not None:
        record['cost-bins'] = cbin.bins
        record['cost-bin-width'] = float(cbin.width)
        record['q'] = float(cbin.q)
        record['optimal-cost'] = str(cbin.optimal_cost)
    for key in ('ru', 'rn'):
        if key in counter.dimensions:
            record['agents'] = sorted(counter.dimensions[key].addinfo['objects'])
            record['resources-file'] = pool.info['resources-file']
    return counter, record
