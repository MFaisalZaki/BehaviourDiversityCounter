from behaviour_diversity_counter.dimensions.base import BehaviourDimension, options
from behaviour_diversity_counter.dimensions.declarations import parse_declarations


def _declared_resources(task, addinfo):
    """``addinfo['resources']`` is the list of ``(:resource ...)`` declaration strings."""
    declared = parse_declarations(options(addinfo).get('resources'), 'resource')
    names = {resource['name'] for resource in declared.values()}
    objects = {str(obj) for obj in task.all_objects if obj.name in names}
    return {'resources_list': declared, 'objects': objects}


def _usage(objects, plan):
    usage = {name: 0 for name in objects}
    for action in plan.actions:
        for used in set(map(str, action.actual_parameters)) & set(objects):
            usage[used] += 1
    return usage


def _pairs(payload):
    """``name=value`` items out of a comma-separated token payload."""
    return dict(item.split('=') for item in payload.split(',') if item)


class ResourceCountDimension(BehaviourDimension):
    """``rc``: how many of the declared resources the plan uses at all."""

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'rc', _declared_resources(task, addinfo))

    def extract(self, plan):
        usage = _usage(self.addinfo['objects'], plan)
        # counts = ','.join(f'{name}={usage[name]}' for name in sorted(usage))
        counts = sum(1 if v > 0 else 0 for v in usage.values())
        self.domain.add(counts)
        return f'{self.name}:{counts}'

    def _counts(self, behaviour):
        return int(self.payload(behaviour))

    def dissimilarity(self, b1, b2):
        counts1, counts2 = self._counts(b1), self._counts(b2)
        # |c - c'| / (c + c'): 0 exactly on equal counts, 1 when one plan uses
        # no declared resource and the other some, in [0, 1] throughout.
        if counts1 == counts2 == 0: return 0.0
        return abs(counts1 - counts2) / (counts1 + counts2)

class ResourceUsedDimension(BehaviourDimension):
    """``ru``: the set of declared resources the plan uses at all."""

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'ru', _declared_resources(task, addinfo))

    def extract(self, plan):
        usage = _usage(self.addinfo['objects'], plan)
        used = ','.join(sorted(name for name, count in usage.items() if count > 0))
        self.domain.add(used)
        return f'{self.name}:' + used

    def _used_set(self, behaviour):
        return set(filter(None, self.payload(behaviour).split(',')))

    def dissimilarity(self, b1, b2):
        # Jaccard distance over the two used sets: a metric in [0, 1].
        s1, s2 = self._used_set(b1), self._used_set(b2)
        if not s1 and not s2:
            return 0.0
        return 1.0 - len(s1 & s2) / len(s1 | s2)
