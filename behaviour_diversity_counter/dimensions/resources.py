from behaviour_diversity_counter.dimensions.base import (
    BehaviourDimension, declaration_source, declared_weight)
from behaviour_diversity_counter.dimensions.declaration_file import parse_declaration_file


def _declared_resources(task, addinfo):
    declared = parse_declaration_file(declaration_source(addinfo), 'resource')
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
    """``rc``: how many declared resource appears in the plan."""

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'rc', _declared_resources(task, addinfo), declared_weight(addinfo))

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
        if counts1 == counts2 == 0: return 0.0
        return self.weight * (1.0 - (abs(counts1-counts2)/(counts1+counts2)))

class ResourceUsedDimension(BehaviourDimension):
    """``ru``: the set of declared resources the plan uses at all."""

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'ru', _declared_resources(task, addinfo), declared_weight(addinfo))

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
        return self.weight * (1.0 - len(s1 & s2) / len(s1 | s2))


class ResourceNumberDimension(BehaviourDimension):
    """``rn``: how many of the declared resources the plan uses at all, with
    the discrete distance -- 0 when the numbers agree and 1 otherwise. This is
    the rover-usage feature of the paper's running example.
    """

    def __init__(self, task, addinfo=None):
        super().__init__(task, 'rn', _declared_resources(task, addinfo), declared_weight(addinfo))

    def extract(self, plan):
        usage = _usage(self.addinfo['objects'], plan)
        number = sum(1 for count in usage.values() if count > 0)
        self.domain.add(number)
        return f'{self.name}:{number}'

    def dissimilarity(self, b1, b2):
        return self.weight * (0.0 if self.payload(b1) == self.payload(b2) else 1.0)
