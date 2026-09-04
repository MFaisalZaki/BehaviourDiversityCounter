from behaviour_diversity_counter.dimensions.base import BehaviourDimension, declaration_source
from behaviour_diversity_counter.dimensions.declaration_file import parse_declaration_file


class ResourceCountDimension(BehaviourDimension):
    def __init__(self, task, addinfo):
        super().__init__(task, 'rc',
                         {'resources_list': parse_declaration_file(declaration_source(addinfo), 'resource')},
                         addinfo.get('weight', 1.0))
        self.addinfo['objects'] = set(map(str,filter(lambda e: e.name in set(map(lambda e: e['name'], self.addinfo['resources_list'].values())), self.task.all_objects)))

    def plan_behaviour(self, plan):
        resource_usage = {o: 0 for o in self.addinfo['objects']}
        for action in plan.actions:
            for used_resource in set.intersection(set(map(str, action.actual_parameters)), set(self.addinfo['objects'])):
                resource_usage[used_resource] += 1
        # One prefixed token, comma-separated: ' $$ ' separates *dimensions*, so it
        # cannot also separate counts within this one. Sorted because addinfo['objects']
        # is a set, whose iteration order varies between processes.
        counts = ','.join(f'{o}={resource_usage[o]}' for o in sorted(resource_usage))
        return f'{self.name}:' + counts


class ResourceUsedDimension(BehaviourDimension):
    def __init__(self, task, addinfo):
        super().__init__(task, 'ru',
                         {'resources_list': parse_declaration_file(addinfo.get('file', None), 'resource')},
                         addinfo.get('weight', 1.0))
        self.addinfo['objects'] = set(map(str,filter(lambda e: e.name in set(map(lambda e: e['name'], self.addinfo['resources_list'].values())), self.task.all_objects)))

    def plan_behaviour(self, plan):
        resource_usage = {o: 0 for o in self.addinfo['objects']}
        for action in plan.actions:
            for used_resource in set.intersection(set(map(str, action.actual_parameters)), set(self.addinfo['objects'])):
                resource_usage[used_resource] += 1
        used = frozenset(o for o, c in resource_usage.items() if c > 0)
        # encode the *set* of used resources so distance() can compute Jaccard.
        return f'{self.name}:' + ','.join(sorted(used))

    def _used_set(self, plan):
        # Match on the token prefix, not a substring: the goal-ordering token
        # may contain 'ru' inside predicate/object names (e.g. 'truck1').
        token = next(filter(lambda e: e.strip().startswith(self.name + ':'), plan.split('$$')), None)
        assert token is not None, 'The dimension value should be present in the plan behaviour.'
        payload = token.strip().replace(self.name + ':', '').strip()
        return set(filter(None, payload.split(',')))

    def distance(self, plan1, plan2):
        s1, s2 = self._used_set(plan1), self._used_set(plan2)
        if not s1 and not s2:
            return 0.0
        return self.weight * (1.0 - len(s1 & s2) / len(s1 | s2))
