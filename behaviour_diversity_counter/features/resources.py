import os

from collections import defaultdict
from itertools import chain, combinations
from lark import Lark, Transformer, v_args
from behaviour_diversity_counter.features.base import DimensionConstructorSimulator


class ResourceCountSimulator(DimensionConstructorSimulator):
    def __init__(self, task, addinfo):
        super().__init__(task, 'rc', {'resources_list': parse_resource_file(addinfo)})
        self.addinfo['objects'] = set(map(str,filter(lambda e: e.name in set(map(lambda e: e['name'], self.addinfo['resources_list'].values())), self.task.all_objects)))
    
    def _estimate_domain(self):
        # the maximum resource count is the number of objects that can be used as resources.
        self.estimated_domain_size = len(set(chain.from_iterable(combinations(self.addinfo['objects'], r) for r in range(1, len(self.addinfo['objects']) + 1))))

    def plan_behaviour(self, plan):
        resource_usage = {o: 0 for o in self.addinfo['objects']}
        for action in plan.actions:
            for used_resource in set.intersection(set(map(str, action.actual_parameters)), set(self.addinfo['objects'])):
                resource_usage[used_resource] += 1
        used = frozenset(o for o, c in resource_usage.items() if c > 0)
        self.domain.add(used)
        # encode the *set* of used resources so distance() can compute Jaccard.
        return f'{self.name}:' + ','.join(sorted(used))

    def _used_set(self, plan):
        token = next(filter(lambda e: self.name in e, plan.split('$$')), None)
        assert token is not None, 'The dimension value should be present in the plan behaviour.'
        payload = token.strip().replace(self.name + ':', '').strip()
        return set(filter(None, payload.split(',')))

    def distance(self, plan1, plan2):
        s1, s2 = self._used_set(plan1), self._used_set(plan2)
        if not s1 and not s2:
            return 0.0
        return 1.0 - len(s1 & s2) / len(s1 | s2)

class ResourceUsedSimulator(DimensionConstructorSimulator):
    def __init__(self, task, addinfo):
        super().__init__(task, 'ru', {'resources_list': parse_resource_file(addinfo)})
        self.addinfo['objects'] = set(map(str,filter(lambda e: e.name in set(map(lambda e: e['name'], self.addinfo['resources_list'].values())), self.task.all_objects)))
    
    def _estimate_domain(self):
        # the maximum resource count is the number of objects that can be used as resources.
        self.estimated_domain_size = len(set(chain.from_iterable(combinations(self.addinfo['objects'], r) for r in range(1, len(self.addinfo['objects']) + 1))))

    def plan_behaviour(self, plan):
        resource_usage = {o: 0 for o in self.addinfo['objects']}
        for action in plan.actions:
            for used_resource in set.intersection(set(map(str, action.actual_parameters)), set(self.addinfo['objects'])):
                resource_usage[used_resource] += 1
        return ' $$ '.join(map(lambda e: f'{e[0]}={e[1]}', resource_usage.items()))
    
    def distance(self, plan1, plan2):
        assert False, 'Distance function is not implemented for ResourceUsedSimulator.'

class ResourceTransformer(Transformer):
    def resource_line(self, token):
        return {
            'name': token[0].value,
            'min':  int(token[1].value),
            'max':  int(token[2].value),
            'delta': int(token[3].value)
        }

def parse_resource_file(inputfile):
    def read_resource_file(resource_input):
        def construct_parser():
            grammar = r'''
                start: resource_line+
                resource_line: "(:resource" (NAME | NAME_WITH_PARENTHESIS) MIN MAX DELTA ")"
                NAME: /[a-zA-Z_][\w-]*/
                NAME_WITH_PARENTHESIS: /[a-zA-Z_]\w*\([^)]*\)/
                MIN: /[0-9]+/
                MAX: /[0-9]+/
                DELTA: /[0-9]+/
                %ignore /\s+/
            '''
            parser = Lark(grammar, parser='lalr', transformer=v_args(inline=True))
            return parser
        # readlines in reource_input
        with open(resource_input, 'r') as f:
            resource_input = f.readlines()
        resource_input = ''.join(resource_input)
        parser = construct_parser()
        tree = parser.parse(resource_input)
        transformer = ResourceTransformer()
        resources = transformer.transform(tree)
        return resources.children

    addition_informaion = defaultdict(dict)
    if inputfile:
        assert os.path.exists(inputfile), f'The resources file {inputfile} does not exist.'
        for resource in read_resource_file(inputfile):
            addition_informaion[resource['name']] = resource
    return addition_informaion
