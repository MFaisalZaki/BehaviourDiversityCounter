import os

from collections import defaultdict
from lark import Lark, Transformer, v_args
from behaviour_diversity_counter.features.base import DimensionConstructorSimulator


class ResourceCountSimulator(DimensionConstructorSimulator):
    def __init__(self, task, addinfo):
        super().__init__(task, 'ru', {'resources_list': parse_resource_file(addinfo)})
        self.addinfo['objects'] = set(map(str,filter(lambda e: e.name in set(map(lambda e: e['name'], self.addinfo['resources_list'].values())), self.task.all_objects)))
    
    def plan_behaviour(self, plan):
        resource_usage = {o: 0 for o in self.addinfo['objects']}
        for action in plan.actions:
            for used_resource in set.intersection(set(map(str, action.actual_parameters)), set(self.addinfo['objects'])):
                resource_usage[used_resource] += 1
        val = len(list(filter(lambda e: e[1] > 0, resource_usage.items())))
        self.domain.add(val)
        return f'{self.name}:' + str(val)

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
