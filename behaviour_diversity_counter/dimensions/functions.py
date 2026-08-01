
import os

from collections import defaultdict
from lark import Lark, Transformer, v_args
from behaviour_diversity_counter.dimensions.base import BehaviourDimension

class NumericFunctionDimension(BehaviourDimension):
    def __init__(self, task, addinfo):
        super().__init__(task, 'function_value', parse_functions_file(addinfo))
    
    def _estimate_domain(self):
        self.estimated_domain_size = 1
        for func_name, func_info in self.addinfo.items():
            self.estimated_domain_size *= len(set(range(func_info['min'], func_info['max'] - func_info['delta'], func_info['delta'])))
    
    def plan_behaviour(self, plan):
        vars_values_over_time = defaultdict(list)
        for t, state in enumerate(plan.states):
            var_map = {str(e).replace('(','_').replace(')','').replace(' ','_').replace(',','') : e for e in state._values}
            for func_name, func_info in self.addinfo.items():
                if not func_info['name'] in var_map: continue
                vars_values_over_time[func_info['name']].append(state.get_value(var_map[func_info['name']]))
        
        # map the values.
        for _, fn in self.addinfo.items():
            varname, minval, maxval, delta = fn['name'], fn['min'], fn['max'], fn['delta']
            boxes = [(idx, i, i+delta) for idx, i in enumerate(range(minval, maxval-delta, delta))]
            if len(vars_values_over_time[varname]) == 0: continue
            current_value = vars_values_over_time[varname][-1].constant_value()
            vars_values_over_time[varname] = next(filter(lambda e: current_value >= e[1] and current_value < e[2], boxes), boxes[-1])[0]
        
        val = ','.join([f'{k}:{str(v)}' for k,v in vars_values_over_time.items()])
        self.domain.add(val)
        return val

class ResourceTransformer(Transformer):
    def resource_line(self, token):
        # Grammar order is NAME MIN MAX DELTA.
        return {
            'name':  token[0].value,
            'min':   int(token[1].value),
            'max':   int(token[2].value),
            'delta': int(token[3].value)
        }

def parse_functions_file(inputfile):
    def read_function_file(resource_input):
        def construct_parser():
            grammar = r'''
                start: resource_line+
                resource_line: "(:function" (NAME | NAME_WITH_PARENTHESIS) MIN MAX DELTA ")"
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
        assert os.path.exists(inputfile), f'The function file {inputfile} does not exist.'
        for resource in read_function_file(inputfile):
            addition_informaion[resource['name']] = resource
    return addition_informaion
