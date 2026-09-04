
from collections import defaultdict
from behaviour_diversity_counter.dimensions.base import BehaviourDimension, declaration_source
from behaviour_diversity_counter.dimensions.declaration_file import parse_declaration_file

class NumericFunctionDimension(BehaviourDimension):
    def __init__(self, task, addinfo):
        super().__init__(task, 'function_value',
                         parse_declaration_file(declaration_source(addinfo), 'function'),
                         addinfo.get('weight', 1.0))

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
        return val
