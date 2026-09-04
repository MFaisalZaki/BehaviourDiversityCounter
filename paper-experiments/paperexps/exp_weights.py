import json
import os, time
from collections import defaultdict
from utils import construct_task
from behaviour_diversity_counter.behaviour_diversity_counter import BehaviourDiversityCounter

def weights_experiment(tasks, basedir, w_values, subset_k_precentage):
    # In this experiment, we claim that adding weights 
    # would change which behaviours are selected.
    # In other words it would show that some dimensions are more important than others.
    for t in tasks:
        _results = defaultdict(lambda: defaultdict(dict))
        _task, _plans, _info = construct_task(t)

        orcale_behaviour_counter = BehaviourDiversityCounter(_task, [('go', {}), ('ru', {'file': _info['resources-file']})])
        orcale_behaviour_counter.behaviours(_plans)
        print(f"| Running runtime experiment for task: {_info['year']} - {_info['domain']} - {_info['inst']}")

        for w in w_values:
            dimslist  = [[('go', {'weight': w}),   ('ru', {'file': _info['resources-file'], 'weight': 1.0})]]
            dimslist += [[('go', {'weight': 1.0}), ('ru', {'file': _info['resources-file'], 'weight': w})]]

            for dims in dimslist:
                k = max(2, int(len(_plans) * subset_k_precentage))
                behaviour_counter = BehaviourDiversityCounter(_task, dims)
                behaviour_counter.behaviours(_plans)
                
                # I want a better key to indicate which dimension is weighted, so I will use the weights in the key
                dims_key = '-'.join([f"{d[0]}-{d[1]['weight']}" for d in dims])
                print(f"| -- | Using dimensions: {dims_key}")
                
                for indicator in ['bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty']:
                    print(f"| -- | -- | Using indicator: {indicator} ...")
                    _starttime = time.time()
                    _extracted_plans = behaviour_counter.extract(_plans, k, indicator=indicator)
                    _endtime = time.time()

                    _results[dims_key][indicator]['time']                  = _endtime - _starttime
                    _results[dims_key][indicator]['num_plans']             = len(_extracted_plans)
                    _results[dims_key][indicator]['num_behaviours']        = len(set([p.behaviour for p in _extracted_plans]))
                    _results[dims_key][indicator]['plans-behaviours']      = [p.behaviour for p in _extracted_plans]
                    _results[dims_key][indicator]['plans']                 = [p.plan_str  for p in _extracted_plans]

                    # now we need to compute the domain coverage per dimension.
                    domain_coverage_conter = BehaviourDiversityCounter(_task, dims)
                    domain_coverage_conter.behaviours(_extracted_plans)

                    _results[dims_key][indicator]['dims-domains']          = {name:list(value.domain) for name, value in domain_coverage_conter.dimensions.items()}
                    _results[dims_key][indicator]['oracle-dims-domains']   = {name:list(value.domain) for name, value in orcale_behaviour_counter.dimensions.items()}

        _dumpfile = os.path.join(basedir, _info['dumpfile-name'])
        with open(_dumpfile, 'w') as f:
            json.dump(_results, f, indent=4)