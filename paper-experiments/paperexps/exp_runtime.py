
import json
import os, time
from collections import defaultdict
from utils import construct_task
from behaviour_diversity_counter.behaviour_diversity_counter import BehaviourDiversityCounter

def runtime_experiment(tasks, basedir, k_values):
    for t in tasks:
        _results = defaultdict(lambda: defaultdict(dict))
        _task, _plans, _info = construct_task(t)

        behaviour_counter = BehaviourDiversityCounter(_task, [('go', {}), ('ru', {'file': _info['resources-file']})])

        # warmup behaviour inference
        behaviour_counter.behaviours(_plans)

        print(f"| Running runtime experiment for task: {_info['year']} - {_info['domain']} - {_info['inst']}")

        for k in k_values:
            if k > len(_plans): continue
            print(f"| -- | Running runtime experiment for k={k} ...")
            for indicator in ['bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty']:
                print(f"| -- | -- | Using indicator: {indicator} ...")
                _starttime = time.time()
                _extracted_plans = behaviour_counter.extract(_plans, k, indicator=indicator)
                _endtime = time.time()
                _results[k][indicator]['time']           = _endtime - _starttime
                _results[k][indicator]['num_plans']      = len(_extracted_plans)
                _results[k][indicator]['num_behaviours'] = len(set([p.behaviour for p in _extracted_plans]))
                _results[k][indicator]['behaviours']     = [p.behaviour for p in _extracted_plans]
                _results[k][indicator]['plans']          = [p.plan_str  for p in _extracted_plans]

        _dumpfile = os.path.join(basedir, _info['dumpfile-name'])
        with open(_dumpfile, 'w') as f:
            json.dump(_results, f, indent=4)