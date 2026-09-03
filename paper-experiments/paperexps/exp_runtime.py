
import json
import os, tempfile, time
from collections import defaultdict
from utils import create_dump_dir, match_plans_with_problems, construct_task
from behaviour_diversity_counter.behaviour_diversity_counter import BehaviourDiversityCounter

def runtime_experiment(parameters):
    basedir = create_dump_dir(parameters['dump-dir'])
    tasks = match_plans_with_problems(parameters['plansdir'], parameters['benchmark'], parameters['ru-info'])

    for t in tasks:
        _results = defaultdict(lambda: defaultdict(dict))
        _task, _plans, _info = construct_task(t)

        behaviour_counter = BehaviourDiversityCounter(_task, [('go', None), ('ru', _info['resources-file'])])

        # warmup behaviour inference
        behaviour_counter.behaviours(_plans)

        for k in parameters['k-values']:
            if k > len(_plans): continue
            for indicator in ['bcoverage', 'bmaxsum', 'bmaxmin', 'bnovelty']:
                _starttime = time.time()
                _extracted_plans = behaviour_counter.extract(_plans, k, indicator=indicator)
                _endtime = time.time()
                _results[k][indicator]['time']           = _endtime - _starttime
                _results[k][indicator]['num_plans']      = len(_extracted_plans)
                _results[k][indicator]['num_behaviours'] = len(set([p.behaviour for p in _extracted_plans]))
                _results[k][indicator]['behaviours']     = [p.behaviour for p in _extracted_plans]

        _dumpfile = os.path.join(basedir, _info['dumpfile-name'])
        with open(_dumpfile, 'w') as f:
            json.dump(_results, f, indent=4)