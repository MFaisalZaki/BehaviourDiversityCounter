
from exp_runtime import runtime_experiment
from exp_weights import weights_experiment

from utils import create_dump_dir, dumpfile_name, match_plans_with_problems

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run experiments on paper-experiments.")
    parser.add_argument('--config-file', type=str, help='Path to the configuration file.')
    parser.add_argument('--experiment-name', type=str, help='Name of the experiment to run.')
    parser.add_argument('--list-tasks', action='store_true',
                        help='Print "task_id<TAB>dumpfile" for every task and exit. '
                             'This is the manifest a slurm array indexes into.')
    parser.add_argument('--task-id', type=str, default=None,
                        help='Run only this task. One array element, one task.')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    exp_parameters = next(filter(lambda e: e['name'] == args.experiment_name, config['experiments']), None)
    assert exp_parameters is not None, f"Experiment '{args.experiment_name}' not found in the configuration file."

    tasks   = match_plans_with_problems(exp_parameters['plansdir'], exp_parameters['benchmark'], exp_parameters['ru-info'])

    if args.list_tasks:
        for t in tasks:
            print(f"{t['task_id']}\t{dumpfile_name(t)}")
        raise SystemExit(0)

    if args.task_id is not None:
        tasks = [t for t in tasks if t['task_id'] == args.task_id]
        assert tasks, f"no task with id '{args.task_id}' in this configuration"

    basedir = create_dump_dir(exp_parameters['dump-dir'])

    match exp_parameters['name']:
        case 'runtime':
            runtime_experiment(tasks, basedir, exp_parameters['k-values'])
        case 'weights':
            weights_experiment(tasks, basedir, exp_parameters['w-values'], exp_parameters['subset-k-percentage'])

    pass