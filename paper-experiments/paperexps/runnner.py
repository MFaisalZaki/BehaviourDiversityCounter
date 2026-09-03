
from exp_runtime import runtime_experiment

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Run experiments on paper-experiments.")
    parser.add_argument('--config-file', type=str, help='Path to the configuration file.')
    parser.add_argument('--experiment-name', type=str, help='Name of the experiment to run.')
    args = parser.parse_args()

    with open(args.config_file, 'r') as f:
        config = json.load(f)

    exp_parameters = next(filter(lambda e: e['name'] == args.experiment_name, config['experiments']), None)
    assert exp_parameters is not None, f"Experiment '{args.experiment_name}' not found in the configuration file."

    match exp_parameters['name']:
        case 'runtime':
            runtime_experiment(exp_parameters)

    pass

    