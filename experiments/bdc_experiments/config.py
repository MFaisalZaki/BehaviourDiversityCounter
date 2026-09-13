"""The one TOML file that drives the sweep.

Loaded into a plain dict, checked against the hard-coded key set below and
hashed, so that a report can name the exact configuration it was produced from.
No dataclasses and no schema library: the file is small and the keys are fixed.
"""

import hashlib
import tomllib
from pathlib import Path

#: The configuration, section by section: key -> (type, required).
#: Anything not listed here is rejected, so a typo is an error rather than a
#: silently ignored setting.
KEYS = {
    'run': {
        'seed': (int, True),
        'results_dir': (str, True),
        'time_limit_generation_s': (int, True),
        'memory_limit_generation_mb': (int, True),
        'time_limit_selection_s': (int, True),
    },
    'benchmark': {
        'source': (str, True),
        'commit': (str, True),
        'domains': (list, True),
        'instances_per_domain': (int, True),
    },
    'generation': {
        'planner': (str, True),
        'modes': (list, True),
        'pool_sizes': (list, True),
        'q_values': (list, True),
    },
    'selection': {
        'k_values': (list, True),
        'kappa_values': (list, True),
    },
    'models': {
        'generic': (dict, True),
        'enabled': (list, False),
    },
    'e1': {'domain': (str, True), 'q': (float, True), 'min_behaviours': (int, True),
           'k': (int, True), 'kappa': (int, True)},
    'e2': {'subsets': (int, True)},
    'e3': {'k_values': (list, True), 'behaviour_range': (list, True)},
    'e4': {'k_range': (list, True)},
    'e5': {'goal_caps': (list, True), 'cost_bin_widths': (list, True), 'k': (int, True)},
    'e6': {'pool_sizes': (list, True), 'repeats': (int, True), 'feature_counts': (list, True)},
}

#: Configs shipped with the package, so `bdcexp run smoke e3` works anywhere.
CONFIG_DIR = Path(__file__).parent / 'configs'


def resolve(path):
    """A config path as given, or the name of one of the packaged configs."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    for name in (f'{path}.toml', str(path)):
        packaged = CONFIG_DIR / name
        if packaged.is_file():
            return packaged
    raise FileNotFoundError(f'no config at {path} and none packaged under {CONFIG_DIR}')


def load(path, results_dir=None):
    """The config as a dict, validated, with a ``meta`` section added.

    ``results_dir`` overrides ``[run].results_dir``; the tests use it to send a
    run into a temporary directory without editing the file.
    """
    source = resolve(path)
    raw = source.read_bytes()
    cfg = tomllib.loads(raw.decode())
    _validate(cfg, source)
    if results_dir is not None:
        cfg['run']['results_dir'] = str(results_dir)
    cfg['meta'] = {
        'config_path': str(source.resolve()),
        'config_hash': hashlib.sha256(raw).hexdigest(),
        'run_name': Path(cfg['run']['results_dir']).name,
    }
    return cfg


def _validate(cfg, source):
    unknown_sections = sorted(set(cfg) - set(KEYS))
    if unknown_sections:
        raise ValueError(f'{source}: unknown section(s) {unknown_sections}; '
                         f'valid sections: {sorted(KEYS)}')
    for section, keys in KEYS.items():
        if section not in cfg:
            raise ValueError(f'{source}: missing section [{section}]')
        unknown = sorted(set(cfg[section]) - set(keys))
        if unknown:
            raise ValueError(f'{source}: unknown key(s) {unknown} in [{section}]; '
                             f'valid keys: {sorted(keys)}')
        for key, (kind, required) in keys.items():
            if key not in cfg[section]:
                if required:
                    raise ValueError(f'{source}: missing key {key} in [{section}]')
                continue
            value = cfg[section][key]
            # A whole number written without a point is an int to tomllib; the
            # config means it as the float it stands for.
            if kind is float and isinstance(value, int) and not isinstance(value, bool):
                cfg[section][key] = value = float(value)
            if not isinstance(value, kind) or isinstance(value, bool) != (kind is bool):
                raise ValueError(f'{source}: [{section}].{key} should be {kind.__name__}, '
                                 f'got {type(value).__name__}')


def results_root(cfg):
    """Everything the sweep writes goes under here."""
    return Path(cfg['run']['results_dir']).expanduser()


def snapshot(cfg):
    """Copy the config as run into the run directory.

    The run directory is the artefact that ships with the paper, so it carries
    the configuration it was produced from; the hash in every manifest ties the
    two together.
    """
    root = results_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'config.toml'
    path.write_bytes(Path(cfg['meta']['config_path']).read_bytes())
    return path
