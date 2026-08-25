"""Experiment configuration: resource limits, pool selection, experiment knobs.

An experiment directory holds one file,

    <exp-dir>/exp-details.json      limits + pool selection + per-experiment knobs

which is what ``exp-configurations/<name>/`` is a template of. The templates are
copied into a working experiment by ``setup_benchmark.sh``, whose limits it then
rewrites, so the configurations stay as they were committed.

There are exactly three experiments and they are named ``a``, ``b`` and ``c``;
each can be switched off, but none can be added. That is deliberate -- the
paper's empirical section is these three and a fourth would have nothing to
report.
"""

import copy
import json
import os

#: The three experiments, in the order they are generated and reported.
EXPERIMENTS = ['a', 'b', 'c']

DEFAULT_EXP_DETAILS = {
    'name': 'default',
    'cfgs': {
        'timelimit': '01:30:00',
        'memorylimit': '8GB',
        # Head-room the scheduler gets on top of the runner's own limits, so a
        # pool that overruns is recorded as such rather than vanishing into a
        # slurm cancellation with no CSV behind it.
        'slurm-time-headroom': '00:05:00',
        'slurm-memory-headroom': '1GB',
        'slurm': {
            'cpus-per-task': 1,
            'partition': None,
            'account': None,
            'qos': None,
            'max-parallel-jobs': 50,
            'max-array-size': 1000,
            'extra-directives': [],
        },
    },
    'pools': {
        # Which FI pools to sweep. Experiment A wants the k = 1000 pools; the
        # other two take one pool per task and the largest is the best one.
        'requested-k': [1000],
        'include-domains': [],
        'exclude-domains': [],
        # Cheap pre-filters, read off the behaviour-count ForbidIterative
        # already recorded in each pool file rather than by simulating.
        'min-behaviour-count': 0,
        'min-plans': 1,
        'require-resources': False,
        # 0 means every pool. 'even' spreads a cap across domains rather than
        # taking the first N alphabetically, which would be a sample of the
        # earliest domains rather than of the benchmark set.
        'max-pools-per-domain': 0,
        'max-pools': 0,
        'selection': 'even',
    },
    'experiments': {
        'seed': 2026,
        'k-nn': 3,
        'multiset-stability': True,
        'a': {
            'enabled': True, 'k': 20, 'repeats': 5,
            'pool-sizes': [25, 50, 100, 200, 500, 1000],
            'k-sweep': [5, 10, 20, 50],
        },
        'b': {'enabled': True, 'k': 20, 'hidden-draws': 50},
        'c': {'enabled': True, 'k': 20, 'step': 0.05, 'require-varying-ru': False},
    },
}


class Experiment:
    """One experiment directory's exp-details.json."""

    def __init__(self, details, path=None):
        self.details = details
        self.path = path

    # -- loading -------------------------------------------------------

    @classmethod
    def default(cls, name='default'):
        details = copy.deepcopy(DEFAULT_EXP_DETAILS)
        details['name'] = name
        return cls(details)

    @classmethod
    def load(cls, exp_dir):
        path = details_path(exp_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'no exp-details.json in {exp_dir}; '
                f'run `bdcevalcli init --exp-dir {exp_dir}` first')
        with open(path) as handle:
            stored = json.load(handle)
        details = _merge(copy.deepcopy(DEFAULT_EXP_DETAILS), stored)
        return cls(details, path)

    def save(self, exp_dir):
        path = details_path(exp_dir)
        os.makedirs(exp_dir, exist_ok=True)
        with open(path, 'w') as handle:
            json.dump(self.details, handle, indent=4)
            handle.write('\n')
        self.path = path
        return path

    # -- reading -------------------------------------------------------

    @property
    def name(self):
        return self.details.get('name', 'default')

    @property
    def cfgs(self):
        return self.details['cfgs']

    @property
    def slurm(self):
        return self.cfgs['slurm']

    @property
    def pools(self):
        return self.details['pools']

    @property
    def knobs(self):
        return self.details['experiments']

    def enabled_experiments(self):
        return [name for name in EXPERIMENTS
                if self.knobs.get(name, {}).get('enabled', False)]

    def experiment(self, name):
        if name not in EXPERIMENTS:
            raise ValueError(f"unknown experiment '{name}'; there are exactly "
                             f'{EXPERIMENTS} and a fourth is out of scope')
        return self.knobs.get(name, {})

    def time_limit_seconds(self):
        return parse_duration(self.cfgs.get('timelimit', '01:30:00'))

    def memory_limit_mb(self):
        return parse_memory(self.cfgs.get('memorylimit', '8GB'))

    def slurm_time(self):
        return format_duration(self.time_limit_seconds()
                               + parse_duration(self.cfgs.get('slurm-time-headroom', '0')))

    def slurm_memory(self):
        total = self.memory_limit_mb() + parse_memory(
            self.cfgs.get('slurm-memory-headroom', '0'))
        return f'{int(total)}M'

    def summary(self):
        enabled = self.enabled_experiments()
        return (f"{self.name}: experiments {'+'.join(enabled) or 'none'}, "
                f"k = {self.experiment('b').get('k', 20)}, "
                f"pools at requested-k {self.pools.get('requested-k')}, "
                f"{self.cfgs['timelimit']} / {self.cfgs['memorylimit']} per pool")


def details_path(exp_dir):
    return os.path.join(os.path.abspath(os.path.expanduser(exp_dir)), 'exp-details.json')


def _merge(base, override):
    """Recursive dict merge, so a configuration may name only what it changes."""
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _merge(base[key], value)
        else:
            base[key] = value
    return base


def parse_duration(text):
    """Seconds from '01:30:00', '90m', '5400' or '1h30m'."""
    text = str(text or '0').strip().lower()
    if not text:
        return 0
    if ':' in text:
        parts = [float(part or 0) for part in text.split(':')]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        hours, minutes, seconds = parts[-3:]
        return int(hours * 3600 + minutes * 60 + seconds)
    total, number = 0, ''
    for character in text:
        if character.isdigit() or character == '.':
            number += character
        elif character in 'hms':
            total += float(number or 0) * {'h': 3600, 'm': 60, 's': 1}[character]
            number = ''
    if number:
        total += float(number)
    return int(total)


def format_duration(seconds):
    seconds = int(seconds)
    return f'{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}'


def parse_memory(text):
    """Megabytes from '8GB', '8G', '8192M' or '8192'."""
    text = str(text or '0').strip().upper().replace('IB', 'B')
    multiplier = 1
    for suffix, factor in (('GB', 1024), ('G', 1024), ('MB', 1), ('M', 1),
                           ('KB', 1 / 1024), ('K', 1 / 1024)):
        if text.endswith(suffix):
            multiplier, text = factor, text[:-len(suffix)]
            break
    try:
        return int(float(text or 0) * multiplier)
    except ValueError:
        return 0
