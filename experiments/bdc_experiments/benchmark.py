"""The classical-domains checkout, and the instances taken out of it.

Each domain directory ships an ``api.py`` naming its problems. It is read with
``ast`` and never imported: the file is benchmark data, and importing it would
run whatever it happens to contain.
"""

import ast
import re
import subprocess

from bdc_experiments.config import results_root


def benchmark_dir(cfg):
    return results_root(cfg) / 'benchmark'


def ensure(cfg):
    """The checkout at the pinned commit, cloned on first use."""
    root = benchmark_dir(cfg)
    if not (root / '.git').is_dir():
        root.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none',
                        cfg['benchmark']['source'], str(root)], check=True)
    head = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                          check=True, capture_output=True, text=True).stdout.strip()
    if head != cfg['benchmark']['commit']:
        subprocess.run(['git', '-C', str(root), 'checkout', '--quiet',
                        cfg['benchmark']['commit']], check=True)
    return root


def checkout_revision(cfg):
    """The commit the checkout actually stands at, for the manifest."""
    root = benchmark_dir(cfg)
    if not (root / '.git').is_dir():
        return None
    return subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                          capture_output=True, text=True).stdout.strip() or None


def read_api(api_path):
    """The ``domains = [...]`` literal of a domain directory's api.py."""
    tree = ast.parse(api_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'domains':
            return ast.literal_eval(node.value)
    raise ValueError(f'{api_path}: no `domains = [...]` assignment')


def _natural(stem):
    """Sort key putting p2 before p10: the instance number, then the text."""
    digits = re.findall(r'\d+', stem)
    return (int(digits[0]) if digits else 0, stem)


def instances(cfg, root=None):
    """Every instance of every configured domain, in instance-number order.

    The id is ``<domain>/<ipc-year>/<problem-file-stem>``: the file name, not a
    position in a list, so it survives a benchmark update. ``inst`` is the
    1-based position of the problem in its api.py entry with the problems
    sorted by path, which is how the pool archive and the ru-info tree number
    the instances (``pfile1, pfile10, pfile11, ...``).
    """
    root = root if root is not None else benchmark_dir(cfg)
    found = []
    for domain in cfg['benchmark']['domains']:
        directory = root / 'classical' / domain
        api = directory / 'api.py'
        if not api.is_file():
            raise FileNotFoundError(f'no api.py for domain {domain} at {api}')
        for entry in read_api(api):
            ipc = str(entry.get('ipc', 'unknown'))
            problems = sorted((tuple(pair) for pair in entry['problems']), key=lambda pair: pair[1])
            for inst, (domain_rel, problem_rel) in enumerate(problems, 1):
                problem = root / 'classical' / problem_rel
                found.append({
                    'id': f'{domain}/{ipc}/{problem.stem}',
                    'domain': domain,
                    'ipc': ipc,
                    'name': entry.get('name', domain),
                    'inst': inst,
                    'stem': problem.stem,
                    'domain_file': str(root / 'classical' / domain_rel),
                    'problem_file': str(problem),
                })
    found.sort(key=lambda i: (i['domain'], _natural(i['stem'])))
    return found

