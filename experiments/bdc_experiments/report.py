"""What every report writes: CSV first, a booktabs table second, a figure
third, and one manifest naming the code, the data and the counts behind them.

Reports are pure functions of the raw dumps. Nothing here loads a pool, builds
a counter or runs a selection.
"""

import csv
import json
import platform
import statistics
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from bdc_experiments import SCHEMA_VERSION
from bdc_experiments.config import results_root

#: Okabe-Ito, safe under the three common colour vision deficiencies.
PALETTE = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000')

#: Stated in every report: the paper leaves tie-breaking arbitrary, the
#: implementation does not.
TIE_RULE = ('Ties are broken deterministically by the lowest index in the '
            'cost-sorted pool, at the opening pair and at every later step.')

PACKAGES = ('behaviour-diversity-counter', 'unified-planning', 'numpy', 'scipy',
            'pandas', 'matplotlib', 'up-symk', 'lark')


def report_dir(cfg, experiment):
    path = results_root(cfg) / 'reports' / experiment
    (path / 'tables').mkdir(parents=True, exist_ok=True)
    (path / 'figures').mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path, rows, columns=None):
    """Rows as they are: floats unrounded, missing values empty, never zero."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = columns or sorted({key for row in rows for key in row})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ('' if row.get(key) is None else row.get(key))
                             for key in columns})
    return path


#: LaTeX's special characters, so a feature name or a dissimilarity formula can
#: be written plainly in the source and still compile.
ESCAPES = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
           '_': r'\_', '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
           '^': r'\textasciicircum{}', '|': r'\textbar{}'}


def escape(text):
    return ''.join(ESCAPES.get(character, character) for character in str(text))


def fmt(value, digits=3):
    """LaTeX cell: rounded here and only here; a missing number stays missing."""
    if value is None:
        return '--'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return escape(value)


def table(path, label, caption, columns, rows, digits=3, aligns=None):
    """A booktabs table the paper can \\input by a name that does not change."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    aligns = aligns or ('l' + 'r' * (len(columns) - 1))
    lines = [r'\begin{table}[t]', r'  \centering', rf'  \begin{{tabular}}{{{aligns}}}',
             r'    \toprule',
             '    ' + ' & '.join(escape(column) for column in columns) + r' \\',
             r'    \midrule']
    for row in rows:
        lines.append('    ' + ' & '.join(fmt(cell, digits) for cell in row) + r' \\')
    lines += [r'    \bottomrule', r'  \end{tabular}',
              rf'  \caption{{{caption}}}', rf'  \label{{{label}}}', r'\end{table}', '']
    path.write_text('\n'.join(lines))
    return path


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

def holm(pvalues):
    """Holm-Bonferroni step-down adjustment, order preserved."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def wilcoxon(x, y):
    """Two-sided signed-rank test, zero differences dropped; (None, None) when
    every difference is zero or too few pairs remain."""
    from scipy.stats import wilcoxon as scipy_wilcoxon
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and a != b]
    if len(pairs) < 1:
        return None, None
    try:
        result = scipy_wilcoxon([a for a, _ in pairs], [b for _, b in pairs],
                                zero_method='wilcox', alternative='two-sided')
    except ValueError:
        return None, None
    return float(result.statistic), float(result.pvalue)


def kendall(x, y):
    """Kendall's tau-b; (None, None) when it is undefined (a constant input)."""
    from scipy.stats import kendalltau
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(pairs) < 2:
        return None, None
    result = kendalltau([a for a, _ in pairs], [b for _, b in pairs], variant='b')
    tau = float(result.statistic) if result.statistic == result.statistic else None
    p = float(result.pvalue) if result.pvalue == result.pvalue else None
    return tau, p


def summarise(values):
    """n, median and the interquartile range: what every table reports."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return {'n': 0, 'median': None, 'q1': None, 'q3': None, 'min': None, 'max': None,
                'mean': None}
    quartiles = (statistics.quantiles(clean, n=4) if len(clean) > 1 else [clean[0]] * 3)
    return {'n': len(clean), 'median': statistics.median(clean),
            'q1': quartiles[0], 'q3': quartiles[2],
            'min': clean[0], 'max': clean[-1], 'mean': statistics.fmean(clean)}


def macro(rows, value_key, group_key='domain'):
    """Mean of the per-domain means: domains contribute unequal instance counts,
    so every table reports this next to the pooled figure."""
    groups = {}
    for row in rows:
        value = row.get(value_key)
        if value is not None:
            groups.setdefault(row.get(group_key), []).append(value)
    means = [statistics.fmean(values) for values in groups.values() if values]
    return statistics.fmean(means) if means else None


def pooled(rows, value_key):
    values = [row[value_key] for row in rows if row.get(value_key) is not None]
    return statistics.fmean(values) if values else None


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def figure(size=(5.2, 3.2)):
    """A matplotlib figure with no title text: the caption carries the words."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size)
    ax.set_prop_cycle(color=list(PALETTE))
    return fig, ax


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# The manifest
# ----------------------------------------------------------------------

def versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def manifest(cfg, experiment, outputs, results, extra=None):
    """One per experiment: the code, the data and the counts behind the tables."""
    from bdc_experiments.generate import SEARCH
    from bdc_experiments.runner import git_revision
    from bdc_experiments import benchmark
    counts = {'total': len(results),
              'failed': sum(1 for r in results if r.get('error')),
              'skipped': sum(1 for r in results if r.get('extra', {}).get('skipped')),
              }
    counts['ok'] = counts['total'] - counts['failed'] - counts['skipped']
    record = {
        'schema': 'manifest', 'version': SCHEMA_VERSION, 'experiment': experiment,
        'written': datetime.now(timezone.utc).isoformat(),
        'git': git_revision(),
        'config': {'path': cfg['meta']['config_path'], 'hash': cfg['meta']['config_hash'],
                   'run_name': cfg['meta']['run_name']},
        'benchmark': {'source': cfg['benchmark']['source'],
                      'pinned_commit': cfg['benchmark']['commit'],
                      'checkout_revision': benchmark.checkout_revision(cfg)},
        'planner': {'name': cfg['generation']['planner'], 'searches': SEARCH,
                    'time_limit_s': cfg['run']['time_limit_generation_s'],
                    'memory_limit_mb': cfg['run']['memory_limit_generation_mb']},
        'selection_time_limit_s': cfg['run']['time_limit_selection_s'],
        'seed': cfg['run']['seed'],
        'packages': versions(),
        'python': platform.python_version(),
        'platform': {'system': platform.system(), 'release': platform.release(),
                     'machine': platform.machine(), 'processor': platform.processor()},
        'tie_breaking': TIE_RULE,
        'tasks': counts,
        'failures': [{'task_id': r['task_id'], 'error': r['error']['type'],
                      'message': r['error']['message']}
                     for r in results if r.get('error')],
        'outputs': [str(path) for path in outputs],
    }
    if extra:
        record['extra'] = extra
    path = report_dir(cfg, experiment) / 'manifest.json'
    path.write_text(json.dumps(record, indent=1))
    return path



def setup_report(cfg):
    """The two files the paper's Setup subsection consumes, from the pools alone."""
    from bdc_experiments import models, pools
    rows, per_domain = [], {}
    for path in pools.pool_files(cfg):
        pool = json.loads(path.read_text())
        entry = per_domain.setdefault(pool['domain'], {
            'domain': pool['domain'], 'ipc_year': pool.get('ipc'), 'instances': set(),
            'pools': 0, 'exhausted': 0, 'timed_out': 0, 'empty': 0, 'plans': 0})
        entry['instances'].add(pool['instance'])
        entry['pools'] += 1
        entry['exhausted'] += bool(pool.get('exhausted'))
        entry['timed_out'] += bool(pool.get('timed_out'))
        entry['empty'] += not pool['plans']
        entry['plans'] += len(pool['plans'])
        key = f"pools_q{pool['q']}"
        entry[key] = entry.get(key, 0) + 1
    for entry in per_domain.values():
        entry['instances'] = len(entry['instances'])
        rows.append(entry)
    rows.sort(key=lambda row: row['domain'])
    columns = ['domain', 'ipc_year', 'instances', 'pools'] + \
        sorted({key for row in rows for key in row if key.startswith('pools_q')}) + \
        ['exhausted', 'timed_out', 'empty', 'plans']

    model_rows = []
    for spec in models.registry(cfg).values():
        for feature in spec.features:
            description, size_rule, dissimilarity = models.DIMENSION_DOC[feature.key]
            model_rows.append({
                'model': spec.name,
                'domains': 'all' if spec.domains is None else ' '.join(spec.domains),
                'feature': feature.key, 'description': description,
                'params': json.dumps(models._sortable(feature.params)),
                'dimension_size_rule': size_rule, 'dissimilarity': dissimilarity,
                'weight': feature.weight if feature.weight is not None
                          else f'uniform 1/{len(spec.features)}',
            })

    out = report_dir(cfg, 'setup')
    written = [write_csv(out / 'benchmark.csv', rows, columns),
               write_csv(out / 'models.csv', model_rows)]
    written.append(table(
        out / 'tables' / 'setup_benchmark.tex', 'tab:setup-benchmark',
        'Benchmark domains, the instances phase one solved and the pools it produced.',
        columns, [[row.get(column) for column in columns] for row in rows], digits=0))
    model_columns = ['model', 'domains', 'feature', 'dimension_size_rule', 'dissimilarity', 'weight']
    written.append(table(
        out / 'tables' / 'setup_models.tex', 'tab:setup-models',
        'The diversity models: one generic control on every domain, and the '
        'domain-specific models written by the authors as domain expert.',
        model_columns, [[row[column] for column in model_columns] for row in model_rows],
        aligns='llll' + 'l' + 'r'))
    written.append(manifest(cfg, 'setup', written, []))
    return written
