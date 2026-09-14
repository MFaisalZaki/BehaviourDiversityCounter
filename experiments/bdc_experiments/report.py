"""What every report writes: CSV first, a booktabs table second, a figure
third, and one manifest naming the code, the data and the counts behind them.

Reports are pure functions of the raw dumps. Nothing here loads a pool, builds
a counter or runs a selection: a selected set at k is the first k plans of the
recorded run, and its indicators are read off the behaviour dump's matrix.
"""

import csv
import json
import platform
import statistics
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from bdc_experiments import SCHEMA_VERSION, reference
from bdc_experiments.config import results_root
from bdc_experiments.runner import INDICATORS, load_dump, load_results

#: Okabe-Ito, safe under the three common colour vision deficiencies.
PALETTE = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442', '#000000')

#: Stated in every report: the paper leaves tie-breaking arbitrary, the
#: implementation does not.
TIE_RULE = ('Ties are broken deterministically by the lowest index in the '
            'cost-sorted pool, at the opening pair and at every later step.')

PACKAGES = ('behaviour-diversity-counter', 'unified-planning', 'numpy', 'scipy',
            'matplotlib', 'lark')


def report_dir(cfg, name):
    path = results_root(cfg) / 'reports' / name
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
# Reading the selection results
# ----------------------------------------------------------------------

def usable(results):
    """The results a report may read numbers from: neither failed nor skipped."""
    return [r for r in results
            if not r.get('error') and not (r.get('extra') or {}).get('skipped')]


def selections(cfg, keep=lambda name: True):
    """The usable selection results whose model name ``keep`` accepts, each
    with its behaviour dump attached under ``dump``."""
    found = []
    for result in usable(load_results(cfg, 'select')):
        if keep(result['model']['name']):
            result['dump'] = load_dump(cfg, result)
            found.append(result)
    return found


def by_pool(results):
    """``(instance, pool_stem) -> {model name: result}``."""
    grouped = {}
    for result in results:
        pool = result['pool']
        grouped.setdefault((pool['instance'], pool['pool_stem']), {})[result['model']['name']] = result
    return grouped


def entry(result, indicator, kappa):
    """The recorded run of one selection: kappa is matched for B-Novelty and
    ignored for the three kappa-free rules, which were run once."""
    return next(e for e in result['extra']['selections']
                if e['indicator'] == indicator and (indicator != 'bnovelty' or e['kappa'] == kappa))


def score(dump, chosen, kappa):
    """The four indicators of a set of behaviours, given as indices into the
    dump's ``distinct`` list (repeats allowed), off the dump's matrix. Under
    the stability model there is no matrix and the tuple is the action set."""
    if dump['matrix'] is not None:
        d = lambda i, j: dump['matrix'][i][j]
    else:
        d = lambda i, j: reference.ref_stability(dump['distinct'][i][0].split(' ; '),
                                                 dump['distinct'][j][0].split(' ; '))
    return {name: reference.ref_indicator(name, list(chosen), d, kappa) for name in INDICATORS}


def head(result, k=None, kappa=None):
    """The nine mandatory fields of a row about this result."""
    pool, dump = result['pool'], result['dump']
    return {'instance': pool['instance'], 'domain': pool['domain'], 'q': pool['q'],
            'N': pool['requested'], 'model': result['model']['name'], 'k': k, 'kappa': kappa,
            'pool_size': pool['size'], 'b': len(dump['distinct'])}


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------

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


def group(rows, keys):
    """``tuple of key values -> rows``, in first-appearance order."""
    grouped = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    return grouped


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------

def figure(size=(5.2, 3.2), **kwargs):
    """A matplotlib figure with no title text: the caption carries the words."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(figsize=size, **kwargs)
    for ax in (axes.flat if hasattr(axes, 'flat') else [axes]):
        ax.set_prop_cycle(color=list(PALETTE))
    return fig, axes


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # No creation date in the PDF: `bdcexp report` has to rebuild the whole of
    # reports/ byte for byte from the dumps, and a timestamp inside a figure
    # would make every rebuild look like a change.
    fig.savefig(path, metadata={'CreationDate': None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def boxes(ax, groups, series, values_of):
    """Grouped box plots: one box per (group, series), a colour per series.
    ``values_of(group, s)`` gives the sample; empty samples are left out."""
    width = 0.8 / max(len(series), 1)
    for offset, s in enumerate(series):
        colour = PALETTE[offset % len(PALETTE)]
        samples = {i: values_of(g, s) for i, g in enumerate(groups)}
        samples = {i: v for i, v in samples.items() if v}
        if samples:
            drawn = ax.boxplot(list(samples.values()), patch_artist=True, manage_ticks=False,
                               widths=width * 0.8,
                               positions=[i + (offset - (len(series) - 1) / 2) * width
                                          for i in samples])
            for box in drawn['boxes']:
                box.set_facecolor(colour)
            for median in drawn['medians']:
                median.set_color('black')
        ax.plot([], [], color=colour, linewidth=6, label=str(s))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([str(g) for g in groups])


# ----------------------------------------------------------------------
# The manifest, and the setup report
# ----------------------------------------------------------------------

def versions():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def manifest(cfg, name, outputs, results, extra=None):
    """One per report: the code, the data and the counts behind the tables."""
    from bdc_experiments import benchmark, generate
    from bdc_experiments.runner import git_revision
    counts = {'total': len(results),
              'failed': sum(1 for r in results if r.get('error')),
              'skipped': sum(1 for r in results if (r.get('extra') or {}).get('skipped'))}
    counts['ok'] = counts['total'] - counts['failed'] - counts['skipped']
    record = {
        'schema': 'manifest', 'version': SCHEMA_VERSION, 'report': name,
        'written': datetime.now(timezone.utc).isoformat(),
        'git': git_revision(),
        'config': {'path': cfg['meta']['config_path'], 'hash': cfg['meta']['config_hash'],
                   'run_name': cfg['meta']['run_name']},
        'benchmark': {'source': cfg['benchmark']['source'],
                      'pinned_commit': cfg['benchmark']['commit'],
                      'checkout_revision': benchmark.checkout_revision(cfg)},
        'planner': {**generate.PLANNER,
                    'archive': str(generate.archive(cfg) or ''),
                    'resources': str(generate._data(cfg, 'benchmark', 'resources') or ''),
                    'time_limit_s': cfg['generation']['time_limit_s'],
                    'limits': 'those of the runs that produced the archive; a pool short of its '
                              'size at or over the time limit is recorded as timed_out'},
        'selection_time_limit_s': cfg['run']['time_limit_selection_s'],
        'seed': cfg['run']['seed'],
        'selection_grid': dict(cfg['selection']),
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
    path = report_dir(cfg, name) / 'manifest.json'
    path.write_text(json.dumps(record, indent=1))
    return path


def setup_report(cfg):
    """The two files the paper's Setup subsection consumes, from the pools alone."""
    from bdc_experiments import models, pools
    per_domain = {}
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
    rows = sorted(({**e, 'instances': len(e['instances'])} for e in per_domain.values()),
                  key=lambda row: row['domain'])
    columns = ['domain', 'ipc_year', 'instances', 'pools'] + \
        sorted({key for row in rows for key in row if key.startswith('pools_q')}) + \
        ['exhausted', 'timed_out', 'empty', 'plans']

    model_rows = []
    for spec in [models.generic_spec(cfg), *models.PER_DOMAIN, models.STABILITY]:
        for feature in spec.features:
            description, size_rule, dissimilarity = models.DIMENSION_DOC[feature.key]
            model_rows.append({
                'model': spec.name,
                'domains': 'all' if spec.domains is None else ' '.join(spec.domains),
                'feature': feature.key, 'description': description,
                'params': json.dumps(models._sortable(feature.params)),
                'dimension_size_rule': size_rule, 'dissimilarity': dissimilarity,
                'weight': feature.weight if feature.weight is not None
                          else f'uniform 1/{len(spec.features)}'})

    out = report_dir(cfg, 'setup')
    model_columns = ['model', 'domains', 'feature', 'dimension_size_rule', 'dissimilarity', 'weight']
    written = [
        write_csv(out / 'benchmark.csv', rows, columns),
        write_csv(out / 'models.csv', model_rows),
        table(out / 'tables' / 'setup_benchmark.tex', 'tab:setup-benchmark',
              'Benchmark domains, the instances phase one solved and the pools it produced.',
              columns, [[row.get(column) for column in columns] for row in rows], digits=0),
        table(out / 'tables' / 'setup_models.tex', 'tab:setup-models',
              'The diversity models: the generic control on every domain, the domain-specific '
              'models written by the authors as domain expert, and the literature\'s model as '
              'one feature, the stability distance over action sets. E2 reweights the '
              'astronaut\'s model and E3 varies the generic model\'s feature count; those '
              'variants are named in the result files.',
              model_columns, [[row[column] for column in model_columns] for row in model_rows],
              aligns='lllllr'),
    ]
    return written + [manifest(cfg, 'setup', written, [])]
