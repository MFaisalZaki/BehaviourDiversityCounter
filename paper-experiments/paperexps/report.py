"""Writing what the report stage produces: CSVs, booktabs tables, figures
where matplotlib is installed, and the manifest every run leaves behind."""

import csv
import json
import os
import platform
import subprocess
import sys
import time

from stats import holm, iqr, median, wilcoxon

#: Okabe-Ito, colour-blind safe.
PALETTE = ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']


def write_csv(path, rows, columns=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if columns is None:
        columns = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})
    return path


def _cell(value):
    if isinstance(value, float):
        return f'{value:.6g}'
    if isinstance(value, bool):
        return int(value)
    return '' if value is None else value


def fmt(value, digits=3):
    if value is None:
        return '--'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value != value:
            return '--'
        if abs(value) >= 1000 or (abs(value) < 0.001 and value != 0):
            return f'{value:.2e}'
        return f'{value:.{digits}f}'
    return str(value).replace('_', r'\_').replace('$', r'\$')


def latex_table(path, columns, rows, caption, label, digits=3, align=None):
    """A booktabs table; ``columns`` is a list of (key, header) pairs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    align = align or ('l' + 'r' * (len(columns) - 1))
    lines = [r'\begin{table}[t]', r'\centering', r'\caption{' + caption + '}',
             r'\label{' + label + '}', r'\small',
             r'\begin{tabular}{' + align + '}', r'\toprule',
             ' & '.join(header for _, header in columns) + r' \\', r'\midrule']
    for row in rows:
        lines.append(' & '.join(fmt(row.get(key), digits) for key, _ in columns) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))
    return path


def matplotlib_or_none():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return plt
    except Exception:                                                   # noqa: BLE001
        return None


def save_figure(plt, fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def group_by(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    return groups


def relative_score_matrix(rows, group_keys, selector_key, selectors, score_columns):
    """Mean relative score: per group, each score column divided by the best
    selector's value in that group (groups whose best is not positive are
    skipped for that column), then averaged over groups.  Returns
    ``({selector: {column: mean}}, {column: groups counted})``.
    """
    sums = {selector: {column: 0.0 for column in score_columns} for selector in selectors}
    counts = {column: 0 for column in score_columns}
    for group in group_by(rows, group_keys).values():
        by_selector = {row[selector_key]: row for row in group}
        if any(selector not in by_selector for selector in selectors):
            continue
        for column in score_columns:
            values = {selector: by_selector[selector].get(column) for selector in selectors}
            if any(value is None for value in values.values()):
                continue
            best = max(values.values())
            if best <= 0:
                continue
            for selector, value in values.items():
                sums[selector][column] += value / best
            counts[column] += 1
    matrix = {selector: {column: (sums[selector][column] / counts[column]) if counts[column] else None
                         for column in score_columns} for selector in selectors}
    return matrix, counts


def matrix_table(path, matrix, selectors, score_columns, caption, label, headers=None):
    headers = headers or {column: column for column in score_columns}
    columns = [('selector', 'selector')] + [(column, headers[column]) for column in score_columns]
    rows = [{'selector': selector, **matrix[selector]} for selector in selectors]
    return latex_table(path, columns, rows, caption, label)


def paired_wilcoxon_rows(rows, pair_keys, selector_key, first, second, value_key, comparisons):
    """One Wilcoxon row per comparison group: ``first`` against ``second``
    on ``value_key``, paired by ``pair_keys``, Holm-corrected across the rows."""
    out = []
    for group_name, group_rows in comparisons:
        first_values, second_values = {}, {}
        for row in group_rows:
            key = tuple(row.get(k) for k in pair_keys)
            if row[selector_key] == first:
                first_values[key] = row[value_key]
            elif row[selector_key] == second:
                second_values[key] = row[value_key]
        keys = sorted(set(first_values) & set(second_values))
        x = [first_values[k] for k in keys]
        y = [second_values[k] for k in keys]
        test = wilcoxon(x, y) if keys else {'n': 0, 'p': None, 'median_diff': None, 'iqr_diff': None, 'statistic': None, 'method': None}
        out.append({'comparison': group_name, 'first': first, 'second': second, 'n': test['n'],
                    'median_diff': test['median_diff'], 'iqr_diff': test['iqr_diff'],
                    'statistic': test['statistic'], 'p': test['p'], 'method': test['method']})
    adjusted = holm([row['p'] for row in out])
    for row, p_holm in zip(out, adjusted):
        row['p_holm'] = p_holm
    return out


def _git_revision():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:                                                   # noqa: BLE001
        return None


def versions():
    out = {'python': sys.version.split()[0], 'platform': platform.platform(),
           'git': _git_revision()}
    for name in ('unified_planning', 'numpy', 'lark', 'scipy', 'matplotlib'):
        try:
            module = __import__(name)
            out[name] = getattr(module, '__version__', 'installed')
        except Exception:                                               # noqa: BLE001
            out[name] = None
    try:
        from importlib.metadata import version
        out['behaviour-diversity-counter'] = version('behaviour-diversity-counter')
    except Exception:                                                   # noqa: BLE001
        out['behaviour-diversity-counter'] = None
    return out


def write_manifest(path, name, config_file, params, started, outputs, notes=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    manifest = {
        'experiment': name, 'config_file': os.path.abspath(config_file),
        'seed': params.get('seed'), 'parameters': {k: v for k, v in params.items() if k != 'output'},
        'versions': versions(), 'started': started,
        'ended': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'outputs': outputs, 'notes': notes or [],
    }
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=2, default=str)
    return path


def summary(values):
    values = [v for v in values if v is not None]
    if not values:
        return {'n': 0, 'min': None, 'p5': None, 'median': None, 'iqr': None, 'max': None}
    from stats import quantile
    return {'n': len(values), 'min': min(values), 'p5': quantile(values, 0.05),
            'median': median(values), 'iqr': iqr(values), 'max': max(values)}
