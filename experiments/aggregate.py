'''
    This file summarises cross-validation results across folds, seeds and models.
'''

import os
import glob
import json
import argparse
import collections
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']


def load_cells(runs_dir):
    '''Read the result of every completed cell under a runs directory.'''
    paths = sorted(glob.glob(os.path.join(runs_dir, '*', 'metrics.json')))
    return [json.load(open(p)) for p in paths]


def mean_sd(values):
    '''Compute the mean and the sample standard deviation of a list of values.'''
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def format_cell(values):
    '''Render a mean and a standard deviation as a string of fixed width.'''
    if not values:
        return '-'
    mean, sd = mean_sd(values)
    return f'{mean:.3f}+-{sd:.3f}'


def report_overall(cells, part):
    '''Print the held-out test performance of each model, averaged over folds and seeds.'''
    by_model = collections.defaultdict(list)
    for cell in cells:
        by_model[cell['model']].append(cell)
    print(f'=== held-out TEST, {part} mAP-50: mean +- sd over folds x seeds ===')
    header = f'{"model":14} {"n":>3} {"all":>14} ' + ' '.join(f'{n:>14}' for n in CLASS_NAMES)
    print(header)
    for model in sorted(by_model):
        group = by_model[model]
        row = f'{model:14} {len(group):3d} '
        row += f'{format_cell([c["test"][part]["map50"] for c in group]):>14} '
        for name in CLASS_NAMES:
            values = [c['test'][part]['ap50_by_class'][name] for c in group
                      if name in c['test'][part]['ap50_by_class']]
            row += f'{format_cell(values):>14} '
        print(row)


def report_per_fold(cells, part):
    '''Print the test performance of each fold, averaged over seeds.'''
    folds = sorted({c['fold'] for c in cells})
    by_model = collections.defaultdict(list)
    for cell in cells:
        by_model[cell['model']].append(cell)
    print(f'\n=== per-fold {part} mAP-50 (all classes), averaged over seeds ===')
    print(f'{"model":14} ' + ' '.join(f'{"f" + str(f):>7}' for f in folds))
    for model in sorted(by_model):
        row = f'{model:14} '
        for fold in folds:
            values = [c['test'][part]['map50'] for c in by_model[model] if c['fold'] == fold]
            row += f'{statistics.mean(values):>7.3f} ' if values else f'{"-":>7} '
        print(row)


def report_seed_spread(cells, part):
    '''Print how much the test performance varies across the seeds of one fold.'''
    groups = collections.defaultdict(list)
    for cell in cells:
        groups[(cell['model'], cell['fold'])].append(cell['test'][part]['map50'])
    spreads = [statistics.stdev(v) for v in groups.values() if len(v) > 1]
    print(f'\n=== seed spread: sd across seeds within a fold ({part} mAP-50) ===')
    if not spreads:
        print('  need more than one seed per fold')
        return
    print(f'  median={statistics.median(spreads):.4f} max={max(spreads):.4f} cells={len(spreads)}')


def main():
    '''Print every cross-validation summary table for one metric family.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    parser.add_argument('--part', default='seg', choices=['seg', 'box'])
    args = parser.parse_args()

    cells = load_cells(args.runs)
    print(f'loaded {len(cells)} cells from {args.runs}\n')
    report_overall(cells, args.part)
    report_per_fold(cells, args.part)
    report_seed_spread(cells, args.part)


if __name__ == '__main__':
    main()
