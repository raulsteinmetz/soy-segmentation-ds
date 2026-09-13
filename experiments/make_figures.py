'''
    This file renders the cross-validation result figures as PNG files.
'''

import os
import csv
import glob
import json
import argparse
import collections
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']
CLASS_LABELS = ['Caruru weed', 'Grassy weed', 'Soy plant']
MODEL_ORDER = ['yolov5m-seg', 'yolov5l-seg', 'yolov5x-seg',
               'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg']
V5_COLOR = '#B0873B'
V8_COLOR = '#4A7A34'
CLASS_COLORS = ['#9C4A2C', '#B0873B', '#4A7A34']


def load_cells(runs_dir):
    '''Read the result of every completed cell.'''
    cells = {}
    for path in glob.glob(os.path.join(runs_dir, '*', 'metrics.json')):
        d = json.load(open(path))
        cells[(d['model'], d['fold'], d['seed'])] = d
    return cells


def model_color(model):
    '''Choose the colour that marks the family a model belongs to.'''
    return V5_COLOR if 'v5' in model else V8_COLOR


def style(ax):
    '''Apply the axis styling shared by every figure.'''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color='#DDDDDD', linewidth=0.7)
    ax.set_axisbelow(True)


def fig_model_comparison(cells, out):
    '''Plot the overall and per class test mAP-50 of every model.'''
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    panels = [('All classes', None)] + list(zip(CLASS_LABELS, CLASS_NAMES))
    for ax, (title, cls) in zip(axes, panels):
        means, errs, colors = [], [], []
        for model in MODEL_ORDER:
            vals = [c['test']['seg']['map50'] if cls is None
                    else c['test']['seg']['ap50_by_class'][cls]
                    for k, c in cells.items() if k[0] == model]
            means.append(statistics.mean(vals))
            errs.append(statistics.stdev(vals))
            colors.append(model_color(model))
        x = range(len(MODEL_ORDER))
        ax.bar(x, means, yerr=errs, capsize=3, color=colors,
               edgecolor='white', linewidth=0.8, error_kw={'linewidth': 1.1})
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.replace('-seg', '') for m in MODEL_ORDER],
                           rotation=45, ha='right', fontsize=9)
        ax.set_title(title, fontsize=11)
        style(ax)
    axes[0].set_ylabel('Test mask mAP-50')
    axes[0].set_ylim(0, 1.0)
    fig.suptitle('Held-out test performance, 6 folds x 3 seeds (error bars = sd across cells)',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)


def fig_per_fold(cells, out):
    '''Plot the test mAP-50 of each fold so that the variation between folds can be compared with the differences between models.'''
    folds = sorted({k[1] for k in cells})
    fig, ax = plt.subplots(figsize=(9, 4.6))
    width = 0.13
    for i, model in enumerate(MODEL_ORDER):
        means = [statistics.mean([c['test']['seg']['map50']
                                  for k, c in cells.items() if k[0] == model and k[1] == f])
                 for f in folds]
        ax.bar([f + (i - 2.5) * width for f in folds], means, width,
               label=model.replace('-seg', ''), color=model_color(model),
               alpha=1.0 if 'v8' in model else 0.6,
               edgecolor='white', linewidth=0.6)
    ax.set_xticks(folds)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.set_ylabel('Test mask mAP-50')
    ax.set_title('Fold-to-fold variation dwarfs model differences', fontsize=12)
    ax.legend(ncol=3, fontsize=8.5, frameon=False)
    ax.set_ylim(0, 0.85)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_paired(cells, out):
    '''Plot the difference between YOLOv8 and YOLOv5 on the cells that share a fold and a seed.'''
    keys = sorted({(k[1], k[2]) for k in cells})
    v8 = [m for m in MODEL_ORDER if 'v8' in m]
    v5 = [m for m in MODEL_ORDER if 'v5' in m]
    diffs = [statistics.mean([cells[(a, f, s)]['test']['seg']['map50'] for a in v8])
             - statistics.mean([cells[(b, f, s)]['test']['seg']['map50'] for b in v5])
             for f, s in keys]
    order = sorted(range(len(diffs)), key=lambda i: diffs[i])
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar(range(len(diffs)), [diffs[i] for i in order],
           color=[V8_COLOR if diffs[i] > 0 else '#9C4A2C' for i in order],
           edgecolor='white', linewidth=0.6)
    mean = statistics.mean(diffs)
    ax.axhline(0, color='#444444', linewidth=1)
    ax.axhline(mean, color='#444444', linestyle='--', linewidth=1.2,
               label=f'mean {mean:+.3f}')
    ax.set_xticks(range(len(diffs)))
    ax.set_xticklabels([f'f{keys[i][0]}s{keys[i][1]}' for i in order], fontsize=8)
    ax.set_ylabel('YOLOv8 minus YOLOv5 (mask mAP-50)')
    ax.set_title('Paired family difference on identical fold and seed cells', fontsize=12)
    ax.legend(frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_selection_premium(cells, out):
    '''Plot the gap between the selected checkpoint and the mean of the final validation epochs.'''
    fig, ax = plt.subplots(figsize=(8, 4.4))
    keys = ['metrics/mAP50(M)', 'metrics/mAP_0.5(M)']
    data, colors = [], []
    for model in MODEL_ORDER:
        vals = []
        for k, c in cells.items():
            if k[0] != model:
                continue
            key = next(x for x in keys if x in c['val_converged'])
            vals.append(c['val_at_best']['seg']['map50'] - c['val_converged'][key]['mean'])
        data.append(vals)
        colors.append(model_color(model))
    parts = ax.boxplot(data, patch_artist=True, widths=0.6,
                       medianprops={'color': 'white', 'linewidth': 1.6})
    for patch, color in zip(parts['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('none')
    ax.axhline(0, color='#444444', linewidth=1)
    ax.set_xticklabels([m.replace('-seg', '') for m in MODEL_ORDER],
                       rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Selected checkpoint minus converged mean')
    ax.set_title('Selection premium: picking the best epoch on the eval split\n'
                 'inflates mAP-50 by roughly 0.09', fontsize=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_training_curves(runs_dir, out):
    '''Plot validation mAP-50 across epochs for one seed of every model.'''
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for model in MODEL_ORDER:
        path = os.path.join(runs_dir, f'{model}_f0_s0', 'results.csv')
        rows = list(csv.reader(open(path)))
        header = [c.strip() for c in rows[0]]
        data = [r for r in rows[1:] if len(r) == len(header)]
        col = header.index('metrics/mAP50(M)' if 'metrics/mAP50(M)' in header
                           else 'metrics/mAP_0.5(M)')
        values = [float(r[col]) for r in data]
        ax.plot(range(1, len(values) + 1), values, label=model.replace('-seg', ''),
                color=model_color(model), linewidth=1.4,
                alpha=1.0 if 'v8' in model else 0.55)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation mask mAP-50')
    ax.set_title('Training curves, fold 0 seed 0 (early stopping disabled)', fontsize=12)
    ax.legend(ncol=3, fontsize=8.5, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_leakage(out):
    '''Plot how far held-out images sit from training images under the random splits and under the grouped folds.'''
    labels = ['Original\nrandom split', 'f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    ratios = [1.00, 1.45, 1.45, 1.31, 1.40, 1.44, 1.30]
    colors = ['#9C4A2C'] + [V8_COLOR] * 6
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(ratios)), ratios, color=colors, edgecolor='white', linewidth=0.8)
    ax.axhline(1.0, color='#9C4A2C', linestyle='--', linewidth=1.2)
    ax.text(len(ratios) - 0.4, 1.01, 'no separation from training set',
            ha='right', va='bottom', fontsize=8.5, color='#9C4A2C')
    ax.set_xticks(range(len(ratios)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('NN distance ratio (test vs train self-distance)')
    ax.set_ylim(0, 1.6)
    ax.set_title('Held-out separation in ResNet-18 feature space', fontsize=12)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main():
    '''Render every figure into the output directory.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cells = load_cells(args.runs)
    print(f'loaded {len(cells)} cells')
    fig_model_comparison(cells, os.path.join(args.out, 'model_comparison.png'))
    fig_per_fold(cells, os.path.join(args.out, 'per_fold_variance.png'))
    fig_paired(cells, os.path.join(args.out, 'paired_v8_vs_v5.png'))
    fig_selection_premium(cells, os.path.join(args.out, 'selection_premium.png'))
    fig_training_curves(args.runs, os.path.join(args.out, 'training_curves.png'))
    fig_leakage(os.path.join(args.out, 'split_leakage.png'))
    for name in sorted(os.listdir(args.out)):
        print(f'  {name}')


if __name__ == '__main__':
    main()
