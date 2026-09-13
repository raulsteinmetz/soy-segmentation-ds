'''
    This file renders a PDF of predictions on the images most densely covered by weeds.
'''

import os
import re
import glob
import json
import argparse
import collections

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']
CLASS_COLORS = [(230, 0, 255), (0, 230, 255), (255, 90, 0)]
WEED_CLASSES = (0, 1)


def test_fold_of_video(cv_dir):
    '''Map each source video to the fold that holds it in its test split.'''
    mapping = {}
    for fold in range(6):
        assignment = json.load(open(os.path.join(cv_dir, f'fold{fold}', 'videos.json')))
        for video in assignment['test']:
            mapping[video] = fold
    return mapping


def read_polygons(label_path, width, height):
    '''Read the polygons and classes of one label file, scaled to pixels.'''
    polygons, classes = [], []
    for line in open(label_path):
        parts = line.split()
        if not parts:
            continue
        points = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        points[:, 0] *= width
        points[:, 1] *= height
        polygons.append(points)
        classes.append(int(parts[0]))
    return polygons, classes


def weed_fraction(label_path, width, height):
    '''Measure the fraction of the frame covered by weed masks.'''
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons, classes = read_polygons(label_path, width, height)
    for polygon, cls in zip(polygons, classes):
        if cls in WEED_CLASSES:
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
    return float(np.count_nonzero(mask)) / mask.size


def rank_images(dataset, width, height):
    '''Rank every dataset image by how much of it weeds cover.'''
    ranked = []
    for image_path in sorted(glob.glob(os.path.join(dataset, 'images', '*.jpg'))):
        name = os.path.basename(image_path)
        label_path = os.path.join(dataset, 'labels', os.path.splitext(name)[0] + '.txt')
        video = re.match(r'(.+?)_frame(\d+)_jpg\.rf\.', name).group(1)
        ranked.append((weed_fraction(label_path, width, height), video, image_path))
    ranked.sort(reverse=True)
    return ranked


def select(ranked, count, per_video, ceiling):
    '''Take the densest frames, skipping outliers and limiting how many come from one recording.'''
    taken = []
    seen = collections.Counter()
    for fraction, video, image_path in ranked:
        if fraction > ceiling or seen[video] >= per_video:
            continue
        taken.append((fraction, video, image_path))
        seen[video] += 1
        if len(taken) == count:
            break
    return taken


def draw_instances(image, polygons, classes):
    '''Draw the polygons onto a copy of the image, coloured by class.'''
    canvas = image.copy()
    overlay = image.copy()
    for polygon, cls in zip(polygons, classes):
        points = polygon.astype(np.int32)
        cv2.fillPoly(overlay, [points], CLASS_COLORS[cls])
        cv2.polylines(canvas, [points], True, (255, 255, 255), 3)
        cv2.polylines(canvas, [points], True, CLASS_COLORS[cls], 2)
    return cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0)


def checkpoint_path(weights_dir, model, fold, seed):
    '''Build the path of the checkpoint trained without the fold being shown.'''
    flat = os.path.join(weights_dir, f'{model}_f{fold}_s{seed}.pt')
    nested = os.path.join(weights_dir, f'{model}_f{fold}_s{seed}', 'weights', 'best.pt')
    return flat if os.path.exists(flat) else nested


def main():
    '''Render the weed heavy predictions into a PDF for use as the main figure.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=os.path.join(ROOT, 'dataset'))
    parser.add_argument('--cv', default=os.path.join(ROOT, 'splits', 'cv'))
    parser.add_argument('--weights', default=os.path.join(ROOT, 'models', 'grouped'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'figures', 'teaser.pdf'))
    parser.add_argument('--model', default='yolov8m-seg')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--conf', type=float, default=0.35)
    parser.add_argument('--count', type=int, default=6)
    parser.add_argument('--columns', type=int, default=3)
    parser.add_argument('--per-video', type=int, default=2)
    parser.add_argument('--ceiling', type=float, default=0.6)
    parser.add_argument('--truth', action='store_true')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from ultralytics import YOLO

    probe = cv2.imread(sorted(glob.glob(os.path.join(args.dataset, 'images', '*.jpg')))[0])
    height, width = probe.shape[:2]

    folds = test_fold_of_video(args.cv)
    ranked = rank_images(args.dataset, width, height)
    chosen = select(ranked, args.count, args.per_video, args.ceiling)
    print(f'weed coverage of the {args.count} densest frames '
          f'{[round(f * 100, 1) for f, _, _ in chosen]} percent')

    loaded = {}
    panels = []
    for fraction, video, image_path in chosen:
        fold = folds[video]
        if fold not in loaded:
            weights = checkpoint_path(args.weights, args.model, fold, args.seed)
            print(f'  loading fold {fold} from {weights}', flush=True)
            loaded[fold] = YOLO(weights)
        result = loaded[fold].predict(source=image_path, conf=args.conf, verbose=False)[0]
        image = cv2.imread(image_path)
        polygons = [] if result.masks is None else list(result.masks.xy)
        classes = [int(c) for c in result.boxes.cls]
        rendered = draw_instances(image, polygons, classes)
        if args.truth:
            truth_polygons, truth_classes = read_polygons(
                os.path.join(args.dataset, 'labels',
                             os.path.splitext(os.path.basename(image_path))[0] + '.txt'),
                width, height)
            rendered = np.hstack([draw_instances(image, truth_polygons, truth_classes), rendered])
        panels.append((rendered, fraction, fold, len(result.boxes)))
        print(f'  {os.path.basename(image_path)[:38]} fold {fold} '
              f'{len(result.boxes)} instances, {fraction * 100:.1f} percent weed')

    rows = -(-len(panels) // args.columns)
    figure, axes = plt.subplots(rows, args.columns, layout='constrained',
                                figsize=(4.2 * args.columns * (2 if args.truth else 1), 4.5 * rows))
    for axis, (rendered, fraction, fold, count) in zip(np.ravel(axes), panels):
        axis.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        axis.set_title(f'{fraction * 100:.1f} percent weed cover, held out in fold {fold}',
                       fontsize=9)
        axis.axis('off')
    for axis in np.ravel(axes)[len(panels):]:
        axis.axis('off')

    figure.legend(handles=[Patch(facecolor=np.array(c[::-1]) / 255, label=n.replace('_', ' '))
                           for n, c in zip(CLASS_NAMES, CLASS_COLORS)],
                  loc='outside lower center', ncol=3, frameon=False, fontsize=11)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    figure.savefig(args.out, format='pdf', bbox_inches='tight')
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
