'''
    This file renders held-out predictions across every growth stage of the plantation.
'''

import os
import re
import glob
import json
import argparse

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']
CLASS_COLORS = [(255, 0, 230), (255, 230, 0), (0, 90, 255)]
DATE_LABELS = {'20221216': 'Dec 16', '20221221': 'Dec 21', '20221227': 'Dec 27',
               '20230103': 'Jan 3', '20230110': 'Jan 10', '20230114': 'Jan 14'}


def test_fold_of_video(cv_dir):
    '''Map each source video to the fold that holds it in its test split.'''
    mapping = {}
    for fold in range(6):
        assignment = json.load(open(os.path.join(cv_dir, f'fold{fold}', 'videos.json')))
        for video in assignment['test']:
            mapping[video] = fold
    return mapping


def pick_images(cv_dir, folds, per_video):
    '''Choose evenly spaced test images from every video.'''
    picks = []
    for video, fold in sorted(folds.items()):
        paths = sorted(glob.glob(os.path.join(cv_dir, f'fold{fold}', 'test', 'images', f'{video}_*')),
                       key=lambda p: int(re.search(r'_frame(\d+)_', p).group(1)))
        for q in [0.35, 0.65][:per_video]:
            picks.append((video, fold, paths[int(len(paths) * q)]))
    return picks


def draw_instances(image, polygons, classes):
    '''Draw the polygons onto a copy of the image, coloured by class.'''
    canvas = image.copy()
    overlay = image.copy()
    for poly, cls in zip(polygons, classes):
        pts = poly.astype(np.int32)
        cv2.fillPoly(overlay, [pts], CLASS_COLORS[cls])
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 3)
        cv2.polylines(canvas, [pts], True, CLASS_COLORS[cls], 2)
    return cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0)


def truth_polygons(label_path, width, height):
    '''Read ground truth polygons from a YOLO segmentation label file.'''
    polygons, classes = [], []
    for line in open(label_path):
        parts = line.split()
        if not parts:
            continue
        pts = np.array([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        pts[:, 0] *= width
        pts[:, 1] *= height
        polygons.append(pts)
        classes.append(int(parts[0]))
    return polygons, classes


def label_bar(image, text, height=34):
    '''Add a caption bar above an image.'''
    bar = np.full((height, image.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, text, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1,
                cv2.LINE_AA)
    return np.vstack([bar, image])


def legend_strip(width):
    '''Build a strip showing the colour used for each of the three classes.'''
    strip = np.full((40, width, 3), 245, dtype=np.uint8)
    x = 14
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        cv2.rectangle(strip, (x, 13), (x + 22, 29), color, -1)
        cv2.putText(strip, name.replace('_', ' '), (x + 30, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        x += 240
    return strip


def main():
    '''Render a grid of growth stages and a comparison of ground truth with predictions.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', default=os.path.join(ROOT, 'splits', 'cv8'))
    parser.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    parser.add_argument('--model', default='yolov8m-seg')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--conf', type=float, default=0.35)
    args = parser.parse_args()

    from ultralytics import YOLO

    os.makedirs(args.out, exist_ok=True)
    folds = test_fold_of_video(args.cv)
    picks = pick_images(args.cv, folds, per_video=2)
    loaded = {}
    rendered = {}

    for video, fold, path in picks:
        if fold not in loaded:
            weights = os.path.join(args.runs, f'{args.model}_f{fold}_s{args.seed}',
                                   'weights', 'best.pt')
            print(f'  loading fold{fold}: {weights}', flush=True)
            loaded[fold] = YOLO(weights)
        result = loaded[fold].predict(source=path, conf=args.conf, verbose=False)[0]
        image = cv2.imread(path)
        polygons = [] if result.masks is None else list(result.masks.xy)
        classes = [int(c) for c in result.boxes.cls]
        rendered[(video, path)] = draw_instances(image, polygons, classes)
        print(f'  {os.path.basename(path)[:34]} fold{fold} '
              f'{len(result.boxes)} instances')

    dates = sorted({v.split('-')[0] for v in folds})
    plots = {d: sorted(v for v in folds if v.startswith(d)) for d in dates}

    rows = []
    for row_index, plot_label in enumerate(['Plot A, lower weed pressure',
                                            'Plot B, higher weed pressure']):
        tiles = []
        for date in dates:
            video = plots[date][row_index]
            path = [p for v, _, p in picks if v == video][0]
            tile = cv2.resize(rendered[(video, path)], (320, 320))
            caption = f'{DATE_LABELS[date]}  held out in fold {folds[video]}'
            tiles.append(label_bar(tile, caption))
        row = np.hstack(tiles)
        rows.append(label_bar(row, plot_label, height=40))
    grid = np.vstack(rows)
    grid = np.vstack([grid, legend_strip(grid.shape[1])])
    out_grid = os.path.join(args.out, 'predictions_by_stage.png')
    cv2.imwrite(out_grid, grid)
    print(f'wrote {out_grid} {grid.shape[1]}x{grid.shape[0]}')

    pairs = []
    for date in dates:
        video = plots[date][1]
        path = [p for v, _, p in picks if v == video][0]
        label_path = path.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        source = cv2.imread(path)
        polygons, classes = truth_polygons(label_path, source.shape[1], source.shape[0])
        truth = cv2.resize(draw_instances(source, polygons, classes), (300, 300))
        pred = cv2.resize(rendered[(video, path)], (300, 300))
        pairs.append(np.vstack([label_bar(truth, f'{DATE_LABELS[date]}  ground truth'),
                                label_bar(pred, f'{DATE_LABELS[date]}  prediction')]))
    comparison = np.hstack(pairs)
    comparison = np.vstack([comparison, legend_strip(comparison.shape[1])])
    out_pair = os.path.join(args.out, 'prediction_vs_truth.png')
    cv2.imwrite(out_pair, comparison)
    print(f'wrote {out_pair} {comparison.shape[1]}x{comparison.shape[0]}')


if __name__ == '__main__':
    main()
