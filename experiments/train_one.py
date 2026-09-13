'''
    This file trains and evaluates one YOLOv8 segmentation cell of the cross-validation matrix.
'''

import os
import csv
import sys
import json
import argparse
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']
EVAL_IOU = 0.7
EVAL_HALF = True


def pack_metrics(metrics):
    '''Collect box and mask metrics from an ultralytics results object.'''
    packed = {}
    for part in ('box', 'seg'):
        metric = getattr(metrics, part)
        classes = [int(c) for c in metric.ap_class_index]
        packed[part] = {
            'map50': float(metric.map50),
            'map': float(metric.map),
            'mp': float(metric.mp),
            'mr': float(metric.mr),
            'ap50_by_class': {CLASS_NAMES[c]: float(v) for c, v in zip(classes, metric.ap50)},
            'p_by_class': {CLASS_NAMES[c]: float(v) for c, v in zip(classes, metric.p)},
            'r_by_class': {CLASS_NAMES[c]: float(v) for c, v in zip(classes, metric.r)},
        }
    return packed


def converged_stats(csv_path, tail):
    '''Summarise the final validation epochs without taking the best epoch.'''
    rows = list(csv.DictReader(open(csv_path)))
    stats = {}
    for column in rows[0]:
        if not column.startswith('metrics/'):
            continue
        values = [float(row[column]) for row in rows][-tail:]
        stats[column] = {
            'mean': statistics.mean(values),
            'sd': statistics.pstdev(values),
            'n': len(values),
        }
    return stats


def main():
    '''Train one cell, evaluate it once on the held-out test split and write metrics.json.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cv', default=os.path.join(ROOT, 'splits', 'cv'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'runs'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--tail', type=int, default=30)
    parser.add_argument('--reeval', action='store_true')
    args = parser.parse_args()

    tag = f'{args.model}_f{args.fold}_s{args.seed}'
    data = os.path.join(args.cv, f'fold{args.fold}', 'data.yaml')
    target = os.path.join(args.out, tag, 'metrics.json')
    if os.path.exists(target) and not args.reeval:
        print(f'[skip] {tag} already complete')
        return

    from ultralytics import YOLO

    if args.reeval:
        model = YOLO(os.path.join(args.out, tag, 'weights', 'best.pt'))
    else:
        model = YOLO(f'{args.model}.pt')
        model.train(
            data=data, epochs=args.epochs, patience=args.epochs,
            batch=8, imgsz=640, seed=args.seed, deterministic=True,
            optimizer='SGD', lr0=0.01, lrf=0.01, momentum=0.937,
            weight_decay=0.0005, warmup_epochs=3.0, cos_lr=False,
            project=args.out, name=tag, exist_ok=True, plots=True, val=True, workers=8,
        )

    result = {
        'tag': tag,
        'fold': args.fold,
        'model': args.model,
        'seed': args.seed,
        'names': CLASS_NAMES,
        'test': pack_metrics(model.val(data=data, split='test', plots=False,
                                       iou=EVAL_IOU, half=EVAL_HALF)),
        'val_at_best': pack_metrics(model.val(data=data, split='val', plots=False,
                                              iou=EVAL_IOU, half=EVAL_HALF)),
        'val_converged': converged_stats(os.path.join(args.out, tag, 'results.csv'), args.tail),
    }
    with open(target, 'w') as handle:
        json.dump(result, handle, indent=1)
    print(f'[done] {tag} test seg mAP50={result["test"]["seg"]["map50"]:.4f}')


if __name__ == '__main__':
    main()
