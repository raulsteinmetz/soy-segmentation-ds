'''
    This file trains and evaluates one YOLOv5 segmentation cell of the cross-validation matrix.
'''

import os
import sys
import json
import argparse
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']
TABLE_FIELDS = 11
EVAL_IOU = '0.7'


def run(command, cwd, capture):
    '''Run a subcommand and stop the process if it fails.'''
    print('+ ' + ' '.join(command), flush=True)
    stream = subprocess.PIPE if capture else None
    result = subprocess.run(command, cwd=cwd, text=True, stdout=stream,
                            stderr=subprocess.STDOUT if capture else None)
    if result.returncode != 0:
        print(result.stdout or '')
        sys.exit(f'command failed with code {result.returncode}')
    return result.stdout or ''


def parse_table(text):
    '''Read the summary table of each class printed by the YOLOv5 validator.'''
    wanted = set(CLASS_NAMES) | {'all'}
    rows = {}
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) != TABLE_FIELDS or tokens[0] not in wanted:
            continue
        values = [float(t) for t in tokens[3:]]
        rows[tokens[0]] = {
            'box': dict(zip(('p', 'r', 'map50', 'map'), values[:4])),
            'seg': dict(zip(('p', 'r', 'map50', 'map'), values[4:])),
        }
    return rows


def reshape(rows):
    '''Convert the parsed rows into the same shape that train_one.py writes.'''
    packed = {}
    for part in ('box', 'seg'):
        present = [n for n in CLASS_NAMES if n in rows]
        packed[part] = {
            'map50': rows['all'][part]['map50'],
            'map': rows['all'][part]['map'],
            'mp': rows['all'][part]['p'],
            'mr': rows['all'][part]['r'],
            'ap50_by_class': {n: rows[n][part]['map50'] for n in present},
            'p_by_class': {n: rows[n][part]['p'] for n in present},
            'r_by_class': {n: rows[n][part]['r'] for n in present},
        }
    return packed


def main():
    '''Train one cell with the vendored YOLOv5 repo and write metrics.json.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--cv', default=os.path.join(ROOT, 'splits', 'cv'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'runs'))
    parser.add_argument('--repo', default=os.path.join(ROOT, 'yolov5'))
    parser.add_argument('--weights', default=os.path.join(ROOT, 'weights'))
    parser.add_argument('--checkpoints', default=os.path.join(ROOT, 'models', 'grouped'))
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--reeval', action='store_true')
    args = parser.parse_args()

    tag = f'{args.model}_f{args.fold}_s{args.seed}'
    data = os.path.join(args.cv, f'fold{args.fold}', 'data.yaml')
    run_dir = os.path.join(args.out, tag)
    target = os.path.join(run_dir, 'metrics.json')
    if os.path.exists(target) and not args.reeval:
        print(f'[skip] {tag} already complete')
        return

    if not args.reeval:
        run([sys.executable, 'segment/train.py', '--img', '640', '--batch', '8',
             '--epochs', str(args.epochs), '--data', data,
             '--weights', os.path.join(args.weights, f'{args.model}.pt'),
             '--optimizer', 'SGD', '--seed', str(args.seed),
             '--patience', str(args.epochs),
             '--project', args.out, '--name', tag, '--exist-ok',
             '--workers', '8', '--device', '0'], args.repo, False)

    best = (os.path.join(args.checkpoints, tag, 'best.pt') if args.reeval
            else os.path.join(run_dir, 'weights', 'best.pt'))

    def evaluate(task):
        '''Evaluate the selected checkpoint on one split.'''
        text = run([sys.executable, 'segment/val.py', '--data', data, '--weights', best,
                    '--task', task, '--img', '640', '--batch', '8', '--device', '0',
                    '--iou-thres', EVAL_IOU],
                   args.repo, True)
        return reshape(parse_table(text))

    result = {
        'tag': tag,
        'fold': args.fold,
        'model': args.model,
        'seed': args.seed,
        'names': CLASS_NAMES,
        'test': evaluate('test'),
        'val_at_best': evaluate('val'),
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(target, 'w') as handle:
        json.dump(result, handle, indent=1)
    print(f'[done] {tag} test seg mAP50={result["test"]["seg"]["map50"]:.4f}')


if __name__ == '__main__':
    main()
