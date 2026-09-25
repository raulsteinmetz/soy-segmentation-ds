'''
    This file evaluates every held-out recording on its own to give results for each growth stage.
'''

import os
import glob
import json
import argparse
import collections
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']
DATE_LABELS = {'20221216': 'Dec 16', '20221221': 'Dec 21', '20221227': 'Dec 27',
               '20230103': 'Jan 3', '20230110': 'Jan 10', '20230114': 'Jan 14'}


def build_video_splits(cv_dir, out_dir):
    '''Create a test split holding one recording, for every held-out recording.'''
    videos = {}
    for fold in range(6):
        assignment = json.load(open(os.path.join(cv_dir, f'fold{fold}', 'videos.json')))
        for video in assignment['test']:
            target = os.path.join(out_dir, video)
            for sub in ('images', 'labels'):
                os.makedirs(os.path.join(target, 'test', sub), exist_ok=True)
            for image in glob.glob(os.path.join(cv_dir, f'fold{fold}', 'test', 'images',
                                                f'{video}_*')):
                label = image.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
                for source, sub in ((image, 'images'), (label, 'labels')):
                    link = os.path.join(target, 'test', sub, os.path.basename(source))
                    if not os.path.islink(link):
                        os.symlink(os.path.realpath(source), link)
            yaml_path = os.path.join(target, 'data.yaml')
            with open(yaml_path, 'w') as handle:
                handle.write(f'path: {os.path.abspath(target)}\n')
                handle.write('train: test/images\nval: test/images\ntest: test/images\n\n')
                handle.write(f'nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n')
            videos[video] = (fold, yaml_path)
    return videos


def evaluate(model_path, yaml_path, iou, half):
    '''Evaluate one checkpoint on the split of a single recording.'''
    from ultralytics import YOLO
    metrics = YOLO(model_path).val(data=yaml_path, split='test', plots=False,
                                   iou=iou, half=half, verbose=False)
    seg = metrics.seg
    by_class = {CLASS_NAMES[int(c)]: float(v)
                for c, v in zip(seg.ap_class_index, seg.ap50)}
    return {'map50': float(seg.map50), 'ap50_by_class': by_class}


def main():
    '''Evaluate every cell on its held-out recordings and write the table of growth stages.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--cv', default=os.path.join(ROOT, 'splits', 'cv8'))
    parser.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    parser.add_argument('--splits', default=os.path.join(ROOT, 'splits', 'cv8_video'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'results', 'stage_results.json'))
    parser.add_argument('--model', default='yolov8m-seg')
    parser.add_argument('--iou', type=float, default=0.7)
    parser.add_argument('--half', action='store_true', default=True)
    args = parser.parse_args()

    videos = build_video_splits(args.cv, args.splits)
    print(f'built {len(videos)} single-video splits')

    results = []
    for video, (fold, yaml_path) in sorted(videos.items()):
        for seed in (0, 1, 2):
            weights = os.path.join(args.runs, f'{args.model}_f{fold}_s{seed}',
                                   'weights', 'best.pt')
            scores = evaluate(weights, yaml_path, args.iou, args.half)
            results.append({'video': video, 'date': video.split('-')[0], 'fold': fold,
                            'seed': seed, 'model': args.model, **scores})
            print(f'  {video} seed{seed} mAP50={scores["map50"]:.4f}', flush=True)

    json.dump(results, open(args.out, 'w'), indent=1)

    print(f'\n=== {args.model}: mask mAP-50 by growth stage (held-out video, mean over seeds) ===')
    print(f'{"stage":10} {"plot A":>16} {"plot B":>16} {"caruru":>9} {"grassy":>9} {"soy":>9}')
    by_date = collections.defaultdict(list)
    for row in results:
        by_date[row['date']].append(row)
    for date in sorted(by_date):
        rows = by_date[date]
        plots = sorted({r['video'] for r in rows})
        cells = []
        for video in plots:
            vals = [r['map50'] for r in rows if r['video'] == video]
            cells.append(f'{statistics.mean(vals):.3f}+-{statistics.pstdev(vals):.3f}')
        per_class = []
        for name in CLASS_NAMES:
            vals = [r['ap50_by_class'][name] for r in rows if name in r['ap50_by_class']]
            per_class.append(f'{statistics.mean(vals):.3f}' if vals else '-')
        print(f'{DATE_LABELS[date]:10} {cells[0]:>16} {cells[1]:>16} '
              f'{per_class[0]:>9} {per_class[1]:>9} {per_class[2]:>9}')


if __name__ == '__main__':
    main()
