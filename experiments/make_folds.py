'''
    This file builds cross-validation folds in which no recording is shared between splits.
'''

import re
import os
import json
import glob
import shutil
import argparse
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']


def index_images(root):
    '''Group every image and label path in the dataset by source video.'''
    videos = collections.defaultdict(list)
    for image_path in glob.glob(os.path.join(root, 'images', '*')):
        name = os.path.basename(image_path)
        match = re.match(r'(.+?)_frame(\d+)_jpg\.rf\.', name)
        label_path = os.path.join(root, 'labels', os.path.splitext(name)[0] + '.txt')
        videos[match.group(1)].append((int(match.group(2)), image_path, label_path))
    return videos


def build_folds(videos):
    '''Assign videos to six folds so each video is tested exactly once.'''
    dates = sorted({video.split('-')[0] for video in videos})
    low_weed = [sorted(v for v in videos if v.startswith(d))[0] for d in dates]
    high_weed = [sorted(v for v in videos if v.startswith(d))[1] for d in dates]
    folds = {}
    for fold in range(6):
        test = [low_weed[fold], high_weed[(fold + 3) % 6]]
        valid = [low_weed[(fold + 1) % 6], high_weed[(fold + 4) % 6]]
        train = sorted(v for v in videos if v not in test + valid)
        folds[fold] = {'train': train, 'valid': sorted(valid), 'test': sorted(test)}
    return folds


def write_fold(out_dir, fold, assignment, videos):
    '''Create the split folders of one fold as symbolic links and write its data.yaml.'''
    fold_dir = os.path.join(out_dir, f'fold{fold}')
    shutil.rmtree(fold_dir, ignore_errors=True)
    counts = {}
    for split, split_videos in assignment.items():
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(fold_dir, split, sub))
        count = 0
        for video in split_videos:
            for _, image_path, label_path in videos[video]:
                for source, sub in ((image_path, 'images'), (label_path, 'labels')):
                    link = os.path.join(fold_dir, split, sub, os.path.basename(source))
                    os.symlink(os.path.abspath(source), link)
                count += 1
        counts[split] = count
    with open(os.path.join(fold_dir, 'data.yaml'), 'w') as handle:
        handle.write(f'path: {os.path.abspath(fold_dir)}\n')
        handle.write('train: train/images\nval: valid/images\ntest: test/images\n\n')
        handle.write(f'nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n')
    with open(os.path.join(fold_dir, 'videos.json'), 'w') as handle:
        json.dump(assignment, handle, indent=1)
    return counts


def main():
    '''Build every fold and report its split sizes.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=os.path.join(ROOT, 'dataset'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'splits', 'cv'))
    args = parser.parse_args()

    videos = index_images(args.src)
    print(f'indexed {sum(len(v) for v in videos.values())} images across {len(videos)} videos')
    folds = build_folds(videos)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'folds.json'), 'w') as handle:
        json.dump({str(k): v for k, v in folds.items()}, handle, indent=1)
    for fold, assignment in folds.items():
        counts = write_fold(args.out, fold, assignment, videos)
        tested = [v.split('-')[1] for v in assignment['test']]
        print(f'fold{fold}: train={counts["train"]:4d} valid={counts["valid"]:4d} '
              f'test={counts["test"]:4d} test_videos={tested}')


if __name__ == '__main__':
    main()
