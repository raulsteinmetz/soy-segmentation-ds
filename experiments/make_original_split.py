'''
    This file builds random train, valid and test splits matched in size to the original split.
'''

import os
import glob
import json
import random
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPLITS = ('train', 'valid', 'test')
CLASS_NAMES = ['caruru_weed', 'grassy_weed', 'soy_plant']


def index_entries(src):
    '''List every image and its label in the dataset.'''
    entries = []
    for image_path in glob.glob(os.path.join(src, 'images', '*')):
        name = os.path.basename(image_path)
        label_path = os.path.join(src, 'labels', os.path.splitext(name)[0] + '.txt')
        entries.append((image_path, label_path))
    return sorted(entries)


def build_split(entries, sizes, seed):
    '''Draw one random split of the images at the original split sizes.'''
    order = list(entries)
    random.Random(seed).shuffle(order)
    assignment = {}
    start = 0
    for split in SPLITS:
        assignment[split] = order[start:start + sizes[split]]
        start += sizes[split]
    return assignment


def write_split(out_dir, fold, assignment):
    '''Create the folders of one split as symbolic links and write its data.yaml.'''
    fold_dir = os.path.join(out_dir, f'fold{fold}')
    shutil.rmtree(fold_dir, ignore_errors=True)
    counts = {}
    for split, pairs in assignment.items():
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(fold_dir, split, sub))
        for image_path, label_path in pairs:
            for source, sub in ((image_path, 'images'), (label_path, 'labels')):
                link = os.path.join(fold_dir, split, sub, os.path.basename(source))
                os.symlink(os.path.abspath(source), link)
        counts[split] = len(pairs)
    with open(os.path.join(fold_dir, 'data.yaml'), 'w') as handle:
        handle.write(f'path: {os.path.abspath(fold_dir)}\n')
        handle.write('train: train/images\nval: valid/images\ntest: test/images\n\n')
        handle.write(f'nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n')
    with open(os.path.join(fold_dir, 'images.json'), 'w') as handle:
        json.dump({split: sorted(os.path.basename(image) for image, _ in pairs)
                   for split, pairs in assignment.items()}, handle, indent=1)
    return counts


def main():
    '''Build one random split for each seed and report its sizes.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=os.path.join(ROOT, 'dataset'))
    parser.add_argument('--out', default=os.path.join(ROOT, 'splits', 'cvo'))
    parser.add_argument('--splits', type=int, default=3)
    parser.add_argument('--sizes', type=int, nargs=3, default=[658, 161, 181],
                        metavar=('TRAIN', 'VALID', 'TEST'))
    args = parser.parse_args()

    entries = index_entries(args.src)
    sizes = dict(zip(SPLITS, args.sizes))
    print(f'indexed {len(entries)} images, split sizes {sizes}')
    os.makedirs(args.out, exist_ok=True)
    for fold in range(args.splits):
        counts = write_split(args.out, fold, build_split(entries, sizes, fold))
        print(f'fold{fold} (seed {fold}): train={counts["train"]:4d} '
              f'valid={counts["valid"]:4d} test={counts["test"]:4d}')


if __name__ == '__main__':
    main()
