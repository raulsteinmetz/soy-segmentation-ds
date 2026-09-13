'''
    This file writes the manifest of cross-validation cells to run.
'''

import argparse

MODELS = ['yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg',
          'yolov5m-seg', 'yolov5l-seg', 'yolov5x-seg']


def build_cells(models, folds, seeds):
    '''List every combination of model, fold and seed, grouped by model.'''
    return [(m, f, s) for m in models for f in range(folds) for s in range(seeds)]


def main():
    '''Write the manifest file and report how many chunks it needs.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='manifest_full.txt')
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--folds', type=int, default=6)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=8)
    parser.add_argument('--paired', action='store_true')
    args = parser.parse_args()

    cells = ([(m, i, i) for m in args.models for i in range(args.folds)] if args.paired
             else build_cells(args.models, args.folds, args.seeds))
    with open(args.out, 'w') as handle:
        for model, fold, seed in cells:
            handle.write(f'{model} {fold} {seed}\n')
    chunks = -(-len(cells) // args.chunk)
    print(f'{len(cells)} cells -> {args.out} ({chunks} chunks of {args.chunk})')


if __name__ == '__main__':
    main()
