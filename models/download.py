'''
    This file downloads the trained GrowingSoy checkpoints from the Hugging Face Hub.
'''

import os
import argparse

REPO_ID = 'rsteinmetz/growing-soy'
ROOT = os.path.dirname(os.path.abspath(__file__))
DIVISIONS = ('grouped', 'random')
MODELS = ('yolov5m-seg', 'yolov5l-seg', 'yolov5x-seg',
          'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg')


def build_patterns(division, model):
    '''Build the patterns that select the requested checkpoints on the Hub.'''
    divisions = DIVISIONS if division == 'all' else (division,)
    models = MODELS if model == 'all' else (model,)
    return [f'{d}/{m}_*/best.pt' for d in divisions for m in models]


def main():
    '''Download the selected checkpoints into this directory.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--division', default='all', choices=('all',) + DIVISIONS)
    parser.add_argument('--model', default='all', choices=('all',) + MODELS)
    parser.add_argument('--out', default=ROOT)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=REPO_ID, repo_type='model',
                             allow_patterns=build_patterns(args.division, args.model),
                             local_dir=args.out)
    count = sum(1 for _, _, files in os.walk(path)
                for name in files if name.endswith('.pt'))
    print(f'downloaded {count} checkpoints to {path}')


if __name__ == '__main__':
    main()
