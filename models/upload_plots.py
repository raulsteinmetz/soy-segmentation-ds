'''
    This file uploads the training plots of one data division to the Hugging Face Hub.
'''

import os
import argparse

REPO_ID = 'rsteinmetz/growing-soy'


def main():
    '''Upload every training plot of one division of the data.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--division', required=True, choices=('grouped', 'random'))
    parser.add_argument('--repo', default=REPO_ID)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    total = sum(os.path.getsize(os.path.join(root, name))
                for root, _, files in os.walk(args.src) for name in files)
    count = sum(len(files) for _, _, files in os.walk(args.src))
    print(f'uploading {count} plots ({total / 1e9:.2f} GB) to {args.repo}', flush=True)

    HfApi().upload_folder(folder_path=args.src, path_in_repo=f'plots/{args.division}',
                          repo_id=args.repo, repo_type='model',
                          allow_patterns=['*.png', '*.jpg'], commit_message='.')
    print(f'uploaded {count} plots to plots/{args.division}/')


if __name__ == '__main__':
    main()
