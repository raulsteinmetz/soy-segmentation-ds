'''
    This file uploads the checkpoints of one data division to the Hugging Face Hub.
'''

import os
import argparse

REPO_ID = 'rsteinmetz/growing-soy'


def collect(src, division):
    '''Map every run checkpoint to the path it takes on the Hub.'''
    files = {}
    for tag in sorted(os.listdir(src)):
        checkpoint = os.path.join(src, tag, 'weights', 'best.pt')
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f'no checkpoint for {tag} at {checkpoint}')
        files[checkpoint] = f'{division}/{tag}/best.pt'
    return files


def main():
    '''Upload every checkpoint trained under one division of the data.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--division', required=True, choices=('grouped', 'random'))
    parser.add_argument('--repo', default=REPO_ID)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    files = collect(args.src, args.division)
    total = sum(os.path.getsize(p) for p in files)
    print(f'uploading {len(files)} checkpoints ({total / 1e9:.1f} GB) to {args.repo}', flush=True)

    api = HfApi()
    for index, (local, remote) in enumerate(sorted(files.items(), key=lambda kv: kv[1]), 1):
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo, repo_type='model', commit_message='.')
        print(f'  [{index}/{len(files)}] {remote}', flush=True)
    print(f'uploaded {len(files)} checkpoints to {args.division}/')


if __name__ == '__main__':
    main()
