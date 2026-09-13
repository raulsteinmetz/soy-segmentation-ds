'''
    This file measures how far held-out images sit from the training set in feature space.
'''

import os
import re
import glob
import json
import argparse

import numpy as np
import torch
import torchvision
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def index_images(root):
    '''List every dataset image with its source video.'''
    records = {}
    for path in glob.glob(os.path.join(root, 'images', '*')):
        name = os.path.basename(path)
        video = re.match(r'(.+?)_frame(\d+)_jpg\.rf\.', name).group(1)
        records[name] = {'path': path, 'video': video}
    return records


def embed(records, keys, batch_size):
    '''Embed the images with a ResNet-18 trained on ImageNet and scale the features to unit length.'''
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    features = []
    with torch.no_grad():
        for start in range(0, len(keys), batch_size):
            chunk = keys[start:start + batch_size]
            batch = torch.stack([transform(Image.open(records[k]['path']).convert('RGB'))
                                 for k in chunk])
            features.append(net(batch))
    stacked = torch.cat(features).numpy()
    return stacked / np.linalg.norm(stacked, axis=1, keepdims=True)


def nearest_ratio(distances, index, train_keys, test_keys):
    '''Compare how far test images sit from training images with how far training images sit from each other.'''
    train_rows = [index[k] for k in train_keys]
    test_rows = [index[k] for k in test_keys]
    test_nearest = np.median(distances[np.ix_(test_rows, train_rows)].min(axis=1))
    train_nearest = np.median(distances[np.ix_(train_rows, train_rows)].min(axis=1))
    return test_nearest, train_nearest, test_nearest / train_nearest


def main():
    '''Report how far held-out images sit from training images under the random splits and the grouped folds.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default=os.path.join(ROOT, 'dataset'))
    parser.add_argument('--folds', default=os.path.join(ROOT, 'splits', 'cv', 'folds.json'))
    parser.add_argument('--random', default=os.path.join(ROOT, 'splits', 'cvo'))
    parser.add_argument('--batch', type=int, default=64)
    args = parser.parse_args()

    records = index_images(args.src)
    keys = sorted(records)
    print(f'embedding {len(keys)} images')
    features = embed(records, keys, args.batch)
    distances = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * (features @ features.T)))
    np.fill_diagonal(distances, np.inf)
    index = {k: i for i, k in enumerate(keys)}

    print(f'\n{"split":26} {"med test-train":>15} {"med train-train":>16} {"ratio":>8} {"n":>7}')
    print('-' * 76)
    random_ratios = []
    for fold_dir in sorted(glob.glob(os.path.join(args.random, 'fold*'))):
        assignment = json.load(open(os.path.join(fold_dir, 'images.json')))
        near, self_near, ratio = nearest_ratio(distances, index,
                                               assignment['train'], assignment['test'])
        random_ratios.append(ratio)
        print(f'{"  random " + os.path.basename(fold_dir):26} {near:15.4f} {self_near:16.4f} '
              f'{ratio:8.2f} {len(assignment["test"]):7d}')

    folds = json.load(open(args.folds))
    ratios = []
    for fold, assignment in sorted(folds.items()):
        train_videos = set(assignment['train'])
        test_videos = set(assignment['test'])
        train_keys = [k for k in keys if records[k]['video'] in train_videos]
        test_keys = [k for k in keys if records[k]['video'] in test_videos]
        near, self_near, ratio = nearest_ratio(distances, index, train_keys, test_keys)
        ratios.append(ratio)
        print(f'{"  grouped fold" + fold:26} {near:15.4f} {self_near:16.4f} '
              f'{ratio:8.2f} {len(test_keys):7d}')
    print('-' * 76)
    print(f'random mean ratio = {np.mean(random_ratios):.2f}, '
          f'grouped mean ratio = {np.mean(ratios):.2f}')


if __name__ == '__main__':
    main()
