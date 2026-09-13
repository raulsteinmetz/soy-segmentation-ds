'''
    This file extracts the dataset frames from the source recordings.
'''

import os
import cv2
import argparse

SIZE = (600, 600)
STEP = 20


def video_names(folder):
    '''List the recording files found in a folder.'''
    return sorted(name for name in os.listdir(folder) if name.endswith('.MP4'))


def extract_frames(video, folder, out_dir, step, start):
    '''Write one frame out of every step frames of a recording, resized to the dataset resolution.'''
    capture = cv2.VideoCapture(os.path.join(folder, video))
    count = start
    success = True
    while success:
        success, image = capture.read()
        if count % step == start and success:
            image = cv2.resize(image, SIZE, interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(os.path.join(out_dir, f'{video[:-4]}_frame{count}.jpg'), image)
        count += 1


def main():
    '''Extract frames from every recording in the source folder.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--videos', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--step', type=int, default=STEP)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for video in video_names(args.videos):
        extract_frames(video, args.videos, args.out, args.step, args.start)
        print(f'  extracted {video}')


if __name__ == '__main__':
    main()
