'''
    This file writes the metadata table describing every image of the dataset.
'''

import os
import csv
import glob
import argparse

SESSION_PASS = {'20221216-GX010015': ('2022-12-16', 'A'), '20221216-GX010025': ('2022-12-16', 'B'),
                '20221221-GX010104': ('2022-12-21', 'A'), '20221221-GX010110': ('2022-12-21', 'B'),
                '20221227-GX010163': ('2022-12-27', 'A'), '20221227-GX010170': ('2022-12-27', 'B'),
                '20230103-GX010189': ('2023-01-03', 'A'), '20230103-GX010195': ('2023-01-03', 'B'),
                '20230110-GX010220': ('2023-01-10', 'A'), '20230110-GX010226': ('2023-01-10', 'B'),
                '20230114-GX010233': ('2023-01-14', 'A'), '20230114-GX010238': ('2023-01-14', 'B')}


def parse_name(name):
    '''Split an image file name into its recording video and frame number.'''
    stem = name.split('_jpg.rf.')[0]
    video, frame = stem.rsplit('_frame', 1)
    return video, int(frame)


def main():
    '''Write one row of metadata for every image in the dataset.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    rows = []
    for image_path in sorted(glob.glob(os.path.join(args.root, 'images', '*.jpg'))):
        name = os.path.basename(image_path)
        video, frame = parse_name(name)
        session, pass_id = SESSION_PASS[video]
        rows.append([name, video, session, pass_id, frame])

    out = os.path.join(args.root, 'metadata.csv')
    with open(out, 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'video', 'session', 'pass', 'frame'])
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    main()
