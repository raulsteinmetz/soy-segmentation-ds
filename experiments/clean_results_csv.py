'''
    This file trims appended YOLOv5 result files so that only the final training run remains.
'''

import os
import csv
import glob
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def last_run_start(rows):
    '''Find the row index where the final training run begins.'''
    starts = [i for i, r in enumerate(rows) if i > 0 and r and r[0].strip() == '0']
    return starts[-1]


def main():
    '''Trim every result file that contains more than one run.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', default=os.path.join(ROOT, 'runs'))
    args = parser.parse_args()

    for path in sorted(glob.glob(os.path.join(args.runs, '*', 'results.csv'))):
        rows = list(csv.reader(open(path)))
        epochs = [r[0].strip() for r in rows[1:] if r]
        if len(epochs) == len(set(epochs)):
            continue
        start = last_run_start(rows)
        shutil.copy(path, path.replace('results.csv', 'results_all_runs.csv'))
        with open(path, 'w', newline='') as handle:
            csv.writer(handle).writerows([rows[0]] + rows[start:])
        print(f'  {os.path.basename(os.path.dirname(path))}: '
              f'{len(rows)} -> {len(rows) - start + 1} rows')


if __name__ == '__main__':
    main()
