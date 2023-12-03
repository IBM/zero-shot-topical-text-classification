import pandas as pd
from pathlib import Path
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--flow', type=str, default='deberta', choices=['flan', 'deberta'],
                    help='which model to use: deberta or flan')
def aggregate_all(out_dir, flow):
    base_dir = Path(os.path.join(out_dir, flow, 'aggregate'))
    files = list(base_dir.glob('*/*/*.csv'))
    df = pd.concat([pd.read_csv(f) for f in files])
    for metric in (set(df.columns)-set(['dataset'])):
        result_df = df.pivot_table(index=['dataset'], values=[metric])
        result_df.to_csv(os.path.join(base_dir, f'{metric}.csv'))
        print(f'{metric}\t{result_df.mean()[0]:.2f} for {len(result_df)} datasets')

def main():
    args = parser.parse_args()
    aggregate_all(args.output_dir, args.flow)

if __name__ == '__main__':
    main()