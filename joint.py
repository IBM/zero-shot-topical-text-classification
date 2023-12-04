import json

import pandas as pd
import argparse
import os
import numpy as np
import utils

JOINT_VERSION="1.5"

SPLIT_NAMES = ['train', 'dev'] #, 'test']
parser = argparse.ArgumentParser(description='Prepare data for tweet evaluation')
parser.add_argument('--input_dirs', type=str, required=True, help='Comma separated input dirs')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--version', type=str, required=True, help='version to update on change of code')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument('--sampling_method', type=str, default='min_ds_dev', choices=['use_all', 'min_ds_dev', 'min_ds_dev_and_train', 'min_ds_train'],
        help='Method to sample data')

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = [f'{split}.csv' for split in SPLIT_NAMES]
__ARGPARSER__ = parser


## TODO: add training_method arg to calling function

def read_dataset(input_dir):
    splits = {}
    for split in SPLIT_NAMES:
        df = pd.read_csv(os.path.join(input_dir, f'{split}.csv'))
        print(f"{input_dir} {split} {len(df)}")
        splits[split] = df
    return splits


def main():
    args = parser.parse_args()
    utils.set_seed(args.seed)
    run_joint(input_dirs=args.input_dirs, output_dir=args.output_dir, sampling_method=args.sampling_method)

def run_joint(input_dirs, output_dir, sampling_method='min_ds_dev'):
    
    os.makedirs(output_dir, exist_ok=True)
    input_dirs = input_dirs.split(',')

    # read all
    datasets = [read_dataset(input_dir) for input_dir in input_dirs]
    joint_metadata = {}
    ds_names = []
    for input_dir in input_dirs:
        with open(os.path.join(input_dir, "metadata.json")) as f:
            metadata = json.load(f)
            ds_names.append(metadata['name'])
            joint_metadata[metadata['name']] = metadata

    with (open(os.path.join(output_dir, "joint_metadata.json"), "tw")) as f:
        json.dump(joint_metadata, f)


    # calculate min size
    smallest_size = {split: min([len(ds[split]) for ds in datasets]) for split in SPLIT_NAMES}
    print('smallest size', smallest_size)



    # concat and save
    for split in SPLIT_NAMES:
        if sampling_method == 'use_all':
            df = pd.concat([ds[split] for ds in datasets])
        elif ('min_ds' in sampling_method):
            if split in sampling_method:
                df = pd.concat([ds[split].sample(smallest_size[split]) for ds in datasets])
            else:
                df = pd.concat([ds[split] for ds in datasets])

        df = df.drop_duplicates().sample(frac=1)
        file = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(file, index=False)
        print(f'Wrote {file} with {len(df)} rows')





if __name__ == '__main__':
    main()
