import pandas as pd
import argparse
import os
import json
import utils

from math import floor

CALCULATE_SAMPLING_VERSION = "1.2"

parser = argparse.ArgumentParser(description='Calculate sampling size each class and dataset')
parser.add_argument('--dataset_names', type=str, required=True, help='Comma serapated input dataset names')
parser.add_argument('--input_dirs', type=str, required=True, help='Comma serapated directories of the datasets')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--version', type=str, required=True, help='version to update on change of code')
parser.add_argument('--limit_class_count', type=int, action='store', default=None, help='max number of class count in train')
parser.add_argument('--distribute_evenly', action='store_true',
                    help='should be small datasets represent their classes in fair fashion?')
parser.add_argument("--seed", type=int, default=42)

__OUTPUT_DIR_ARG__ = "output_dir"

SPLIT_NAMES = ['train', 'dev']
SUFFIX = '_stat.csv'
__OUTPUTS__ = [f'classes_{split}{SUFFIX}' for split in SPLIT_NAMES] + \
    [f'sizes_{split}{SUFFIX}' for split in SPLIT_NAMES] + \
    [f'sampling_{split}{SUFFIX}' for split in SPLIT_NAMES]

__ARGPARSER__ = parser

def distribute_evenly_between_ds(items, budget):
    remained_budget = budget
    result = {}
    for name, count in sorted(items, key = lambda x: x[1]):
        result[name] = floor(min(count, remained_budget/(len(items)-len(result))))
        remained_budget -= result[name]
    return result.items()

def calculate_sampling(dt, minimum, distribute_evenly):
    # dt is counts per dataset
    total_per_class = dt.sum(axis=1).to_dict()
    for class_name, total in total_per_class.items():
        if total <= minimum:
            continue  # nothing to do

        # TODO distribute evenly between datasets
        items = list(dt.loc[class_name].dropna().items())

        if distribute_evenly:
            items = distribute_evenly_between_ds(items, minimum)
        else:
            items = [(dataset_name, count_in_dataset * (minimum/total)) for dataset_name, count_in_dataset in items]

        for dataset_name, value in items:
            dt.at[class_name, dataset_name] = value

    return dt

def read_json(filename):
    with open(filename) as f:
        return json.load(f)


def run_calculate_sampling(dataset_names, input_dirs, output_dir, version, limit_class_count, distribute_evenly):

    os.makedirs(output_dir, exist_ok=True)
    names = dataset_names.split(',')
    dirs = input_dirs.split(',')
    name2json = {name: read_json(os.path.join(input_dir, 'metadata.json'))
                 for name, input_dir in zip(names, dirs)}

    
    for split in SPLIT_NAMES:
        # merge sizes
        name2len = {name: js['sizes'][split] for name,js in name2json.items()}
        file = os.path.join(output_dir, f'sizes_{split}{SUFFIX}')
        pd.DataFrame(name2len, index=['size']).to_csv(file)
        print(f'Wrote {file}')

        # merge classes
        name2class_stat = {name: js['counters'][split] for name, js in name2json.items()}
        df = pd.concat([pd.DataFrame(c, index=[name]) for name, c in name2class_stat.items()]).T
        
        file = os.path.join(output_dir, f'classes_{split}{SUFFIX}')
        df.to_csv(file)
        print(f'Wrote {file}')

        if limit_class_count and split == 'train' :
            df = calculate_sampling(df, limit_class_count, distribute_evenly)
        else:
            pass # no need to calculate sampling
        file = os.path.join(output_dir, f'sampling_{split}{SUFFIX}')
        df.fillna(0).to_csv(file)
        print(f'Wrote {file}')

def main():
    args = parser.parse_args()
    utils.set_seed(args.seed)
    run_calculate_sampling(dataset_names=args.dataset_names, input_dirs=args.input_dirs, output_dir=args.output_dir, version=args.version, limit_class_count=args.limit_class_count, distribute_evenly=args.distribute_evenly)

if __name__ == '__main__':
    main()