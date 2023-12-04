from collections import Counter

import pandas as pd
import argparse
import os
import ast
import utils
import json

SAMPLE_DATA_VERSION = "1.1"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--sampling_dir', type=str, required=True)
parser.add_argument('--limit_class_count', type=int, required=True, help='limits_class_count')
parser.add_argument("--seed", type=int, default=42)

parser.add_argument('--version', type=str, required=True, help='version to update on change of code')

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = ['train.csv', 'dev.csv']
__ARGPARSER__ = parser


def convert_df(df, sampling_column,  seed):
    df = df.sample(frac=1, random_state=seed) 
    counter = Counter() # how many times we saw this class already
    labels_to_use = []

    for _, row in df.iterrows():
        allowed_labels = [l for l in row['label'] if counter[l] < sampling_column.loc[l].iloc[0]]
        labels_to_use.append(allowed_labels)
        counter.update(allowed_labels)
        
    df['labels_to_use'] = labels_to_use
    # TODO remove rows where labels to use is empty
    return df


def convert_dataset(dataset_name, base_dir, sampling_dir, seed=42):
    splits = {}
    for split in ['train', 'dev']:
        df = pd.read_csv(os.path.join(base_dir, f'{split}.csv'), converters={'label': ast.literal_eval})
        sampling_df = pd.read_csv(os.path.join(sampling_dir, f'sampling_{split}_stat.csv'), index_col=0)
        splits[split] = convert_df(df, sampling_df[[dataset_name]], seed)
    return splits


def run_sample_data(dataset_name, input_dir, output_dir, sampling_dir, limit_class_count, version):
        
    os.makedirs(output_dir, exist_ok=True)

    dataset = convert_dataset(dataset_name, input_dir, sampling_dir)
    for split, df in dataset.items():
        file = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(file, index=False)
        print(f'Wrote {file}')

    with open(os.path.join(input_dir, "metadata.json")) as f:
        metadata = json.load(f)
    with (open(os.path.join(output_dir, "metadata.json"), "tw")) as f:
        json.dump(metadata, f)


def main():
    args = parser.parse_args()
    utils.set_seed(args.seed)
    run_sample_data(dataset_name=args.dataset_name, input_dir=args.input_dir, output_dir=args.output_dir,
                    sampling_dir=args.sampling_dir, limit_class_count=args.limit_class_count, version=args.version)


if __name__ == '__main__':
    main()
