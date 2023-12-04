import json

import pandas as pd
import argparse
import os
import ast
import utils
import random

PREPARE_DATA_FOR_TRAIN_T5_VERSION = "1.0"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument('--example_generation_method', type=str, 
                    default='random', choices=['random','filter_multi_label', 'mixed'],
                    help='method to use for generating examples, with focus on negative once')


parser.add_argument('--version', type=str, required=True, help='version to update on change of code')

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = ['train.csv', 'dev.csv']
__ARGPARSER__ = parser


def get_negative_examples(text, neg_classes, n_neg_examples, negative_label):
    neg_classes = neg_classes[:n_neg_examples]
    return [(text, neg_class, negative_label) for neg_class in neg_classes]


def get_score(cls, predictions):
    labels_to_score = {label: score for label, score in zip(predictions["labels"], predictions["scores"])}
    return labels_to_score[cls]


# TODO: add support for multi-label examples
def convert_df(df, split, dataset_name, example_generation_method):
    list_of_tuples = []
    for txt, labels in zip(df['text'].tolist(), df['labels_to_use'].tolist()):
        if example_generation_method == 'filter_multi_label':
            if len(labels) == 0:
                continue
            if len(labels) > 1:
                labels = labels[:1]
            list_of_tuples.append((txt, "mc", ", ".join(labels), dataset_name))
        else:
            list_of_tuples.append((txt, "mc", ", ".join(labels), dataset_name))

    print(f"{split}: created {len(list_of_tuples)} examples")

    result_df = pd.DataFrame(list_of_tuples, columns=['text', 'class', 'label', 'dataset_name'])
    return result_df


def add_pairwise_examples(all_classes, dataset_name, label, labels, list_of_tuples, txt):
    list_of_tuples.append((txt, label, "yes", dataset_name))
    neg_class = random.choice([c for c in all_classes if c not in labels])
    list_of_tuples.append((txt, neg_class, "no", dataset_name))


def convert_dataset(base_dir, dataset_name, example_generation_method):
    splits = {}
    for split in ['train', 'dev']:
        df = pd.read_csv(os.path.join(base_dir, f'{split}.csv'),
                         converters={'label': ast.literal_eval, 'labels_to_use': ast.literal_eval,
                                     'prediction': ast.literal_eval})

        splits[split] = convert_df(df, split, dataset_name, example_generation_method)

    return splits


def main():
    args = parser.parse_args()
    utils.set_seed(args.seed)
    run_prepare_data_for_train_t5(dataset_name=args.dataset_name, input_dir=args.input_dir, output_dir=args.output_dir, example_generation_method=args.example_generation_method)


def run_prepare_data_for_train_t5(dataset_name, input_dir, output_dir, example_generation_method='random'):
    if os.path.exists(os.path.join(output_dir, 'dev.csv')): # let's use the previous results
        return   
     
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, "metadata.json")) as f:
        metadata = json.load(f)

    dataset = convert_dataset(input_dir, dataset_name, example_generation_method)
    for split, df in dataset.items():
        file = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(file, index=False)
        print(f'Wrote {file}')

    with (open(os.path.join(output_dir, "metadata.json"), "tw")) as f:
        json.dump(metadata, f)

    df.to_csv(file, index=False)


if __name__ == '__main__':
    main()
