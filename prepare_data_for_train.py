import json

import pandas as pd
import argparse
import os
import ast
import utils
import random


PREPARE_DATA_FOR_TRAIN_VERSION="1.6"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--balance_factor', type=int, default=3,
                    help='how many negative examples generate per positive one')
parser.add_argument("--seed", type=int, default=42)

parser.add_argument('--version', type=str, required=True, help='version to update on change of code')
parser.add_argument('--example_generation_method', type=str, required=True, choices=['random','word_overlap', 'sbert', 'zero_shot_sampling', "random_by_class"],
                    help='method to use for generating examples, with focus on negative once')

parser.add_argument('--positive_label', type=str, default='ENTAILMENT', help='label of positives')
parser.add_argument('--negative_label', type=str, default='CONTRADICTION', help='label of positives')

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = ['train.csv', 'dev.csv']
__ARGPARSER__ = parser


def get_negative_examples(text, neg_classes, n_neg_examples, negative_label):
    neg_classes = neg_classes[:n_neg_examples]
    return [(text, neg_class, negative_label) for neg_class in neg_classes]


def get_score(cls, predictions):
    labels_to_score = {label: score for label,score in zip(predictions["labels"], predictions["scores"])}
    return labels_to_score[cls]


def get_negative_classes_by_hn_policy(method, all_class_names, texts, labels, data_name_for_sb=None, data_split_for_sb=None):
    # random permutation of neg topics per example
    all_class_names=list(all_class_names)
    neg_topics=[]

    not_empty_hard_neg_count = 0
    for i in range(len(texts)):
        neg_cls = [c for c in all_class_names if c not in labels[i]]
        neg_topics.append(random.sample(neg_cls,len(neg_cls)))
    print(f"total {not_empty_hard_neg_count} / {len(texts)}")
    return neg_topics


def convert_df(df, example_generation_method, balance_factor, dataset_name, split, positive_label, negative_label):
    
    list_of_tuples = []
    all_classes = set(l for ll in df.label for l in ll)
    if example_generation_method == "random":
        neg_classes = get_negative_classes_by_hn_policy(example_generation_method, all_classes, 
            df['text'], df['label'], 
            data_name_for_sb=dataset_name, data_split_for_sb=split)
        
        print(f"{split}: started with {len(df)} examples")
        for ind, row in df.iterrows():
            list_of_tuples.extend([(row['text'], l, positive_label) for l in row['labels_to_use']])
            list_of_tuples.extend(get_negative_examples(row['text'], neg_classes[ind], 
                n_neg_examples=balance_factor * len(row['labels_to_use']), negative_label=negative_label))
        print(f"{split}: created {len(list_of_tuples)} examples (both positive and negative)")
    else:
        raise Exception("Unexpected example_generation_method:" + example_generation_method)

    result_df = pd.DataFrame(list_of_tuples, columns=['text', 'class', 'label'])
    result_df['domain']='general'

    result_df['dataset_name'] = dataset_name

    return result_df


def convert_dataset(dataset_name, base_dir, balance_factor, example_generation_method, positive_label, negative_label):
    splits = {}
    for split in ['train', 'dev']:
        df = pd.read_csv(os.path.join(base_dir, f'{split}.csv'), 
            converters={'label': ast.literal_eval, 'labels_to_use': ast.literal_eval, 'prediction': ast.literal_eval})
    
        # TODO bring logic to convert dataframe to CONTRADICT/ENATILMENT format
        splits[split] = convert_df(df, example_generation_method, balance_factor, dataset_name, split, positive_label, negative_label)

    return splits


def main():
    args = parser.parse_args()
    utils.set_seed(args.seed)
    run_prepare_data_for_train(dataset_name=args.dataset_name, input_dir=args.input_dir, output_dir=args.output_dir, balance_factor=args.balance_factor, example_generation_method=args.example_generation_method, positive_label=args.positive_label, negative_label=args.negative_label)


def run_prepare_data_for_train(dataset_name, input_dir, output_dir, example_generation_method='random', balance_factor=1, positive_label='ENTAILMENT', negative_label='CONTRADICTION'):    
    if os.path.exists(os.path.join(output_dir, 'dev.csv')): # let's use the previous results
        return   
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, "metadata.json")) as f:
        metadata = json.load(f)
    with (open(os.path.join(output_dir, "metadata.json"), "tw")) as f:
        json.dump(metadata, f)

    dataset = convert_dataset(dataset_name, input_dir, balance_factor,example_generation_method, positive_label, negative_label)
    for split, df in dataset.items():
        file = os.path.join(output_dir, f'{split}.csv')
        df.to_csv(file, index=False)
        print(f'Wrote {file}')

    df.to_csv(file, index=False)


if __name__ == '__main__':
    main()
