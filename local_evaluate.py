
from functools import lru_cache
import pandas as pd
import argparse
import os
from sklearn.metrics import classification_report
import numpy as np
import ast
import sklearn
import json
import utils
from collections import Counter, defaultdict


EVALUATE_VERSION = "2.6"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
parser.add_argument('--input_file', type=str, required=True, help='Comma separated input files')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--version', type=str, required=True, help='version to update on change of code')
parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for multilabel evaluation')
parser.add_argument("--seed", type=int, default=42)

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = ['all_results.json']
__ARGPARSER__ = parser


@lru_cache
def get_labels_in_binary_form(labels, class_names):
    labels = set(labels) # looking in a set is more efficient that looking in a list

    return [1 * (cls in labels) for cls in class_names]


def get_scores_in_binary_form(predicted_labels, predicted_scores, class_names):
    label_to_score = {x:y for x,y in zip(predicted_labels, predicted_scores)}
    labels = set(predicted_labels) # looking in a set is more efficient that looking in a list
    return [label_to_score[cls] for cls in class_names]

def get_predictions_over_threshold(predictions, threshold):
    return [p for p, score in zip(predictions['labels'], predictions['scores']) if score > threshold] 


def get_metric(binary_labels, binary_preds, suffix=''):
    return {
        f'accuracy{suffix}': sklearn.metrics.accuracy_score(binary_labels, binary_preds),
        f'micro_f1{suffix}': sklearn.metrics.f1_score(binary_labels, binary_preds, average='micro'),
        f'macro_f1{suffix}': sklearn.metrics.f1_score(binary_labels, binary_preds, average='macro'),
        f'weighted_f1{suffix}': sklearn.metrics.f1_score(binary_labels, binary_preds, average='weighted'),
        f'f1_per_class{suffix}': sklearn.metrics.f1_score(binary_labels, binary_preds, average=None),
    }

def get_multilabel_metrics(dt, multilabel, threshold, class_names):
    actual_class_names = list(set([l for ll in dt.label for l in ll]))
    actual_binary_labels = [get_labels_in_binary_form(tuple(labels), tuple(actual_class_names)) for labels in dt.label]

    actual_multilabel_binary_scores = [get_scores_in_binary_form(predictions['labels'], predictions['scores'],tuple(actual_class_names))
                   for predictions in dt.prediction]

    tuple_class_names = tuple(class_names)
    binary_labels = [get_labels_in_binary_form(tuple(labels), tuple_class_names) for labels in dt.label]
        
    binary_preds = [get_labels_in_binary_form(tuple([predictions['labels'][0]]), tuple_class_names) for predictions in dt.prediction]
    
    multilabel_binary_preds = [get_labels_in_binary_form(tuple(get_predictions_over_threshold(predictions, threshold)), tuple_class_names)
                   for predictions in dt.prediction]

    auc_multilabel = sklearn.metrics.roc_auc_score(
                actual_binary_labels,
                actual_multilabel_binary_scores,
                average='macro'
            ) 


    return {
        **get_metric(binary_labels, binary_preds, suffix=''),
        **get_metric(binary_labels, multilabel_binary_preds, suffix='_multilabel'),
        **{
          'evaluation_size': len(dt),
          'total_classes': len(class_names),
          "auc_multilabel": auc_multilabel
        }
    }


def main():
    args = parser.parse_args()
    utils.set_seed(args.seed)
    run_evaluation(dataset_name=args.dataset_name, input_dir=args.input_dir, input_file=args.input_file, output_dir=args.output_dir, version=args.version, threshold=args.threshold)

def run_evaluation(dataset_name, input_dir, input_file, output_dir, version, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(input_dir, 'metadata.json')) as f:
        metadata = json.load(f)

    df = pd.read_csv(os.path.join(input_dir, input_file), converters={
        'label': ast.literal_eval, 'prediction': ast.literal_eval
    })

    labels_in_dataset = set([l for ll in df['label'] for l in ll])

    labels = sorted([l for l in metadata['labels']] ) # do not remember why we needed this "if l in labels_in_dataset]")
    metrics = get_multilabel_metrics(df, metadata['multilabel'], threshold, labels)
    metrics['labels'] = labels    

    result = pd.Series(metrics)
    result.to_json(os.path.join(output_dir, 'all_results.json'), indent=4)

if __name__ == '__main__':
    main()
