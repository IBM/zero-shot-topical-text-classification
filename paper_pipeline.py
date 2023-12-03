import argparse
import logging
import sys
from typing import NamedTuple
import os

import numpy as np
import pandas as pd
import json

from utils import set_seed

from preprocess import run_preprocess
from calculate_sampling import CALCULATE_SAMPLING_VERSION, run_calculate_sampling
from sample_data import SAMPLE_DATA_VERSION, run_sample_data
from prepare_data_for_train import run_prepare_data_for_train
from prepare_data_for_train_t5 import run_prepare_data_for_train_t5
from train import run_train
from predict import run_predict
from predict_with_generative_models import run_predict_with_generative_models
from joint import run_joint
from local_evaluate import EVALUATE_VERSION, run_evaluation

LIMIT_CLASS_COUNT = 100
DEBERTA_HYPOTHESIS_TEMPLATE = "'This text is about {}.'"
MULTI_CLASS_FLAN_PROMPT_TYPE = 'custom_option_multi_class_custom_option_multi_class'

folds = [["reuters21578", 'claim_stance_topic',
          'unfair_tos',
          'head_qa',
          'banking77',
          'ag_news',
          'yahoo_answers_topics'],
         ['argument_topic',
          'cuad',
          'dbpedia_14',
          'news_category_classification_headline',
          'law_stack_exchange',
          'massive'],
         ['clinc150',
          '20_newsgroups',
          'contract_nli',
          'ledgar',
          'medical_abstracts',
          'financial_tweets',
          ]]

multilabel_datasets = ['contract_nli', 'cuad', 'reuters21578', 'unfair_tos']

train_seeds = [42, 40, 38]

def get_preprocess_path(out_dir, ds):
    return os.path.join(out_dir, "preprocess", ds)


def get_calculating_sample_path(out_dir, fold_number):
    return os.path.join(out_dir, "calculate_sampling", str(fold_number))


def get_sample_data_path(out_dir, ds):
    return os.path.join(out_dir, "sample_data", ds)


def get_prepare_data_for_train_path(out_dir, flow, ds):
    return os.path.join(out_dir,  flow,  'prepare_data_for_train', ds)


def get_joint_path(out_dir, flow, fold_number):
    return os.path.join(out_dir, flow, 'join', str(fold_number))


def get_train_path(out_dir, flow, fold_number, seed):
    return os.path.join(out_dir,   flow, "train", str(fold_number), str(seed))


def get_predict_path(out_dir, flow, ds, seed):
    return os.path.join(out_dir, flow, "predict", ds, str(seed))


def get_evaluate_path(out_dir, flow, ds, seed):
    return os.path.join(out_dir, flow, "evaluate", ds, str(seed))


def get_aggregate_path(out_dir, flow, fold_number, seed):
    return os.path.join(out_dir, flow, 'aggregate', str(fold_number), str(seed))


def calculate_sampling(out_dir, fold_number, dss):
    logging.info(f"calculate_sampling fold number {fold_number}")
    input_dirs = ','.join(get_preprocess_path(out_dir, ds) for ds in dss)
    output_dir = get_calculating_sample_path(out_dir, fold_number)

    run_calculate_sampling(dataset_names=','.join(dss),
                           input_dirs=input_dirs,
                           output_dir=output_dir,
                           version=CALCULATE_SAMPLING_VERSION,
                           limit_class_count=LIMIT_CLASS_COUNT,
                           distribute_evenly=False)
    return output_dir


def sample_data(out_dir, sampling_dir, ds):
    logging.info(f"sampling data from {ds}")
    run_sample_data(dataset_name=ds,
                    input_dir=get_preprocess_path(out_dir, ds),
                    output_dir=get_sample_data_path(out_dir, ds),
                    sampling_dir=sampling_dir,
                    limit_class_count=LIMIT_CLASS_COUNT,
                    version=SAMPLE_DATA_VERSION)


def prepare_data_for_train(out_dir, flow, ds):
    logging.info(f"prepare data for train from {ds}")
    shared_args = {
        "dataset_name": ds,
        "input_dir": get_sample_data_path(out_dir, ds),
        "output_dir": get_prepare_data_for_train_path(out_dir, flow, ds),
    }
    if flow == 'flan':
        run_prepare_data_for_train_t5(
            example_generation_method='filter_multi_label',
            **shared_args)
    elif flow == 'deberta':
        run_prepare_data_for_train(
            example_generation_method='random',
            **shared_args
        )
    else:
        raise Exception(f"base model {flow} is not supported")


def join(out_dir, flow, fold_number, train_datasets):
    input_dirs = ','.join(get_prepare_data_for_train_path(out_dir,  flow, ds) for ds in train_datasets)
    run_joint(input_dirs=input_dirs, output_dir=get_joint_path(out_dir, flow, str(fold_number)))


def train(out_dir, flow, fold_number, seed):
    logging.info(f"starting train for fold number {fold_number}")

    shared_train_args = {
        "dev_df_name": "dev.csv",
        "input_dir": get_joint_path(out_dir, flow, fold_number),
        "num_epochs": 3,
        "train_df_name": "train.csv",
        "seed": seed,
        "output_dir": get_train_path(out_dir, flow, fold_number, seed),

    }
    if flow == 'flan':
        run_train(
            base_model='google/flan-t5-xxl',
            fp16=False,
            gradient_accumulation_steps=16,
            hypothesis_template=MULTI_CLASS_FLAN_PROMPT_TYPE,
            learning_rate=3e-5,
            max_seq_length=512,
            batch_size=1,
            training_method="t5",
            save_steps=500,
            evaluation_and_save_strategy="steps",
            use_lora=True,
            select_best_model=True,
            stop_early=True,
            **shared_train_args,
        )
    elif flow == 'deberta':
        run_train(base_model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                  fp16=True,  # enables fb16
                  gradient_accumulation_steps=1,
                  hypothesis_template=DEBERTA_HYPOTHESIS_TEMPLATE,
                  learning_rate=5e-06,
                  max_seq_length=256,
                  batch_size=32,
                  training_method="pairwise",
                  evaluation_and_save_strategy="epoch",
                  **shared_train_args)
    else:
        raise Exception(f"base model {flow} is not supported")


def predict(out_dir, flow, fold_number, seed, ds):
    logging.info(f"starting predict for fold number {fold_number} dataset {ds}, seed {seed}")
    shared_prediction_args = {
        "input_dir": get_preprocess_path(out_dir, ds),
        "input_files": "test.csv",
        "model_dir": get_train_path(out_dir, flow, fold_number, seed),
        "output_dir": get_predict_path(out_dir, flow, ds, seed),
        "dataset_name": ds,
    }
    if flow == 'flan':
        run_predict_with_generative_models(
            use_logits_processor=True,
            use_lora=True,
            max_seq_length=2048,
            prompt=MULTI_CLASS_FLAN_PROMPT_TYPE,
            batch_size=4 if ds in ('yahoo_answers', 'dbpedia') else 8,
            fp16=False,
            **shared_prediction_args
        )
    elif flow == 'deberta':
        run_predict(
            batch_size=128,
            fp16=True,
            hypothesis_template=DEBERTA_HYPOTHESIS_TEMPLATE,
            **shared_prediction_args
        )
    else:
        raise Exception(f"base model {flow} is not supported")


def evaluate(out_dir, flow, seed, ds):
    logging.info(f"running evaluation for dataset {ds}, seed {seed}")
    
    run_evaluation(dataset_name=ds,
                   input_dir=get_predict_path(out_dir, flow, ds, seed),
                   input_file='test.csv',
                   output_dir=get_evaluate_path(out_dir, flow, ds, seed),
                   version=EVALUATE_VERSION)


def aggregate(out_dir, flow, dss, fold_number, seed):
    logging.info(f"aggregating data for datasets {dss}, seed {seed}")
    
    def read_json(ds, seed):
        with open(os.path.join(get_evaluate_path(out_dir, flow, ds, seed), "all_results.json")) as f:
            return json.load(f)
    result_dir = get_aggregate_path(out_dir, flow, fold_number, seed)
    os.makedirs(result_dir, exist_ok=True)
    ds2jsons = {ds: [read_json(ds, seed)] for ds in dss}
    multiclass_datasets = [ds for ds in dss if ds not in set(multilabel_datasets)]

    for relevant_datasets, suffix in zip([multiclass_datasets, multilabel_datasets],
                                         ['', '_multilabel']):
        relevant_datasets = [ds for ds in relevant_datasets if ds in dss]
        for m in ['macro_f1', 'accuracy', 'micro_f1']:
            metric = f'{m}{suffix}'
            if metric == 'macro_f1_multilabel':
                metric = 'auc_multilabel'
            ds2mn = {ds: np.mean([js[metric] for js in ds2jsons[ds]]) for ds in relevant_datasets}
            ser = (pd.Series(ds2mn) * 100.).round(2)
            ser.index.name = "dataset"
            ser.name = metric
            ser.to_csv(os.path.join(result_dir, f'{metric}.csv'))
            print(f"{metric}: {ser.mean():.2f}")
        print()


parser = argparse.ArgumentParser()
parser.add_argument('--debug_mode', action='store_true', help='Run in debug mode')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
parser.add_argument('--fold_number', type=int, default=0,  choices=[0,1,2] ,help='Which fold to use for evaluation')
parser.add_argument('--flow', type=str, default='deberta', choices=['flan', 'deberta'],
                    help='which model to use: deberta or flan')


def main():
    args = parser.parse_args()
    out_dir = args.output_dir
    fold_number = args.fold_number
    flow = args.flow
    seed = args.seed
    flow = args.flow
    
    args_as_string = f'arguments:flow {flow}, fold_number {fold_number}, seed {seed}, debug_mode {args.debug_mode}'
    logging.info(args_as_string)
    print(args_as_string)

    set_seed(seed)

    if args.debug_mode:
        out_dir = 'debug_output'
        global folds
        folds = [folds[0][0:2], folds[1][0:1]]

    datasets = [ds for fold in folds for ds in fold]

    # TODO extract as separate process
    for ds in datasets:
        run_preprocess(output_dir=get_preprocess_path(out_dir, ds), dataset_name=ds)

    train_datasets = [d for j, l in enumerate(folds) for d in l if j != fold_number]
    sample_dir = calculate_sampling(out_dir, fold_number, train_datasets)
    for ds in train_datasets:
        sample_data(out_dir, sample_dir, ds)
        prepare_data_for_train(out_dir, flow, ds)
    join(out_dir, flow, fold_number, train_datasets)

    train(out_dir, flow, fold_number, seed)

    for ds in folds[fold_number]:
        predict(out_dir, flow, fold_number, seed, ds)
        evaluate(out_dir, flow, seed, ds)
    # TODO make a separate script to aggregate the results.
    aggregate(out_dir, flow, folds[fold_number], fold_number, seed)


if __name__ == '__main__':
    logging.basicConfig(encoding='utf-8', level=logging.DEBUG)
    main()
