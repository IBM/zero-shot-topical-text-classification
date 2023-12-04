import pandas as pd
import argparse
import os
import logging
import json
import torch
import time
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, pipeline)
from transformers.pipelines.pt_utils import KeyDataset
import ast
import utils



PREDICT_VERSION = "1.5"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--model_dir', type=str, required=True, help='model to be used for the predictions')
parser.add_argument("--fp16", action='store_true', help='Should we use fp16 for prediction?')
parser.add_argument('--input_dir', type=str, required=True, help='Input dir')
parser.add_argument('--input_files', type=str, required=True, help='Comma separated Input files')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_seq_length', type=int, default=256)
parser.add_argument('--version', type=str, required=True, help='version to update on change of code')
parser.add_argument('--training_method', type=str, default='pairwise', choices=['pairwise', 'multiclass'],
    help='What approached has been used in training')
parser.add_argument("--hypothesis_template", required=True, default='This text is about {}.')
parser.add_argument("--domain_aware", action='store_true')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument('--sample_size', type=int, help='Sample the data', default=None)

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = []
__ARGPARSER__ = parser

def predict(model_dir, texts_to_infer, label_names, batch_size, max_length, 
        hypothesis_template, training_method, fp16, multilabel:bool, dataset_name, output_dir):

    print(f"Batch size {batch_size}")
    device = 0 if torch.cuda.is_available() else -1

    # We initialize the tokenizer here in order to set the maximum sequence length
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, model_max_length=max_length)
    
    ds = Dataset.from_dict({'text': texts_to_infer})

    preds_list = []
    if training_method == 'pairwise':
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if fp16:
            model = model.half()
        classifier = pipeline("zero-shot-classification", hypothesis_template=hypothesis_template,
                            model=model, tokenizer=tokenizer, device=device)
        start = time.time()
        for text, output in tqdm(zip(texts_to_infer, classifier(KeyDataset(ds, 'text'),
                                                                batch_size=batch_size,
                                                                candidate_labels=label_names, multi_label=multilabel)),
                                    total=len(ds), desc="zero-shot inference"):
            preds_list.append(output)
        end = time.time()        
    else:
        raise Exception(f" Enexpected training method {training_method}")
    
    runtime = [{"dataset_name":dataset_name, "number of texts":len(texts_to_infer), "runtime":end-start,
                    "fp16":fp16,"batch_size":batch_size,"max_seq_length":max_length,"is_multilabel":multilabel,
                    "model_dir":model_dir, "num labels":len(label_names),"hypothesis_template":hypothesis_template,"training_method":training_method}]
    pd.DataFrame(runtime).to_csv(os.path.join(output_dir,"runtime.csv"))
    

    return preds_list


def predict_file(model_dir, input_file, labels, batch_size, max_seq_length, 
        hypothesis_template, training_method, is_domain_aware, fp16, multilabel:bool,dataset_name, output_dir):
    df = pd.read_csv(input_file, converters={'label': ast.literal_eval}).dropna() # TODO do it on vcard base
    df.sort_values('text', key=lambda x: x.str.len(), inplace=True) # for quicker interference
    if is_domain_aware:
        filename = os.path.join(os.path.dirname(input_file), 'domain.csv')
        with open(filename,'r') as f:
            domain = f.readline()   

        hypothesis_template = hypothesis_template.replace('text',domain)

    df['prediction'] = predict(model_dir, df['text'], labels, batch_size, max_seq_length, 
        hypothesis_template, training_method, fp16, multilabel=multilabel,dataset_name=dataset_name, output_dir=output_dir)
    return df


def run_predict(dataset_name, model_dir, fp16, input_dir, input_files, output_dir,hypothesis_template,  
                domain_aware=False, sample_size=None, batch_size=64, max_seq_length=256, training_method="pairwise"):

    if os.path.exists(os.path.join(output_dir, input_files.split(',')[-1])): # let's use the previous results
        return
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(input_dir, 'metadata.json')) as f:
        metadata = json.load(f)
    labels = metadata['labels']
    template = hypothesis_template 

    print('template', template, len(template))  # WORDAROND for arguments with space
    if template[0] == "'":
        template = template[1:-1]
    print('new template', template, len(template))

    # evaluate will need metadata too
    with (open(os.path.join(output_dir, 'metadata.json'), 'tw')) as f:
        json.dump(metadata, f)

    for input_filename in input_files.split(','):
        input_file = os.path.join(input_dir, input_filename)
        df = predict_file(model_dir, input_file, labels, batch_size, 
            max_seq_length, template, training_method, domain_aware, fp16,
            True,output_dir=output_dir, dataset_name=dataset_name)
        if sample_size:
            df = df.sample(sample_size, random_state=seed)
        output_file = os.path.join(output_dir, input_filename)
        df.to_csv(output_file, index=False)
        logging.info(f'Wrote {output_file}')


def main():
    args = parser.parse_args()
    utils.set_seed(args.seed)
    utils.set_torch_seed(args.seed)
    run_predict(dataset_name=args.dataset_name, model_dir=args.model_dir, fp16=args.fp16, input_dir=args.input_dir, input_files=args.input_files, output_dir=args.output_dir, batch_size=args.batch_size, max_seq_length=args.max_seq_length,
                training_method=args.training_method, hypothesis_template=args.hypothesis_template, domain_aware=args.domain_aware, sample_size=args.sample_size)


if __name__ == '__main__':
    main()
