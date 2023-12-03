import pandas as pd
import argparse
import os
import logging
import ast
import json
from collections import Counter
import os
from datasets import load_dataset
from config import ds_to_type # TODO do we need it?

import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Dataset name')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')


def clean_text(text):
    return text.replace("\n", " ").replace("\r", "").replace("  ", " ").replace("–", "-").replace(""", "\"").replace(""", "\"")

def build_row(entry):
    result = {k:v for k,v in zip(entry['key'], entry['value'])}
    result['text'] = clean_text(result['text'])
    if 'labels' in result: # convention of unitxt to use labels key for multilabel, while our code expects label
        result['label'] = result['labels']
        del result['labels']
    
    result['label'] = [l.lower() for l in ast.literal_eval(result['label'])]
    
    return result


def read_dataset(dataset_name):

    # TODO import to install
    try:
        ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
    except: # let's try import again
        importlib.import_module(f'cards.{dataset_name}', package=None)
        ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
    
    splits = {split: pd.DataFrame(build_row(entry) for entry in ds[split]['additional_inputs']) 
              for split in ds.keys()}

    return splits

            

def run_preprocess(output_dir, dataset_name):
    metadata_filename = os.path.join(output_dir, 'metadata.json')
    
    if os.path.exists(metadata_filename): # let's use the previous version
        return
    os.makedirs(output_dir, exist_ok=True)
    dataset = read_dataset(dataset_name)
    out_dir = output_dir
    os.makedirs(out_dir, exist_ok=True)
    labels = set()
    multilabel = False
    multilabel_counters = {}
    counters = {}
    sizes = {}

    for split, df in dataset.items():
       
        file = os.path.join(out_dir, f'{split}.csv')
        if split == 'train':
            df = df.iloc[df.astype(str).drop_duplicates().index].copy()
        df['labels_to_use'] = df.label

        sizes[split] = len(df)

        c = Counter(len(ll) for ll in df.label)
        multilabel |= not (len(c) == 1 and 1 in c)
        multilabel_counters[split] = c
        
        labels |= set([l for ll in df.label for l in ll])
        counters[split] = Counter([l for ll in df.label for l in ll])
        logging.info(f'Wrote {file}')
        df = df[df['text'].str.len() > 0] # let's remove empty texts
        df.dropna().to_csv(file, index=False)



    with (open(metadata_filename, 'tw')) as f:
        json.dump(
            {'name': dataset_name,
             'multilabel': multilabel,
             'type': ds_to_type[dataset_name],
             'labels': list(labels), 
             'sizes': sizes, 
             'counters': counters,
             'multilabel_counters':multilabel_counters},
            f, indent=4)
    logging.info(f'Wrote metadata')


def main():
    args = parser.parse_args()
    run_preprocess(output_dir=args.output_dir, dataset_name=args.dataset_name)

if __name__ == '__main__':
    main()
