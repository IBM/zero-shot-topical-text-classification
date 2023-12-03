import os
from tempfile import TemporaryDirectory
from datasets import load_dataset
from unitxt import add_to_catalog
from datasets import load_dataset as hf_load_dataset, DatasetDict
from unitxt.stream import MultiStream
from typing import Sequence
import requests
import json
import re
import pandas as pd
import zipfile
import ast

from unitxt.blocks import (
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues
)

from unitxt.loaders import Loader

contract_nli_labels_dict = {
    "nda-11":
        "no reverse engineering",
    "nda-16": "return of confidential information",
    "nda-15": "no licensing",
    "nda-10": "confidentiality of agreement",
    "nda-2": "none-inclusion of non-technical information",
    "nda-1": "explicit identification",
    "nda-19": "survival of obligations",
    "nda-12": "permissible development of similar information",
    "nda-20": "permissible post-agreement possession",
    "nda-3": "inclusion of verbally conveyed information",
    "nda-18": "no solicitation",
    "nda-7": "sharing with third-parties",
    "nda-17": "permissible copy",
    "nda-8": "notice on compelled disclosure",
    "nda-13": "permissible acquirement of similar information",
    "nda-5": "sharing with employees",
    "nda-4": "limited use",
}


class LoadZipFromWeb(Loader):
    path: str
    data_files: Sequence[str]

    def _download_from_web(self, url, local_file):
        print(f"Downloading {url}")

        try:
            response = requests.get(url, allow_redirects=True)
        except Exception as e:
            raise Exception(f"Unabled to download {url} in {local_file}", e)
        if response.status_code != 200:
            raise Exception(f'request to {url} returns unexpected http code {response.status_code}')
        with open(local_file, 'wb') as f:
            f.write(response.content)

        print("\nDownload Successful")

    def process(self):
        def process_contract_nli(in_dir):
            
            def clean_text(t):
                t = t.replace("\n", " ").replace("  ", " ").replace("    ", " ").replace("&lt;", "<").replace("“", "\"").replace("”", "\"")
                t = re.sub(r"\s+", " ", t)
                return t

            print("processing contract_nli")
            ds_dict = DatasetDict()
            out_dir = "data/contract_nli_paper/"
            os.makedirs(out_dir, exist_ok=True)

            for data_split in ['train', 'dev', 'test']:
                with open(os.path.join(in_dir, f'{data_split}.json')) as in_file:
                    data = json.load(in_file)
                examples_for_set = {}
                for document in data['documents']:
                    annotations = document['annotation_sets'][0]['annotations']
                    text = document['text']
                    span_texts = [text[span[0]:span[1]] for span in document['spans']]
                    doc_examples = [(span_texts[span_loc], ann_key)
                                    for ann_key, ann_val in annotations.items() if 'Entailment'
                                    in ann_val['choice'] for span_loc in ann_val['spans'] if
                                    not span_texts[span_loc].endswith(':')]
                    span_texts = [st for st in span_texts if not st.endswith(':')]
                    text_to_classes = {x: [] for x in span_texts}
                    for text, label in doc_examples:
                        text_to_classes[text].append(label)
                    examples_for_set = {**examples_for_set, **text_to_classes}
                examples_for_set_as_list = list(examples_for_set.items())
                examples_for_set_as_list = [(clean_text(x[0]), [contract_nli_labels_dict[l] for l in x[1]]) for x in
                                            examples_for_set_as_list]
                examples_for_set_as_list = [x for x in examples_for_set_as_list if len(x[1]) > 0 and len(x[0].split()) > 4]
                split = data_split
                # .replace('dev', 'validation') # TODO remove double renaming
                out_file = os.path.join(out_dir, f'{split}.csv')
                ds_dict[split] = pd.DataFrame()
                ds_dict[split]['labels'] = [x[1] for x in examples_for_set_as_list]
                ds_dict[split]['text'] = [x[0] for x in examples_for_set_as_list]
                ds_dict[split].to_csv(out_file, encoding='utf-8', index=False)
            return out_dir

        with TemporaryDirectory() as download_directory:
            for data_file in self.data_files:
                self._download_from_web(
                     self.path + "/" + data_file, download_directory + "/" + data_file
                )
            with TemporaryDirectory() as extract_directory:
                with zipfile.ZipFile(download_directory + "/" + data_file) as zf:
                    zf.extractall(extract_directory)
                processed_dir = process_contract_nli(extract_directory + "/" + "contract-nli")
                dataset = hf_load_dataset(processed_dir, data_files={'train':'train.csv',
                                                                     'dev':'dev.csv',
                                                                     'test':'test.csv'}, streaming=False)  # TODO labels read as list
                for split in dataset.keys(): 
                    dataset[split] = dataset[split].map(lambda example: {'labels': ast.literal_eval(example['labels']), 'text': example['text']})


            
        return MultiStream.from_iterables(dataset)

import requests
from typing import Dict, Sequence
from pathlib import Path


dataset_name = 'contract_nli'



card = TaskCard(
        loader=LoadZipFromWeb(path='https://stanfordnlp.github.io/contract-nli/resources/',
                           data_files=['contract-nli.zip']),
#        preprocess_steps=[ TODO use multilabel here, and remove from above
#            MapInstanceValues(mappers={'labels': contract_nli_labels_dict})
#        ],
        task=FormTask(
            inputs=["text"],
            outputs=["labels"], 
            metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        ),
        templates=TemplatesList([
            InputOutputTemplate(
                input_format='{text}',
                output_format='{labels}',
            ),
        ])
)

add_to_catalog(artifact=card, name=f'cards.{dataset_name}',overwrite=True)
# ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
