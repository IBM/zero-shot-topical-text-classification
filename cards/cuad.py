import os
from tempfile import TemporaryDirectory
from datasets import load_dataset
from unitxt import add_to_catalog
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset as hf_load_dataset, DatasetDict, Value, Features
from unitxt.stream import MultiStream
from typing import Sequence
import requests
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
        def process_cuad(in_dir):
            def clean_text(t):
                t = t.replace("\n", " ").replace("  ", " ").replace("    ", " ").replace("&lt;", "<").replace("“", "\"").replace("”", "\"")
                t = re.sub(r"\s+", " ", t)
                return t

            # TODO move to mapping
            labels_to_fix = { 'Rofr/Rofo/Rofn':'right of first refusal, right of first offer or right of first negotiation'}


            print("processing cuad")
            out_dir = "data/cuad_paper/"
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            
            path = os.path.join(in_dir, "master_clauses.csv")

            df = pd.read_csv(path)
            data = defaultdict(list)
            for i, row in df.iterrows():
                for c in df.columns:
                    if c in ('Filename', 'Document Name', 'Parties', 'Agreement Date', 'Effective Date') or c.endswith(
                            'Answer') or len(row[c]) == 0:
                        continue
                    column_value = ast.literal_eval(row[c].lower())
                    for cv in column_value:
                        text = clean_text(cv)
                        if c not in data[text]:
                            data[text].append(labels_to_fix.get(c,c))

            out_df = pd.DataFrame({'text': list(data.keys()), 'labels': [x for x in data.values()]})
            out_file = os.path.join(out_dir, f'train.csv')
            out_df.to_csv(out_file, encoding='utf-8', index=False)
            return out_dir

        with TemporaryDirectory() as download_directory:
            for data_file in self.data_files:
                self._download_from_web(
                     self.path + "/" + data_file, download_directory + "/" + data_file
                )
            with TemporaryDirectory() as extract_directory:
                with zipfile.ZipFile(download_directory + "/" + data_file) as zf:
                    zf.extractall(extract_directory)
                processed_dir = process_cuad(extract_directory + "/" + "CUAD_v1")
                dataset = hf_load_dataset(processed_dir, streaming=False)
        return MultiStream.from_iterables(dataset)

import requests
from typing import Dict, Sequence
from pathlib import Path

dataset_name = 'cuad'

card = TaskCard(
        loader=LoadZipFromWeb(path='https://zenodo.org/records/4595826/files/CUAD_v1.zip?download=1',
                           data_files=['cuad.zip']),
        preprocess_steps=[ # TODO use multilabel mapping here, and remove from above
            SplitRandomMix({'train': 'train[70%]', 'dev': 'train[10%]', 'test': 'train[20%]'}),
#            MapInstanceValues(mappers={'labels': contract_nli_labels_dict})
        ],
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
#ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
