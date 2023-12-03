from tempfile import TemporaryDirectory
from datasets import load_dataset
from unitxt import add_to_catalog
from pathlib import Path
from datasets import load_dataset as hf_load_dataset
from unitxt.stream import MultiStream
from typing import Sequence

from unitxt.blocks import (
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues,
    RenameFields,
)

from unitxt.loaders import Loader

class LoadFromWeb(Loader):
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

        with TemporaryDirectory() as temp_directory:
            for data_file in self.data_files:
                self._download_from_web(
                     self.path + "/" + data_file, temp_directory + "/" + data_file
                )
            dataset = hf_load_dataset(temp_directory, streaming=False)

        return MultiStream.from_iterables(dataset)

import requests
from typing import Dict, Sequence
from pathlib import Path
def download_data( url_to_file :Dict[str, str]):
    for url, file in url_to_file.items():
        response = requests.get(url, allow_redirects=True)
        if response.status_code != 200:
            raise Exception(f'request to {url} returns unexpected http code {response.status_code}')
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'wb') as f:
            f.write(response.content)


download_data({
    'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_train.csv': '/tmp/ilyashn/train.csv',
    'https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/medical_tc_test.csv': '/tmp/ilyashn/test.csv'
    })
    


BASE_DIR = 'from_unitext'

dataset_name = 'medical_abstracts'
Path(f'{BASE_DIR}/{dataset_name}').mkdir(parents=True, exist_ok=True)

map_labels = {
    "1": ["neoplasms"], 
    "2": ["digestive system diseases"],
    "3": ["nervous system diseases"],
    "4": ["cardiovascular diseases"],
    "5": ["general pathological conditions"]
}

card = TaskCard(
        loader=LoadFromWeb(path='https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main',
                           data_files=['medical_tc_train.csv', 'medical_tc_test.csv']),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[90%]', 'dev': 'train[10%]', 'test': 'test'}),
            RenameFields(field_to_field={"medical_abstract": "text"}),
            RenameFields(field_to_field={"condition_label": "label"}),
            MapInstanceValues(mappers={'label': map_labels})
        ],

        task=FormTask(
            inputs=["text"],
            outputs=["label"],
            metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        ),
        
        templates=TemplatesList([
            InputOutputTemplate(
                input_format='{text}',
                output_format='{label}',
            ),
        ])
)

add_to_catalog(artifact=card, name=f'cards.{dataset_name}',overwrite=True)
#ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
