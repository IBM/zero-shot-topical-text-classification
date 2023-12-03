from tempfile import TemporaryDirectory
from datasets import load_dataset
from unitxt import add_to_catalog

from datasets import load_dataset as hf_load_dataset
from unitxt.stream import MultiStream
from typing import Sequence
import opendatasets as od

from unitxt.operators import FilterByListsOfValues

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
class LoadFromKaggle(Loader):
    url: str

    def _download_from_kaggle(self, url, data_dir):
        print(f"Downloading {url} ")
        od.download(url, data_dir)

    def process(self):
        with TemporaryDirectory() as temp_directory:
            self._download_from_kaggle(self.url, temp_directory)
            dataset = hf_load_dataset(temp_directory, streaming=False)

        return MultiStream.from_iterables(dataset)



dataset_name = 'news_category_classification_headline'

map_labels = {# 'THE WORLDPOST' is removed
    'EDUCATION': 'education', 'COLLEGE': 'college', 'POLITICS': 'politics', 'WEIRD NEWS': 'weird news', 'IMPACT': 'impact', 'BUSINESS': 'business', 'TECH': 'tech', 'FOOD & DRINK': 'food and drink', 'QUEER VOICES': 'queer voices', 'DIVORCE': 'divorce', 'COMEDY': 'comedy', 
              'WORLD NEWS': 'world news', 'WELLNESS': 'wellness', 'STYLE': 'style', 'HOME & LIVING': 'home and living', 'ENTERTAINMENT': 'entertainment', 'BLACK VOICES': 'black voices', 'MEDIA': 'media', 'WOMEN': 'women', 
              
              'GOOD NEWS': 'good news', 'ENVIRONMENT': 'environment', 'ARTS & CULTURE': 'arts and culture', 'WEDDINGS': 'weddings', 'WORLDPOST': 'worldpost', 'SPORTS': 'sports', 'PARENTING': 'parenting', 'SCIENCE': 'science', 'MONEY': 'money', 'RELIGION': 'religion', 'TASTE': 'taste', 'FIFTY': 'fifty', 'U.S. NEWS': 'u.s. news', 'CULTURE & ARTS': 'culture and arts', 'LATINO VOICES': 'latino voices', 'TRAVEL': 'travel', 'STYLE & BEAUTY': 'style and beauty', 'PARENTS': 'parents', 'ARTS': 'arts', 'GREEN': 'green', 'CRIME': 'crime', 'HEALTHY LIVING': 'healthy living'}

map_labels = {k:[v] for k,v in map_labels.items()}

card = TaskCard(
        loader=LoadFromKaggle(url='https://www.kaggle.com/datasets/rmisra/news-category-dataset'),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[70%]', 'dev': 'train[10%]', 'test': 'train[20%]'}),
            RenameFields(field_to_field={"headline": "text"}),
            RenameFields(field_to_field={"category": "label"}),
            FilterByListsOfValues(required_values={"label": list(map_labels)}), # removes THE WORLDPOST
            MapInstanceValues({'label':map_labels})
        ],
        templates=TemplatesList([
            InputOutputTemplate(
                input_format='{text}',
                output_format='{label}',
            ),
        ]),
        task=FormTask(
            inputs=["text"],
            outputs=["label"],
            metrics=["metrics.f1_micro", "metrics.accuracy", "metrics.f1_macro"],
        ),
)

add_to_catalog(artifact=card, name=f'cards.{dataset_name}',overwrite=True)
# ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
