from unitxt import add_to_catalog
from itertools import product 
from datasets import load_dataset_builder

from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    MapInstanceValues,
)

dataset_name = 'unfair_tos'
ds_builder = load_dataset_builder(path='lex_glue', name=f'{dataset_name}')
classlabels = ds_builder.info.features["labels"]

map_labels = {
    '[]': [],
    **{str([classlabels.index(x) for x in vector]):list(vector) for vector in product(classlabels)},
    **{str([classlabels.index(x) for x in vector]):list(vector) for vector in product(classlabels, classlabels)},
    **{str([classlabels.index(x) for x in vector]):list(vector) for vector in product(classlabels, classlabels, classlabels)}
}

card = TaskCard(
    loader=LoadHF(path='lex_glue', name=f'{dataset_name}'),
    preprocess_steps=[
        SplitRandomMix(
            {'train': 'train', 'dev': 'validation', 'test': 'test'}),
        MapInstanceValues(mappers= {'labels': map_labels}),
    ],
    task="tasks.classification.multi_class",
    templates="templates.classification.multi_class.all"
)

add_to_catalog(artifact=card, name=f'cards.{dataset_name}', overwrite=True)
# ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')




    
    