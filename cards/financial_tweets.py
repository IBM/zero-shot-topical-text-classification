from datasets import load_dataset
from unitxt import add_to_catalog

from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    AddFields,
    TaskCard,
    NormalizeListFields,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues
)


dataset_name = 'financial_tweets'

map_labels = {
    '0': ['analyst update'],
    '1': ['fed and central banks'],
    '2': ['company and product news'],
    '3': ['treasuries and corporate debt'],
    '4': ['dividend'],
    '5': ['earnings'],
    '6': ['energy and oil'],
    '7': ['financials'],
    '8': ['currencies'],
    '9': ['general News and opinion'],
    '10': ['gold, metals and materials'],
    '11': ['initial public offering'],
    '12': ['legal and regulation'],
    '13': ['mergers, acquisitions and investments'],
    '14': ['macro'],
    '15': ['markets'],
    '16': ['politics'],
    '17': ['personnel change'],
    '18': ['stock commentary'],
    '19': ['stock movement'],
}

card = TaskCard(
    loader=LoadHF(path=f'zeroshot/twitter-financial-news-topic'),
    preprocess_steps=[
        SplitRandomMix(
            {'train': 'train[85%]', 'dev': 'train[15%]', 'test':'validation'}),
        MapInstanceValues(mappers={'label': map_labels})
    ],
    task=FormTask(
        inputs=["text"],
        outputs=["label"],
        metrics=["metrics.f1_micro",
                 "metrics.accuracy", "metrics.f1_macro"],
    ),
    templates=TemplatesList([
        InputOutputTemplate(
            input_format='{text}',
            output_format='{label}',
        ),
    ])
)

add_to_catalog(artifact=card, name=f'cards.{dataset_name}', overwrite=True)
# ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
