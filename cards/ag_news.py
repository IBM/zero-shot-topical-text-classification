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


dataset_name = 'ag_news'

map_labels = {
    "0": ["world"],
    "1": ["sports"],
    "2": ["business"],
    "3": ["science and technology"]
}

card = TaskCard(
    loader=LoadHF(path=f'{dataset_name}'),
    preprocess_steps=[
        SplitRandomMix(
            {'train': 'train[87.5%]', 'dev': 'train[12.5%]', 'test': 'test'}),
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
#ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
