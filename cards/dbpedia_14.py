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
    MapInstanceValues,
    RenameFields,
)


dataset_name = 'dbpedia_14'

map_labels = {
    '0': ['company'],
    '1': ['educational institution'],
    '2': ['artist'],
    '3': ['athlete'],
    '4': ['office holder'],
    '5': ['mean of transportation'],
    '6': ['building'],
    '7': ['natural place'],
    '8': ['village'],
    '9': ['animal'],
    '10': ['plant'],
    '11': ['album'],
    '12': ['film'],
    '13': ['written work']
}


card = TaskCard(
    loader=LoadHF(path=f'{dataset_name}'),
    preprocess_steps=[
        SplitRandomMix(
            {'train': 'train[87.5%]', 'dev': 'train[12.5%]', 'test': 'test'}),
        MapInstanceValues(mappers={'label': map_labels}),
        RenameFields(field_to_field={"content": "text"}),
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
#print(ds['dev']['additional_inputs'][0])