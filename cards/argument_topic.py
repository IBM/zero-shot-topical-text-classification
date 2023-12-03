from datasets import load_dataset
from unitxt import add_to_catalog

from unitxt.operators import ListFieldValues

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


dataset_name = 'argument_topic'


card = TaskCard(
    loader=LoadHF(path='ibm/argument_quality_ranking_30k', name=f'{dataset_name}'),
    preprocess_steps=[
        ListFieldValues(fields=["label"], to_field="label"),
        SplitRandomMix(
            {'train': 'train', 'dev': 'validation', 'test': 'test'}),
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