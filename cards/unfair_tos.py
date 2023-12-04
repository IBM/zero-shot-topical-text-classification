from unitxt import add_to_catalog
from datasets import load_dataset

from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    MapInstanceValues,
    FormTask,
    TemplatesList,
    InputOutputTemplate
)

dataset_name = 'unfair_tos'
# TODO extracting classlabels doesn't work :(
# ds_builder = load_dataset_builder(path='lex_glue', name=f'{dataset_name}')
classlabels = ['Limitation of liability', 'Unilateral termination', 'Unilateral change', 'Content removal', 'Contract by using', 'Choice of law', 'Jurisdiction', 'Arbitration']
map_labels = {str(i):v for i,v in enumerate(classlabels)}

card = TaskCard(
    loader=LoadHF(path='lex_glue', name=f'{dataset_name}'),
    preprocess_steps=[
        SplitRandomMix(
            {'train': 'train', 'dev': 'validation', 'test': 'test'}),
        MapInstanceValues(mappers= {'labels': map_labels}, process_every_value=True),
    ],
    task=FormTask(
        inputs=["text"],
        outputs=["labels"],
        metrics=["metrics.f1_micro",
                 "metrics.accuracy", "metrics.f1_macro"],
    ),
    templates=TemplatesList([
        InputOutputTemplate(
            input_format='{text}',
            output_format='{labels}',
        ),
    ])
)

add_to_catalog(artifact=card, name=f'cards.{dataset_name}', overwrite=True)
# ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')




    
    