from datasets import load_dataset
from unitxt import add_to_catalog

from unitxt.operators import ListFieldValues

from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    RenameFields,
)


dataset_name = 'head_qa'

card = TaskCard(
    loader=LoadHF(path=f'{dataset_name}', name='en'),
    preprocess_steps=[
        # TODO why I need to do it field, field?
        RenameFields(field_to_field={"qtext": "text", "category": "label"}),
        ListFieldValues(fields=["label"], to_field="label"),
        SplitRandomMix({'train': 'train', 'dev': 'validation', 'test':'test'}),
        
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
