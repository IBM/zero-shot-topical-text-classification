from datasets import load_dataset
from unitxt import add_to_catalog

from unitxt.operators import ListFieldValues, JoinStr

from unitxt.blocks import (
    LoadHF,
    SplitRandomMix,
    TaskCard,
    FormTask,
    TemplatesList,
    InputOutputTemplate,
    MapInstanceValues,
    RenameFields
)



dataset_name = 'law_stack_exchange'
# additional_text="title"
map_labels = {
    'contract-law': ['contract law'],
    'constitutional-law': ['constitutional law'],
    'liability': ['liability'],
    'trademark': ['trademark'],
    'business': ['business'],
    'criminal-law': ['criminal law'],
    'privacy': ['privacy'],
    'tax-law': ['tax law'],
    'software': ['software'], 
    'internet': ['internet'], 
    'civil-law': ['civil law'],
    'intellectual-property': ['intellectual property'], 
    'licensing': ['licensing'],
    'employment': ['employment'], 
    'contract': ['contract'], 
    'copyright': ['copyright'],
}

card = TaskCard(
    loader=LoadHF(path=f'jonathanli/law-stack-exchange', name='en'),
    preprocess_steps=[
        SplitRandomMix(
            {"train": "test", "test": "train", "dev": "validation"}),
        RenameFields(field_to_field={"text_label": "label"}),
        MapInstanceValues(mappers={'label': map_labels}),
        ListFieldValues(fields=["title", "body"], to_field="text"),
        JoinStr(separator=". ", field="text", to_field="text"),
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
