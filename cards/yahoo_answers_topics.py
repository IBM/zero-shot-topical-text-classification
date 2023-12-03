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


dataset_name = 'yahoo_answers_topics'

original_mappings = {
    "0": "Society & Culture",
    "1": "Science & Mathematics",
    "2": "Health",
    "3": "Education & Reference",
    "4": "Computers & Internet",
    "5": "Sports",
    "6": "Business & Finance",
    "7": "Entertainment & Music",
    "8": "Family & Relationships",
    "9": "Politics & Government"
}

our_mappings = {
    "Sports": "sports",
    "Health": "health",
    "Family & Relationships": "family and relationships",
    "Science & Mathematics": "science and mathematics",
    "Education & Reference": "education and reference",
    "Entertainment & Music": "entertainment and music",
    "Society & Culture": "society and culture",
    "Business & Finance": "business and finance’",
    "Politics & Government": "politics and government",
    "Computers & Internet": "computers and internet"
}

map_labels = {k: [our_mappings[v]] for k, v in original_mappings.items()}

card = TaskCard(
        loader=LoadHF(path=f'{dataset_name}'),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[87.5%]', 'dev': 'train[12.5%]', 'test': 'test'}),
            RenameFields(field_to_field={"topic": "label"}),
            MapInstanceValues(mappers={'label': map_labels}),
            ListFieldValues(fields=["question_title", "question_content", "best_answer"], to_field="text"),
            JoinStr(separator=" ", field="text", to_field="text"),
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
#print(ds['dev']['additional_inputs'][0])