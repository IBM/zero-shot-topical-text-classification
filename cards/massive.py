from datasets import load_dataset
from unitxt import add_to_catalog
from pathlib import Path

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


dataset_name = 'massive'
map_labels = {
    '0': ['getting date or time details'],
    '1': ['changing hue light'],
    '2': ['getting a transport ticket'],
    '3': ['getting a takeaway'],
    '4': ['stock'],
    '5': ['greeting'],
    '6': ['event recommendation'],
    '7': ['music dislikeness'],
    '8': ['turning off wemo'],
    '9': ['cooking recipes'],
    '10': ['currency'],
    '11': ['transport traffic'],
    '12': ['quirky issues'],
    '13': ['the weather'],
    '14': ['turning up the volume'],
    '15': ['adding email contact'],
    '16': ['takeaway order'],
    '17': ['getting email contact'],
    '18': ['increasing hue light'],
    '19': ['location recommendations'],
    '20': ['playing an audio book'],
    '21': ['creating or adding lists'],
    '22': ['the news'],
    '23': ['getting alarm details'],
    '24': ['turning wemo on'],
    '25': ['joke'],
    '26': ['definitions'],
    '27': ['social media'],
    '28': ['music settings'],
    '29': ['audio volume'],
    '30': ['removing from calendar'],
    '31': ['dimming hue light'],
    '32': ['getting calendar details'],
    '33': ['sending en email'],
    '34': ['cleaning'],
    '35': ['turning down volume'],
    '36': ['playing the radio'],
    '37': ['cooking details'],
    '38': ['converting date or time'],
    '39': ['math'],
    '40': ['turning off hue light'],
    '41': ['turning on hue light'],
    '42': ['getting transport details'],
    '43': ['music likeness'],
    '44': ['getting email details'],
    '45': ['playing music'],
    '46': ['muting audio volume'],
    '47': ['posting on social media'],
    '48': ['setting an alarm'],
    '49': ['factoids'],
    '50': ['setting the calendar'],
    '51': ['playing a game'],
    '52': ['removing an alarm'],
    '53': ['removing from lists'],
    '54': ['transport taxi'],
    '55': ['movie recommendations'],
    '56': ['making coffee'],
    '57': ['getting music details'],
    '58': ['playing podcasts'],
    '59': ['getting lists details']
}


card = TaskCard(
    loader=LoadHF(path=f'AmazonScience/{dataset_name}', name='en-US'),
    preprocess_steps=[
        SplitRandomMix(
            {'train': 'train', 'dev': 'validation', 'test': 'test'}),
        RenameFields(field_to_field={"intent": "label", "utt": "text"}),
        MapInstanceValues(mappers= {'label': map_labels})
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

add_to_catalog(artifact=card,
               name=f'cards.{dataset_name}', overwrite=True)
#ds = load_dataset(f'unitxt/data', f'card=cards.{dataset_name},template_card_index=0')
