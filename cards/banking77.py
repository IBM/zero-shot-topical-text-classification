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



dataset_name = 'banking77'


map_labels = {
    '0': ['activating my card'],
    '1': ['age limit'],
    '2': ['apple pay or google pay'],
    '3': ['atm support'],
    '4': ['automatic top up'],
    '5': ['balance that has not been updated after a bank transfer'],
    '6': ['balance that has not been updated after cheque or cash deposit'],
    '7': ['beneficiary who is not allowed'],
    '8': ['canceling a transfer'],
    '9': ['card that is about to expire'],
    '10': ['card acceptance'],
    '11': ['card arrival'],
    '12': ['card delivery estimation'],
    '13': ['card linking'],
    '14': ['card not working'],
    '15': ['card payment fee that was charged'],
    '16': ['card payment not recognised'],
    '17': ['card payment wrong exchange rate'],
    '18': ['card swallowed'],
    '19': ['cash withdrawal charge'],
    '20': ['cash withdrawal not recognised'],
    '21': ['changing pin'],
    '22': ['compromised card'],
    '23': ['contactless not working'],
    '24': ['country support'],
    '25': ['declined card payment'],
    '26': ['declined cash withdrawal'],
    '27': ['declined transfer'],
    '28': ['direct debit payment not recognised'],
    '29': ['disposable card limits'],
    '30': ['editing personal details'],
    '31': ['exchange charge'],
    '32': ['exchange rate'],
    '33': ['exchange via app'],
    '34': ['extra charge on statement'],
    '35': ['failed transfer'],
    '36': ['fiat currency support'],
    '37': ['getting disposable virtual card'],
    '38': ['getting physical card'],
    '39': ['getting spare card'],
    '40': ['getting virtual card'],
    '41': ['lost or stolen card'],
    '42': ['lost or stolen phone'],
    '43': ['ordering physical card'],
    '44': ['forgotten passcode'],
    '45': ['pending card payment'],
    '46': ['pending cash withdrawal'],
    '47': ['pending top up'],
    '48': ['pending transfer'],
    '49': ['blocked pin'],
    '50': ['receiving money'],
    '51': ['refund not showing up'],
    '52': ['refund request'],
    '53': ['reverted card payment'],
    '54': ['supported cards and currencies'],
    '55': ['terminating account'],
    '56': ['top up by bank transfer charge'],
    '57': ['top up by card charge'],
    '58': ['top up by cash or cheque'],
    '59': ['failed top up'],
    '60': ['top up limits'],
    '61': ['top up reverted'],
    '62': ['topping up by card'],
    '63': ['transaction charged twice'],
    '64': ['charged transfer fee'],
    '65': ['transferring into account'],
    '66': ['transfer not received by recipient'],
    '67': ['transfer timing'],
    '68': ['being unable to verify identity'],
    '69': ['verifying my identity'],
    '70': ['verifying source of funds'],
    '71': ['verifying top up'],
    '72': ['virtual card not working'],
    '73': ['visa or mastercard'],
    '74': ['why identity verification is necessary'],
    '75': ['wrong amount of cash received'],
    '76': ['wrong exchange rate for cash withdrawal']
}

card = TaskCard(
    loader=LoadHF(path=f'PolyAI/{dataset_name}'),
    preprocess_steps=[
        SplitRandomMix(
            {'train': 'train[85%]', 'dev': 'train[15%]', 'test': 'test'}),
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
