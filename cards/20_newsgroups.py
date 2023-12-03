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


dataset_name = '20_newsgroups'

map_labels = {
        'alt.atheism': ['atheism'],
        'comp.graphics': ['computer graphics'],
        'comp.os.ms-windows.misc': ['microsoft windows'],
        'comp.sys.ibm.pc.hardware': ['pc hardware'],
        'comp.sys.mac.hardware': ['mac hardware'],
        'comp.windows.x': ['windows x'],
        'misc.forsale': ['for sale'],
        'rec.autos': ['cars'],
        'rec.motorcycles': ['motorcycles'],
        'rec.sport.baseball': ['baseball'],
        'rec.sport.hockey': ['hockey'],
        'sci.crypt': ['cryptography'],
        'sci.electronics': ['electronics'],
        'sci.med': ['medicine'],
        'sci.space': ['space'],
        'soc.religion.christian': ['christianity'],
        'talk.politics.guns': ['guns'],
        'talk.politics.mideast': ['middle east'],
        'talk.politics.misc': ['politics'],
        'talk.religion.misc': ['religion'],
    }

card = TaskCard(
        loader=LoadHF(path=f'SetFit/{dataset_name}'),
        preprocess_steps=[
            SplitRandomMix({'train': 'train[90%]', 'dev': 'train[10%]', 'test': 'test'}),
            RenameFields(field_to_field={"label_text": "label"}),
            MapInstanceValues(mappers={'label': map_labels}), # TODO remove lines with empty texts
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
