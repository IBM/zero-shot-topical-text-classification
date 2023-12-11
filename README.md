# Zero-shot Topical Text Classification

This repostiroy contains code to reproduce experiments from the paper [Zero-shot Topical Text Classification with LLMs - an Experimental Study](https://aclanthology.org/2023.findings-emnlp.647.pdf) (EMNLP, 2023).

The paper evaluates several off-the-shelf models on a diverse collection of datasets that handle topical text classification tasks. Furthermore, the paper shows how additional fine-tuning dedicated to topical text classification can further improve the zero-shot capabilities of LLMs such as the Flan-t5-XXL on this task.

To use our work, you can either:

1. Download our Flan-t5-XXL model - details TBA.
2. Build your own models using the instructions below.

Note: In the paper we described TTC23, a collection of 23 topical text classification datasets. Due to legal restrictions, the repository allows access to 19 of them.

## Introduction

Using this repository you can:

1. Download the datasets used in the paper.
2. Fine-tune DeBERTa-Large-mnli and Flan-t5-XXL on the datasets in the same manner that was described in the paper.
3. Evaluate the models.

## Installation

1. Clone the repository: `git clone git@github.com:IBM/zero-shot-classification-boost-with-self-training.git`.
2. Create a conda environment: `conda create -n zero-shot-ttc python=3.10; conda activate zero-shot-ttc`.
3. Install the project requirements: `pip install -r requirements.txt`.
4. Create api-key using instuctions [here](https://christianjmills.com/posts/kaggle-obtain-api-key-tutorial/) and save kaggle.json in the local dir. We use opendatasets library to fetch the `News cateogry classification` dataset.

To run the experiments, you will need access to a single A100_80GB GPU.

## Reproduce paper experiments

This step describes how to fine-tune DeBERTa-Large-mnli and Flan-t5-XXL similarly to what was done in the paper. That is, we split 19 datasets to 3 folds, train each model on a combination of 2 folds and evaluate on the datasets in the left-out fold. We do this for 3 seeds, so overall there are 18 training processes to complete (2 models, 3 folds, 3 feeds).

The entry point for the experimental setup is `paper_pipeline.py`. This script runs a single experiment pipeline end-to-end (that is, one model, fold and seed). It supports the following arguments:

* `flow`: whether to fine-tune DeBERTa-Large-mnli (`deberta`) or Flan-t5-XXL (`flan`).
* `fold`: the index of the evaluation fold (`0`, `1` or `2`). The index refers to the list of 3 folds that are set in the `folds` variable (line 31). The datasets in the remaining folds will be used for training.
* `seed`: the seed to initialize the pipeline. In the paper, we executed the pipeline with 3 seeds (38, 40 and 42).
* `output_dir`: the location in which models and results are stored.

### Example run

By running the following command - `python pipeline.py --flow flan --fold 0 --seed 38 --output_dir flan_exp` - you will:

* Fine-tune Flan-t5-XXL
* Evaluate fold 0 (`reuters21578, claim_stance_topic,  unfair_tos, head_qa, banking77, ag_news` and `yahoo_answers_topics`) 
* Train on the datasets from folds 1 and 2
* Use seed 38
* Write the output to `flan_exp`

### Run the experiments for remaining folds and seeds

Here are the commands to execute to run the remaining 8 fold and seed combinations for the `flan` pipeline:

`python pipeline.py --flow flan --fold 0 --seed 40 --output_dir flan_exp`

`python pipeline.py --flow flan --fold 0 --seed 42 --output_dir flan_exp`

`python pipeline.py --flow flan --fold 1 --seed 38 --output_dir flan_exp`

`python pipeline.py --flow flan --fold 1 --seed 40 --output_dir flan_exp`

`python pipeline.py --flow flan --fold 1 --seed 42 --output_dir flan_exp`

`python pipeline.py --flow flan --fold 2 --seed 38 --output_dir flan_exp`

`python pipeline.py --flow flan --fold 2 --seed 40 --output_dir flan_exp`

`python pipeline.py --flow flan --fold 2 --seed 42 --output_dir flan_exp`

### Aggregate the results

After all runs have completed, you can use the following script to aggregate them:

`python aggregate.py --flow flan --output_dir flan_exp`

### Run the experiment for fine-tuning DeBERTa-Large-mnli:

`python pipeline.py --flow deberta --fold 0 --seed 40 --output_dir deberta_exp`

`python pipeline.py --flow deberta --fold 0 --seed 42 --output_dir deberta_exp`

`python pipeline.py --flow deberta --fold 1 --seed 38 --output_dir deberta_exp`

`python pipeline.py --flow deberta --fold 1 --seed 40 --output_dir deberta_exp`

`python pipeline.py --flow deberta --fold 1 --seed 42 --output_dir deberta_exp`

`python pipeline.py --flow deberta --fold 2 --seed 38 --output_dir deberta_exp`

`python pipeline.py --flow deberta --fold 2 --seed 40 --output_dir deberta_exp`

`python pipeline.py --flow deberta --fold 2 --seed 42 --output_dir deberta_exp`

And accordingly, to aggregate the results run the following:

python aggregate.py --flow deberta --output_dir deberta_exp

### Caching

There is a caching mehanism that prevents execution of some of the steps (e.g. processing datasets) twice. If the code changes it is recommended to remove the output_dir.

## Reference

If you refer to this work in a paper please cite:

```
@inproceedings{gretz-etal-2023-zero,
    title = "Zero-shot Topical Text Classification with {LLM}s - an Experimental Study",
    author = "Gretz, Shai  and
      Halfon, Alon  and
      Shnayderman, Ilya  and
      Toledo-Ronen, Orith  and
      Spector, Artem  and
      Dankin, Lena  and
      Katsis, Yannis  and
      Arviv, Ofir  and
      Katz, Yoav  and
      Slonim, Noam  and
      Ein-Dor, Liat",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.647",
    pages = "9647--9676",
    abstract = "Topical Text Classification (TTC) is an ancient, yet timely research area in natural language processing, with many practical applications. The recent dramatic advancements in large LMs raise the question of how well these models can perform in this task in a zero-shot scenario. Here, we share a first comprehensive study, comparing the zero-shot performance of a variety of LMs over TTC23, a large benchmark collection of 23 publicly available TTC datasets, covering a wide range of domains and styles. In addition, we leverage this new TTC benchmark to create LMs that are specialized in TTC, by fine-tuning these LMs over a subset of the datasets and evaluating their performance over the remaining, held-out datasets. We show that the TTC-specialized LMs obtain the top performance on our benchmark, by a significant margin. Our code and model are made available for the community. We hope that the results presented in this work will serve as a useful guide for practitioners interested in topical text classification.",
}
```


## License

This codebase is released under the Apache 2.0 license. The full text of the license can be found in [LICENSE](LICENSE). Note, models and datasets associated with this work have separate licenses.

