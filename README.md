# Zero-shot Topical Text Classification
Code to reproduce experiments from the upcoming paper "Zero-shot Topical Text Classification with LLMs - an Experimental Study" (EMNLP, 2023).

Using this repository you can:

1. Download the datasets used in the paper.
2. Fine-tune DeBERTa-Large-mnli and Flan-t5-XXL on the datasets in the same manner that was described in the paper.
3. Evaluate the models.

Note: In the paper we described TTC23, a collection of 23 topical text classification datasets. Due to legal restrictions, the repository allows access to 19 of them.

## Installation

1. Clone the repository: `git clone git@github.com:IBM/zero-shot-classification-boost-with-self-training.git`.
2. Create a conda environment: `conda create -n zero-shot-ttc python=3.10; conda activate zero-shot-ttc`.
3. Install the project requirements: `pip install -r requirements.txt`.

To run the experiments, you will need access to a single A100_80GB GPU.

## Fine-tune DeBERTa-Large and Flan-t5-XXL

This step describes how to fine-tune DeBERTa-Large-mnli and Flan-t5-XXL similarly to what was done in the paper. That is, we split 19 datasets to 3 folds, train each model on a combination of 2 folds and evaluate on the datasets in the left-out fold.

The entry point for the experimental setup is `paper_pipeline.py`. This script runs a single experiment pipeline end-to-end. It supports the following arguments:

* `flow`: whether to fine-tune DeBERTa-Large-mnli (`deberta`) or Flan-t5-XXL (`flan`).
* `fold`: the index of the evaluation fold (`0`, `1` or `2`). The index refers to the list of 3 folds that are set in the `folds` variable (line 31). The datasets in the remaining folds will be used for training.
* `seed`: the seed to initialize the pipeline. In the paper, we executed the pipeline with 3 seeds (38, 40 and 42).
* `output_dir`: the location in which models and results are stored.

### Example run

`python pipeline.py flow=flan fold=0 seed=38 output_dir=flan_exp`: 
* Fine-tune Flan-t5-XXL
* Use fold 0 for evaluation (`reuters21578,
    claim_stance_topic,  unfair_tos, head_qa, banking77, ag_news` and `yahoo_answers_topics`) 
* Train on the datasets from folds `1` and `2`
* Use seed 38
* The output will be written to `flan_exp`

### Aggregate the results

TBA
