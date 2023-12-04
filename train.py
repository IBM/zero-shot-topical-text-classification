import argparse
import json
import random
import logging
import sys
from typing import Dict, Callable

import evaluate
import numpy as np
import pandas as pd
import os
from datasets import Dataset, DatasetDict
from datasets import load_metric, concatenate_datasets


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, InputFeatures, \
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EvalPrediction, \
    PreTrainedTokenizerBase, BitsAndBytesConfig
from transformers import AutoTokenizer, AutoConfig, EarlyStoppingCallback
import torch
import utils
import shutil
from pathlib import Path
from predict_with_generative_models import format_prompts

from peft import prepare_model_for_kbit_training, PeftConfig

import torch.distributed as dist


TRAIN_VERSION="1.0"


label2id = {
    'negative': 0,
    'positive': 1,
    'neutral': 2,
}
def read_file(file):
    df = pd.read_csv(file)
    df = df[['text', 'label']]
    
    return Dataset.from_pandas(df)

    dataset = DatasetDict()
    dataset['train'] = read_file(train_file)
    dataset['validation'] = read_file(dev_file)
    dataset['test'] = read_file(test_file)

    return dataset

def preprocess_and_tokenize(model_name, dataset, hypothesis_template, tokenizer, model,
                            max_seq_length):
    def lower_label_for_bart(label, model_name):
        if 'bart' in model_name:
            out = label.lower()
        elif 'DeBERTa' in model_name or 'deberta-v3' in model_name:
            out = label.lower()
        elif 'roberta' in model_name or 'deberta' in model_name:
            out = label
        else:
            raise ValueError(f"Model {model_name} not supported")
        return out

    labels = [model.config.label2id[lower_label_for_bart(lbl, model_name)] for lbl in dataset['label']]
    tokenized = []

    for text, cls, label, dataset_name in zip(
          dataset['text'], dataset['class'], labels, dataset['dataset_name']
        ):

        hypothesis = hypothesis_template.format(cls)
        try:
            inputs = (tokenizer.encode_plus(text, hypothesis, return_tensors='pt',
                                        add_special_tokens=True, pad_to_max_length=True,
                                        max_length=max_seq_length, truncation='only_first'))
        except:
            print(hypothesis, text, cls, label, dataset_name)
            continue
        if "deberta" in model_name.lower():
            tokenized.append(InputFeatures(input_ids=inputs['input_ids'].squeeze(0),
                                           attention_mask=inputs['attention_mask'].squeeze(0),
                                           token_type_ids=inputs['token_type_ids'],
                                           label=label))
        elif "roberta" in model_name or "bart" in model_name:
            tokenized.append(InputFeatures(input_ids=inputs['input_ids'].squeeze(0),
                                           attention_mask=inputs['attention_mask'].squeeze(0),
                                           #  token_type_ids=inputs['token_type_ids'],
                                           label=label))
        else:
            raise NotImplementedError('only supporting deberta roberta and bart models')

    random.shuffle(tokenized)
    return tokenized

def remove_checkpoints(output_dir):
    for p in Path(output_dir).glob('checkpoint-*'):
        try:
            shutil.rmtree(str(p))
        except:
            pass # ok, sometime somebody else removes the checkpoint, TODO check it and move to back to train.py

def get_model_for_config(base_model):
    if 'roberta' in base_model:
        return 'roberta-large-mnli'
    elif 'bart' in base_model:
        return 'facebook/bart-large-mnli'
    elif 'microsoft/deberta-v3-large' in base_model:
        return 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
    elif 'MoritzLaurer/DeBERTa' in base_model and 'large' in base_model:
        return 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'
    elif 'MoritzLaurer/DeBERTa' in base_model and 'small' in base_model:
        return 'MoritzLaurer/DeBERTa-v3-small-mnli-fever-docnli-ling-2c'
    elif 'deberta' in base_model:
        return 'Narsil/deberta-large-mnli-zero-cls'
    return None


def prepare_label(label, class_name, prompt):
    if "pair_wise_option" not in prompt:
        if label.lower() == "yes":
            return class_name
        else:
            return "not " + class_name
    else:
        return label.lower()
        # raise Exception(f"prompt {prompt} not supported in flan pair-wise training")


# TODO: improve run-time by removing tokenization
def preprocess_and_tokenize_t5(dataset, prompt, tokenizer,
                               max_seq_length, joint_metadata, output_path):
    def preprocess_func(examples):
        inputs = (
            tokenizer(examples['text'], truncation=True))  ###CHECK
        labels = tokenizer(examples['label'],
                           truncation=True)
        inputs['labels'] = labels['input_ids']
        return inputs

    examples = preprocess_t5_multi_class(dataset, prompt, tokenizer,
                                         max_seq_length, joint_metadata)

    examples.to_csv(output_path, index=False)
    tokenized_datasets = examples.map(preprocess_func, batched=True)
    return tokenized_datasets.remove_columns(
        examples.column_names)  # needed because of this thread https://github.com/huggingface/transformers/issues/15505


def preprocess_t5_multi_class(dataset, prompt, tokenizer,
                              max_seq_length, joint_metadata):

    df = dataset.to_pandas()
    data_grouped_by_dataset = df.groupby('dataset_name')

    all_texts = []
    all_labels = []

    for name, df in data_grouped_by_dataset:
        all_classes = joint_metadata[name]['labels']
        texts = format_prompts(prompt, df['text'],
                               all_classes, tokenizer, max_seq_length, shuffle_classes=True)
        all_labels.extend(df['label'])
        all_texts.extend(texts)
    examples = Dataset.from_dict({'text': all_texts, 'label': all_labels})
    return examples


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def get_eval_func(tokenizer: PreTrainedTokenizerBase, metric_id: str) -> Callable:
    def eval_func(eval_preds: EvalPrediction):
        preds = eval_preds.predictions
        labels = eval_preds.label_ids
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels.tolist(), skip_special_tokens=True)
        metric = evaluate.load(metric_id)
        res = metric.compute(references=decoded_labels, predictions=decoded_preds)
        return res

    return eval_func


def finetune_t5_model(base_model, seed, train_df,
                              dev_df,
                              learning_rate, batch_size,
                              prompt,
                              out_dir, max_seq_length, num_epochs,
                              select_best_model,
                              fp16, joint_metadata, gradient_accumulation_steps, use_lora,
                              evaluation_and_save_strategy, save_steps):

    from peft import TaskType, LoraConfig, get_peft_model
    train_dataset = Dataset.from_pandas(train_df)
    if dev_df is not None:
        dev_dataset = Dataset.from_pandas(dev_df)
    else:
        dev_dataset = None
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True,
                                              model_max_length=max_seq_length)

    os.makedirs(out_dir, exist_ok=True)

    tokenized_train = preprocess_and_tokenize_t5(train_dataset, prompt, tokenizer,
                               max_seq_length, joint_metadata, os.path.join(out_dir, "train.csv"))
    tokenized_dev = preprocess_and_tokenize_t5(dev_dataset, prompt, tokenizer,
                                                 max_seq_length, joint_metadata, os.path.join(out_dir, "dev.csv"))

    bnb_config = None

    model_loading_args = {
        "pretrained_model_name_or_path": base_model,
        "quantization_config": bnb_config,
        "trust_remote_code": True,
        "torch_dtype":  None,
        "load_in_8bit": False,
        # "device_map": "auto"
    }

    model = AutoModelForSeq2SeqLM.from_pretrained(**model_loading_args)

    if use_lora:
        task_type = TaskType.SEQ_2_SEQ_LM

        config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=None,
            lora_dropout=0.05,
            bias="none",
            task_type=task_type
        )

        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer,
                                           model=model,
                                           padding=True,
                                           label_pad_token_id=tokenizer.eos_token_id
                                           )

    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs = num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        logging_strategy=evaluation_and_save_strategy,
        save_strategy=evaluation_and_save_strategy,
        evaluation_strategy=evaluation_and_save_strategy,
        save_steps=save_steps,
        logging_steps=save_steps,
        eval_steps=save_steps,

        save_total_limit=1,
        load_best_model_at_end=select_best_model,
        metric_for_best_model="exact_match",

        predict_with_generate=True,
        generation_max_length=20,
        generation_num_beams=1,

        fp16=fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_accumulation_steps=1,
        optim="adamw_hf",
        lr_scheduler_type="linear",
        learning_rate=learning_rate,
        warmup_steps=2,
        use_mps_device=False,
        overwrite_output_dir=True,
        seed=seed,
        report_to="none",
    )

    print(f"training_args {training_args}")

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_eval_func(tokenizer, "exact_match"),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]

    )

    trainer.train()
    trainer.save_model(out_dir)
    trainer.save_state()
    tokenizer.save_pretrained(out_dir)

    return out_dir


def finetune_entailment_model(base_model, seed, train_df,
                              dev_df,
                              learning_rate, batch_size,
                              hypothesis_template,
                              out_dir, max_seq_length, num_epochs,
                              select_best_model,
                              stop_early,
                              fp16, gradient_accumulation_steps,
                              evaluation_and_save_strategy,
                              save_steps):

    train_dataset = Dataset.from_pandas(train_df)
    if dev_df is not None:
        dev_dataset = Dataset.from_pandas(dev_df)
    else:
        dev_dataset = None
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True,
                _from_pipeline='zero-shot-classification', model_max_length=max_seq_length)

    model_for_config = get_model_for_config(base_model)
    if model_for_config:
        config_for_model = AutoConfig.from_pretrained(model_for_config)
        num_labels = config_for_model.num_labels
        model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=num_labels)
        model.config.label2id = config_for_model.label2id
        model.config.id2label = config_for_model.id2label
    else:
        model = AutoModelForSequenceClassification.from_pretrained(base_model)

    tokenized_train = preprocess_and_tokenize(base_model, train_dataset, hypothesis_template, tokenizer, model, max_seq_length)

    callbacks = []
    if stop_early:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=2, 
                                  early_stopping_threshold=1e-3)
        ) 

    def compute_metrics(p):
        if 'bart' in base_model:
            logits = p.predictions[0]
        elif 'roberta' in base_model:
            logits = p.predictions
        elif 'deberta' in base_model:
            logits = p.predictions
        elif 'DeBERTa' in base_model:
            logits = p.predictions
        else:
            raise Exception(f"unexepected base model {base_model}")
        labels = p.label_ids
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    metric = load_metric("accuracy")
    if dev_dataset is not None:
        tokenized_dev = preprocess_and_tokenize(base_model, dev_dataset, hypothesis_template, tokenizer,
                                                model, max_seq_length)
    else:
        tokenized_dev = None

    training_args = TrainingArguments(output_dir=out_dir,
                                      overwrite_output_dir=True,
                                      num_train_epochs=num_epochs,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      learning_rate=learning_rate,
                                      load_best_model_at_end=select_best_model,
                                      save_total_limit=1,
                                      save_strategy=evaluation_and_save_strategy,
                                      evaluation_strategy=evaluation_and_save_strategy if stop_early else "no",
                                      save_steps=save_steps,
                                      metric_for_best_model='accuracy',
                                      report_to="none",
                                      seed=seed,
                                      fp16=fp16,
                                      gradient_accumulation_steps=gradient_accumulation_steps
                                      # half_precision_backend="cuda_amp" #cuda_amp auto or apex
                                      )
    print(f"training_args {training_args}")
    # model = torch.compile(model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        callbacks=callbacks,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    trainer.evaluate()

    return out_dir


parser = argparse.ArgumentParser(description='params for pair-wise training')

#parser.add_argument("--output_dir", required=True)
# TODO add types and actions. 
parser.add_argument("--base_model", required=True)
parser.add_argument("--input_dir", required=True)
parser.add_argument("--held_out_input_dir", default=None)
parser.add_argument("--learning_rate", required=True)
parser.add_argument("--train_df_name", required=True)
parser.add_argument("--dev_df_name", required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", required=True)
parser.add_argument("--hypothesis_template", required=True, default='This text is about {}')
parser.add_argument("--max_seq_length", required=True)
parser.add_argument("--num_epochs", required=True)
parser.add_argument("--select_best_model", action='store_true')
parser.add_argument("--stop_early", action='store_true')
parser.add_argument("--output_dir", required=True)
parser.add_argument("--use_lora", action='store_true')
parser.add_argument('--training_method', type=str, default='pairwise', 
                    choices=['pairwise', 't5'],
    help='What approached to be been used in training')
parser.add_argument("--fp16", action='store_true', help='Should we use fp16 for training?')
parser.add_argument("--gradient_accumulation_steps", default=1)

parser.add_argument("--evaluation_and_save_strategy", default="epoch")
parser.add_argument("--save_steps", type=int, default=500)

parser.add_argument('--version', type=str, required=True, help='version to update on change of code')

def main():
    args = parser.parse_args()
    run_train(base_model=args.base_model, input_dir=args.input_dir, held_out_input_dir=args.held_out_input_dir, 
              learning_rate=args.learning_rate, train_df_name=args.train_df_name, dev_df_name=args.dev_df_name, 
              batch_size=args.batch_size, hypothesis_template=args.hypothesis_template, max_seq_length=args.max_seq_length, 
              num_epochs=args.num_epochs, select_best_model=args.select_best_model, stop_early=args.stop_early, 
              output_dir=args.output_dir, use_lora=args.use_lora,
              training_method=args.training_method, fp16=args.fp16, gradient_accumulation_steps=args.gradient_accumulation_steps,
              evaluation_and_save_strategy=args.evaluation_and_save_strategy, save_steps=args.save_steps, 
              seed=args.seed)

def run_train(base_model, input_dir,  learning_rate, train_df_name, dev_df_name, 
              batch_size, max_seq_length, num_epochs, output_dir, fp16, 
              use_lora=False,
              select_best_model=False,
              stop_early=False,
              seed=42,
              held_out_input_dir=None,
              hypothesis_template="This text is about {}.",
              training_method="pairwise",  
              gradient_accumulation_steps=1, 
              evaluation_and_save_strategy="epoch", 
              save_steps=500,
              ):
    
    logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

    utils.set_seed(seed)
    utils.set_torch_seed(seed)
    
    if training_method == "t5":
        config_file = os.path.join(output_dir, 'adapter_config.json')
    elif training_method == 'pairwise':
        config_file = os.path.join(output_dir, 'config.json')
    else:
        raise Exception(f"unsupported training method {training_method}")
    
    if config_file and os.path.exists(config_file): # let's use the previous results
        return
    
    train_df = pd.read_csv(os.path.join(input_dir ,train_df_name))
    if dev_df_name is not None:
        if held_out_input_dir is not None:
            dev_df = pd.read_csv(os.path.join(held_out_input_dir, dev_df_name))
        else:
            dev_df = pd.read_csv(os.path.join(input_dir, dev_df_name))
    else:
        dev_df = None

    with open(os.path.join(input_dir, "joint_metadata.json")) as f:
        joint_metadata = json.load(f)

    if held_out_input_dir is not None:
        with open(os.path.join(held_out_input_dir, "joint_metadata.json")) as f:
            held_out_joint_metadata = json.load(f)
        joint_metadata.update(held_out_joint_metadata)

    template = hypothesis_template

    print('template', template, len(template)) # WORDAROND for arguments with space
    if template[0] == "'":
        template = template[1:-1]
    print('new template', template, len(template))

    if training_method == 'pairwise':
        finetune_entailment_model(base_model, int(seed), train_df, dev_df,
                                float(learning_rate), int(batch_size),
                                template,
                                output_dir, int(max_seq_length), int(num_epochs), 
                                select_best_model, 
                                stop_early, 
                                fp16, int(gradient_accumulation_steps),
                                evaluation_and_save_strategy,
                                save_steps)

    elif training_method  == "t5":
        finetune_t5_model(base_model=base_model, seed=int(seed), train_df=train_df, dev_df=dev_df,
                                learning_rate=float(learning_rate), batch_size=int(batch_size),
                                prompt=template,
                                out_dir=output_dir, max_seq_length=int(max_seq_length), num_epochs=int(num_epochs),
                                select_best_model=select_best_model,
                                fp16=fp16, joint_metadata=joint_metadata,
                                gradient_accumulation_steps=int(gradient_accumulation_steps), use_lora=use_lora,
                                evaluation_and_save_strategy=evaluation_and_save_strategy,
                                save_steps=save_steps)
    else:
        raise Exception(f"Unexpected training method {training_method}")

    # remove_checkpoints(output_dir)

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = []
__ARGPARSER__ = parser

if __name__ == "__main__":
    main()
