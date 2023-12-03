import math
import random
import re
from difflib import get_close_matches
import time
import numpy as np
import torch
import tqdm
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer


import pandas as pd
import argparse
import os
import logging
import json

import ast
import utils

from config import paper_datasets, prompt_config

PREDICT_WITH_GENERATIVE_MODELS_VERSION = "1.1"
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset')
parser.add_argument('--model_dir', type=str, default='google/flan-t5-small', help='model to be used for the predictions')
parser.add_argument('--input_dir', type=str, required=True, help='Input dir')
parser.add_argument('--input_files', type=str, required=True, help='Comma separated Input files')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--training_method', type=str, default='T5', help='Type of method to run predict '
                                                                       '(multi-class or pair-wise)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_seq_length', type=int, default=128)
parser.add_argument('--prompt', type=str, help='Prompt to send to generation', default='Classify text into categories: {class_names}.\ntext: {text}\ncategory: ')
parser.add_argument('--num_sequences', type=int, help='Number of sequences to generate', default=1)
parser.add_argument("--fp16", action='store_true', help='Should we use fp16 for prediction?')
parser.add_argument('--version', type=str, required=True, help='version to update on change of code')
parser.add_argument('--use_lora',  action='store_true', help='whether to use LoRA model')
parser.add_argument('--sample_size', type=int, help='Sample the data', default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument('--use_logits_processor',  action='store_true', help='whether to use the logits processor')

__OUTPUT_DIR_ARG__ = "output_dir"
__OUTPUTS__ = []
__ARGPARSER__ = parser

#
# def bullets(args):
#     return "\n".join(["- " + t.strip() for t in args['class_names'].split(",")])
#
#
# def str_join_or(args):
#     topics = ["\"" + c.strip() + "\"" for c in args['class_names'].split(",")]
#     return ', '.join(topics[:-1]) + ' or ' + topics[-1]
#
#


def simple_match_to_class_name(txt, class_names):
    def _normalize(name):
        return name.lower().replace('/', ' and ').replace('&', ' and ')

    norm_txt = _normalize(txt)
    norm_class_names = [_normalize(c) for c in class_names]
    return [class_names[i] for i, norm_cls in enumerate(norm_class_names) if norm_txt == norm_cls][: 1]


def generated_text_to_class_name(txt, class_names, max_matches=1):
    def _normalize(name):
        return name.lower().replace('/', ' and ').replace('&', ' and ')

    norm_txt = _normalize(txt)
    norm_class_names = [_normalize(c) for c in class_names]
    matches = get_close_matches(norm_txt, norm_class_names, max_matches)
    if len(matches) == 0:
        matches = [cls for cls in norm_class_names if norm_txt in cls][: max_matches]
    return [class_names[norm_class_names.index(m)] for m in matches]


def arrange_scores(predicted_classes, predicted_scores, class_names):
    res = []
    for cls_list, score_list in zip(predicted_classes, predicted_scores): # going over predictions for each text
        all_scores = [(c, 0.) for c in class_names]
        for cls, score in zip(cls_list, score_list): # going over each single prediction
            idx = class_names.index(cls)
            all_scores[idx] = (cls, score)
        all_scores = sorted(all_scores, key=lambda x: x[1], reverse=True)
        res.append(all_scores)
    return res


def clean_text(text):
    return text.replace("categories: ", "").lower()


class ClassNameTokens:
    def __init__(self, device, tokenizer, class_names, fp16):
        self.device = device
        tokens = [tokenizer.encode(cn) for cn in class_names]
        max_len = max([len(t) for t in tokens])
        padded_tokens = [[0] + t + [0] * (max_len - len(t)) for t in tokens]
        class_name_tokens = torch.tensor(padded_tokens)
        self.class_name_tokens = class_name_tokens.to(device) if device is not None else class_name_tokens
        self.max_num_tokens = max_len
        self.fp16 = fp16

        self.next_token_probs = None  # probabilities of possible next tokens
        self.seq_token_probs = None  # seq of probabilities of selected tokens

    # logits_processor is used in generation_utils.py, line 2048
    def logits_processor(self, input_ids, scores):
        in_seq_len = input_ids.shape[1]
        if in_seq_len == 1:
            self.seq_token_probs = [[] for _ in range(len(input_ids))]
        else:
            selected_tokens = input_ids[:, -1] # excluding first dummy token
            probs = [possible_tokens[token.item()] for possible_tokens, token in
                     zip(self.next_token_probs, selected_tokens)] # getting probs of tokens from previous step
            for seq, prob in zip(self.seq_token_probs, probs): # adding probs to sequence, one sequence per entry in input_ids
                seq.append(prob)

        self.next_token_probs = [None for _ in range(len(input_ids))]

        next_token_scores = torch.full_like(scores, -torch.inf, dtype=torch.float32)
        # next_token_scores = torch.Tensor([[-torch.inf] * scores.shape[1]] * scores.shape[0])

        prev_tokens = self.class_name_tokens[:, : in_seq_len]
        next_tokens = self.class_name_tokens[:, in_seq_len]
        if self.fp16:
            scores = scores.type(torch.float32)
        for i, (in_seq, in_seq_logits) in enumerate(zip(input_ids, scores)):
            class_mask = torch.sum(in_seq == prev_tokens, dim=1, dtype=torch.int) == in_seq_len
            seq_next_tokens = torch.where(class_mask, next_tokens, -torch.ones_like(next_tokens)) # selecting next tokens that are permitted according to the class ids
            if self.device is not None:
                seq_next_tokens = seq_next_tokens.cpu()
            seq_next_tokens = [t for t in set(seq_next_tokens.numpy()) - {-1}]
            if len(seq_next_tokens) == 0:
                seq_next_tokens = [1]
            next_token_scores[i, seq_next_tokens] = in_seq_logits[seq_next_tokens] # getting scores of permitted tokens

            if in_seq_len == 1:
                # calibrate the first token score
                best_token = torch.argmax(in_seq_logits).item()
                worst_token = torch.argmin(in_seq_logits).item()
                extra = [t for t in [best_token, worst_token] if t not in seq_next_tokens]
            else:
                extra = []

            if len(extra) > 0: # getting probs of next tokens conditions on the scores of all permitted tokens
                seq_next_tokens_probs = torch.softmax(in_seq_logits[(seq_next_tokens + extra)], dim=0)[:-len(extra)]
            else:
                seq_next_tokens_probs = torch.softmax(in_seq_logits[seq_next_tokens], dim=0)

            self.next_token_probs[i] = {t: prob.item() for t, prob in zip(seq_next_tokens, seq_next_tokens_probs)}

        return next_token_scores


def concat_list_batches(list_batches, class_batch_size=100):
    size = len(list_batches[0]) # number of total texts
    assert all(len(lst) == size for lst in list_batches)
    if class_batch_size > 1:
        return [list(np.concatenate(list(lst[i] for lst in list_batches))) for i in range(size)]
    else:
        return [list(np.concatenate([list(lst[i] for lst in list_batches)], axis=0)) for i in range(size)]


def infer_with_score(texts, class_names, model_name, tokenizer, batch_size, multi_label, fp16, use_lora,
                     use_logits_processor):
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    if "flan" in model_name or "train" in model_name: # assuming trained model is flan-based
        if use_lora:

            peft_model_name_or_path = model_name
            peft_config = PeftConfig.from_pretrained(peft_model_name_or_path)
            model_id = peft_config.base_model_name_or_path

            model_loading_args = {
                "pretrained_model_name_or_path": model_id,
                "quantization_config": None,
                "trust_remote_code": True
            }

            model = AutoModelForSeq2SeqLM.from_pretrained(**model_loading_args)
            model = PeftModel.from_pretrained(model, peft_model_name_or_path)
            print("Loaded model", str(model))
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        raise ValueError(f"{model_name} not supported")
    if device is not None:
        model.to(device)
        if fp16:
            model = model.half()

    permitted_class_names = class_names if len(class_names) > 1 else (class_names + ["not " + class_names[0]]) if \
        "Answer yes or no" not in texts[0] else ['yes', 'no']

    print(permitted_class_names)

    class_name_tokens = ClassNameTokens(device, tokenizer, permitted_class_names, fp16)

    predicted_texts = []
    predicted_classes = []
    predicted_scores = []
    for begin in tqdm.trange(0, len(texts), batch_size):
        text_batch = texts[begin: begin + batch_size]

        generate_and_add_to_results_list_with_logits_processor(class_name_tokens, permitted_class_names, device, model,
                                                                   multi_label, predicted_classes,
                                                                   predicted_scores, predicted_texts, text_batch,
                                                                   tokenizer)

    return predicted_texts, predicted_classes, predicted_scores


def generate_and_add_to_results_list_with_logits_processor(class_name_tokens, class_names, device, model, multi_label,
                                                           predicted_classes, predicted_scores, predicted_texts, text_batch, tokenizer):
    input_ids_batch = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True).input_ids
    if device is not None:
        input_ids_batch = input_ids_batch.to(device)
    generated_outputs = model.generate(
        input_ids=input_ids_batch,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=class_name_tokens.max_num_tokens,
        num_return_sequences=1,
        logits_processor=[lambda input_ids, scores: class_name_tokens.logits_processor(input_ids, scores)],
    )
    gen_sequences = generated_outputs.sequences[:,
                    1:]  # all sequences excluding dummy token. size of gen_sequences: batch_size * num_sequences
    seq_scores = torch.Tensor(
        [np.product(token_probs) for token_probs in class_name_tokens.seq_token_probs])  # all probs
    for ex_begin in range(0, len(gen_sequences), 1):
        ex_sequences = gen_sequences[ex_begin: ex_begin + 1,
                       :]  # all sequences generated for given example (determined by num_sequences)
        ex_seq_scores = seq_scores[ex_begin: ex_begin + 1]
        ex_gen_texts = [clean_text(tokenizer.decode(seq, skip_special_tokens=True)) for seq in ex_sequences]

        ex_uniq_gen_texts = list(set(ex_gen_texts))
        ex_uniq_scores = [ex_seq_scores[ex_gen_texts.index(txt)].item() for txt in ex_uniq_gen_texts]
        ex_classes = {}  # collecting classes and max score for each class generated in example sequences
        for txt, txt_score in zip(ex_uniq_gen_texts, ex_uniq_scores):
            cls_matches = simple_match_to_class_name(txt, class_names)
            for cls_name in cls_matches:
                ex_classes[cls_name] = max(txt_score, ex_classes.get(cls_name, -np.inf))
        ex_class_names = list(ex_classes.keys())
        if len(ex_class_names) == 0:
            print(f"no text found for generation {ex_uniq_gen_texts} : {class_names}")
            ex_class_names = [class_names[0]]
            ex_classes[class_names[0]] = 0.0
        ex_class_scores = torch.Tensor([ex_classes[k] for k in ex_class_names])
        ex_norm_class_scores = ex_class_scores
        predicted_texts.append(ex_uniq_gen_texts)  # adding list of strings
        predicted_classes.append(ex_class_names)  # adding all classes
        predicted_scores.append(ex_norm_class_scores.detach().numpy())


def comma_split(class_names, shuffle_classes):
    if shuffle_classes:
        random.shuffle(class_names)
    return ", ".join(class_names)


def format_prompts(prompt, texts, class_names, tokenizer, max_seq_length, shuffle_classes=False):
    prompt = get_prompt_to_use(class_names, prompt)
    if len(class_names) == 1 and "yes or no" not in prompt:  # this prompt needs <cls, not cls> as options
        class_names += ["not " + class_names[0]]
    res = []

    prompt_param = {"text": "", "class_names":comma_split }

    # assumption: function written in prompt can relate only to class_names
    param_names = [n[1:-1] for n in re.compile(r'\{[^\{]*\}').findall(prompt)]

    class_names_param = "class_names"

    for name in param_names:
        if name not in ('class_names', 'text'):
            func = globals().get(name, None)
            assert func is not None, f'Function {name} not found in global context'
            prompt_param[name] = func
            if "flan" in name:
                class_names_param = name

    prompt_text = prompt.format(**prompt_param)
    prompt_length = len(tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).input_ids[0])
    remaining_length = max_seq_length-prompt_length
    for txt in texts:
        local_prompt_param = {}
        if remaining_length > 0:
            trimmed_text_tokenized = tokenizer(txt, return_tensors="pt", padding=True, truncation=True).input_ids[0][:remaining_length]
            trimmed_text = tokenizer.decode(trimmed_text_tokenized, skip_special_tokens=True)
        else:
            trimmed_text = txt
        local_prompt_param['text'] = trimmed_text
        local_prompt_param[class_names_param] = prompt_param[class_names_param](class_names, shuffle_classes)
        res.append(prompt.format(**local_prompt_param))

    return res


def get_prompt_to_use(class_names, prompt):
    prompt_value = prompt_config[prompt]
    if type(prompt_value) is str: # single prompt
        prompt = prompt_value
    else:  # 2 prompts, one multiclass and one pairwise
        assert (len(prompt_value) == 2)
        if len(class_names) == 1:
            prompt = prompt_value[1]  # using pairwise prompt
        else:  # using multiclass prompt
            prompt = prompt_value[0]
    return prompt


def run_predict_with_generative_models(input_dir, input_files, model_dir, prompt, batch_size, max_seq_length,
                                       fp16, output_dir, dataset_name, use_lora, use_logits_processor,
                                       ):
    if os.path.exists(os.path.join(output_dir, input_files.split(',')[-1])): # let's use the previous results
        return
    
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, 'metadata.json')) as f:
        metadata = json.load(f)
    class_names = metadata['labels']
    is_multilabel = metadata['multilabel']
    with (open(os.path.join(output_dir, 'metadata.json'), 'tw')) as f:
        json.dump(metadata, f)

    for input_filename in input_files.split(','):
        input_file = os.path.join(input_dir, input_filename)
        df = pd.read_csv(input_file, converters={'label': ast.literal_eval})
        df.sort_values('text', key=lambda x: x.str.len(), inplace=True)
        texts = df['text']

        tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, model_max_length=max_seq_length)
        num_classes = len(class_names)
        class_batch_size = 1 if is_multilabel else num_classes
        predicted_texts = []
        predicted_classes = []
        predicted_scores = []
        start = time.time()
        for begin in range(0, num_classes, class_batch_size):
            class_names_batch = class_names[begin: begin + class_batch_size]
            if class_batch_size < num_classes:
                print(f'class batch {begin // class_batch_size} of {math.ceil(num_classes / class_batch_size)}: {class_names_batch}')
            prompt_batch = format_prompts(
                prompt=prompt,
                texts=texts,
                class_names=class_names_batch,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length
            )
            batch_predicted_texts, batch_names, batch_scores = infer_with_score(
                        texts=prompt_batch,
                        class_names=class_names_batch,
                        model_name=model_dir,
                        tokenizer=tokenizer,
                        batch_size=batch_size,
                        multi_label=is_multilabel,
                        fp16=fp16,
                        use_lora=use_lora,
                        use_logits_processor=use_logits_processor,
                    )

            predicted_texts.append(batch_predicted_texts)
            if class_batch_size == 1:
                predicted_classes.append([class_names_batch[0] for ans in batch_names])
                if "pair_wise" not in prompt: # temporary solution to identify that the options are <cls, not cls>
                    print("finding class name in prediction")
                    predicted_scores.append([score.item() if cls[0] == class_names_batch[0] else 1 - score.item() for cls, score in
                                            zip(batch_names, batch_scores)])
                else:
                    print("finding yes / no in prediction")
                    predicted_scores.append([score.item() if cls[0] == "yes" else 1-score.item() for cls, score in zip(batch_names, batch_scores)])
            else:
                predicted_classes.append(batch_names)
                predicted_scores.append(batch_scores)
        end = time.time()
        runtime = [{"dataset_name":dataset_name, "number of texts":len(texts), "runtime":end-start, "prompt":prompt,
                    "fp16":fp16,"batch_size":batch_size,"max_seq_length":max_seq_length,"class_batch_size":class_batch_size,
                    "num labels":num_classes,
                    "is_multilabel":is_multilabel,"model_dir":model_dir}]
        pd.DataFrame(runtime).to_csv(os.path.join(output_dir,"runtime.csv"))
        # concatenate the results for class name batches, produce scores for all the classes, and store the scores
        predicted_texts_concat = concat_list_batches(predicted_texts) # is needed?
        predicted_names_concat = concat_list_batches(predicted_classes, class_batch_size)
        predicted_scores_concat = concat_list_batches(predicted_scores, class_batch_size)

        # arranging tuples of predicted classes and scores
        predictions_and_scores = arrange_scores(predicted_names_concat, predicted_scores_concat, class_names)
        res = []
        for text, prediction_and_score in zip(texts, predictions_and_scores):
            res.append({'sequence': text, 'labels': [x[0] for x in prediction_and_score], 'scores': [x[1] for x in prediction_and_score]})
        df.loc[:, 'prediction'] = res
        output_file = os.path.join(output_dir, input_filename)
        df.to_csv(output_file, index=False)
        logging.info(f'Wrote {output_file}')


def main():
    args = parser.parse_args()
    
    run_predict_with_generative_models(input_dir=args.input_dir, input_files=args.input_files,
                                       model_dir=args.model_dir, prompt=args.prompt, batch_size=args.batch_size,
                                       max_seq_length=args.max_seq_length, fp16=args.fp16,
                                       output_dir=args.output_dir, dataset_name=args.dataset_name,
                                       use_lora=args.use_lora, use_logits_processor=args.use_logits_processor,
                                       )


if __name__ == '__main__':
    main()
