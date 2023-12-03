import numpy as np
import torch
import random
import os

from transformers import AutoTokenizer

def set_torch_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Torch Random seed set as {seed}")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def get_api_key():
    key = os.environ.get('BAM_API_KEY')
    if not key:
        return os.environ.get('SUPER_BAM_KEY_0')
    return key

def get_tokenizer(name, model_max_length):
    return AutoTokenizer.from_pretrained(name, model_max_length=model_max_length)


def cut_text(txt, max_available_tokens, tokenizer):
    result = txt
    tkns = tokenizer(txt)
    if len(tkns['input_ids']) > max_available_tokens:
        last_fitting_token_span = tkns[0].token_to_chars(
            max_available_tokens - 1 - 1)  # span of the last fitting token, not including the end of input
        end_of_last_fitting_token = last_fitting_token_span[1]
        # end of the last fitting token
        new_text = txt[0:end_of_last_fitting_token]
        if not txt.startswith(new_text):
            for i in range(len(new_text)):
                if new_text[i] != txt[i]:
                    print(
                        f"Problem {i}, {new_text[0:i + 1]}, {txt[0:i + 1]},")
                    break
        result = new_text
        if len(tokenizer(result)['input_ids']) != max_available_tokens:
            print(
                f"Got  {len(tokenizer(result)['input_ids'])} tokens instead of {max_available_tokens}")
    return result


def cut_texts_for_prompt(texts, prompt, max_text_len, tokenizer):
    #prompt_len = len(tokenizer(prompt)["input_ids"])
    prompt_len = max(len(tokenizer(prompt)["input_ids"]), 150) # TODO decide, how to implement it correctly  

    txt_limit = max_text_len - prompt_len
    assert txt_limit > 0

    new_texts = [cut_text(t, txt_limit, tokenizer) for t in texts]

    return new_texts


model_to_token_limit = {'bigscience/bloom': 4096,
 'salesforce/codegen2-16b': 2048,
 'prakharz/dial-flant5-xl': 2048,
 'tiiuae/falcon-40b': 8192,
 'google/flan-t5-xl': 4096,
 'google/flan-t5-xxl': 4096,
 'google/flan-ul2': 4096,
 'eleutherai/gpt-neox-20b': 8192,
 'togethercomputer/gpt-neoxt-chat-base-20b': 8192,
 'meta-llama/llama-2-13b': 4096,
 'meta-llama/llama-2-13b-chat': 4096,
 'meta-llama/llama-2-13b-chat-beam': 4096,
 'meta-llama/llama-2-70b': 4096,
 'meta-llama/llama-2-70b-chat': 4096,
 'meta-llama/llama-2-7b': 4096,
 'meta-llama/llama-2-7b-chat': 4096,
 'mosaicml/mpt-30b': 2048,
 'ibm/mpt-7b-instruct': 2048,
 'bigscience/mt0-xxl': 4096,
 'openassistant/oasst-sft-4-pythia-12b': 8192,
 'bigcode/starcoder': 8192,
 'google/ul2': 4096}
genai_model_type_to_huggingface = {
            "salesforce/codegen-16b-mono" : "Salesforce/codegen-16B-mono",
            "prakharz/dial-flant5-xl" : "prakharz/DIAL-FLANT5-XL",
            "togethercomputer/gpt-jt-6b-v1" : "togethercomputer/GPT-JT-6B-v1",
            "eleutherai/gpt-neox-20b/gpt-jt-6b-v1" : "EleutherAI/gpt-neox-20b",
        }
