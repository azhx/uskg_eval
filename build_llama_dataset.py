# build a structured finetuning dataset for llama
# For each dataset, we want to extract ~ 5k examples
# we also want to add a subset of the alpaca instruction finetuning dataset.
# We should include the struct_in and the text_in inside the input, and we should make the instruction task specific for the different types of struct inputs.

# For each dataset, we need to decide which instruction to use. Each dataset has one instruction and struct_in + text_in template. 
# for bird, we need to decide what structured input to use.
# we will hold out cosql and sqa

import json
import random
from tqdm import tqdm
import argparse
import multiprocessing
import pickle
from datasets import Dataset
from collections import defaultdict
import copy
from transformers import AutoTokenizer
from functools import partial



IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    # "prompt_input": (
    #     "[INST] {instruction}\n\n[INPUT] {input} [/INST] "
    # ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "task_label": (
        "Task: {task_label}\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_instruction": (
        "### Input:\n{input}\n\n### Response:\n"
    )
}

# def tokenize_fn(seq_in, seq_out, tokenizer):
#     example = f"{seq_in}{seq_out}{tokenizer.eos_token}"
#     # we don't save this as pytorch tensors at the moment, because we want to save space when saving the dataset
#     tokenized_source = tokenizer(
#         seq_in,
#         truncation=False, # do not truncate, do not pad when we are just building the dataset
#     )
#     tokenized_example = tokenizer(
#         example,
#         truncation=False, # do not truncate, do not pad when we are just building the dataset
#     )
#     input_ids = tokenized_example['input_ids']
#     labels = copy.deepcopy(input_ids)
#     labels[:len(tokenized_source['input_ids'])] = IGNORE_INDEX
#     return dict(input_ids=input_ids, labels=labels)

def process_datapoint(d, arg_path2spec, arg_path2name):
    spec = arg_path2spec[d['arg_path']]
    random.seed(1)
    if type(spec['instruction']) == list:
        instruction = random.choice(spec['instruction'])
    else:
        instruction = spec['instruction']
    prompt = PROMPT_DICT[args.prompt_type]
    format_str = prompt.format(instruction=instruction, input=spec['input_format'], task_label = arg_path2name[d['arg_path']])
    # the following line replaces struct_in or text_in as needed, and does not error if the format 
    # string does not contain one of struct_in/text_in. In that case, the non-existent field is simply ignored
    formatted_input = format_str.format(struct_in=d['struct_in'], text_in=d["text_in"])
    datapoint = {
        'instruction': instruction,
        'input_format': spec['input_format'],
        "struct_in" : d['struct_in'],
        "text_in" : d['text_in'],
        'output': d['seq_out'],
        'source': d['arg_path'],
        'formatted_input': formatted_input,
    }


    return datapoint


def main(args):
    with open(args.spec_file) as f:
        spec = json.load(f)
    data = pickle.load(open(args.data, "rb"))

    #tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)

    # in data, the arg_path field contains the meta_tuning path for the dataset cfg
    # in our instruction tuning spec, each dataset name will be a subset of at least one of
    # the arg_path names. We will try to come up with a mapping from arg_paths to dataset names
    # to look up the prompt format and instructions that we want
    arg_path2spec = {}
    arg_path2name = {}
    arg_paths = set()
    cnts = defaultdict(int)
    for d in data[0]:
        arg_paths.add(d['arg_path'])
    for dname in spec:
        matches = 0
        for arg_path in arg_paths:
            if dname in arg_path:
                arg_path2spec[arg_path] = spec[dname]
                arg_path2name[arg_path] = dname
                matches += 1
        cnts[dname] += 1
        assert matches <= 1 # arg_paths and dataset names need to have a one-to-one correspondance.

    # parallelize mapping over the dataset in data[0] using process_datapoint


    partial_process_data = partial(process_datapoint, arg_path2spec=arg_path2spec, arg_path2name=arg_path2name)
    print("starting imap")
    with multiprocessing.Pool(processes=32) as pool:
        # Use tqdm to create a progress bar
        all_data = list(tqdm(pool.imap(partial_process_data, data[0]), total=len(data[0])))
    
    # save the tokenized dataset to a hf dataset
    # tokenized_data is a list of dicts, each dict has keys input_ids, labels
    # dataset = Dataset.from_dict(tokenized_data)
    # dataset.save_to_disk(f'./{args.out}_tokenized')

    print(cnts)
    print("total dataset size", len(all_data))

    # dump_all_data to a json file
    with open(f'./{args.out}', 'w') as f:
        json.dump(all_data, f, indent=4)



if __name__ == '__main__':
    # argparse for output file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--out', type=str, default='llama_data.json')
    parser.add_argument('--prompt_type', type=str, default='prompt_input')
    parser.add_argument('--spec_file', type=str, default='./instuning_format_spec.json')
    parser.add_argument('--tokenizer_path', type=str, default='/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/codellama/CodeLlama-7b-Instruct-hf')

    args = parser.parse_args()

    main(args)
    # try:
    #     main(args)
    # except:
    #     import pdb; pdb.post_mortem()