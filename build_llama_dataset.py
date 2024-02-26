# This file is responsible for taking in a data processed using uskg_gen_dataset.py, and a spec file
# and generating a processed dataset that usable for tuning llama family of models specifically.

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
import torch
from datasets import load_dataset

IGNORE_INDEX = -100
PROMPT_DICT = {
    "inst": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "sysinst": (
        "[INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. "
        "You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output "
        "format, if specified.\n<</SYS>>\n\n{instruction} {input} [/INST]"
    ),
    "sysinst_patch": (
        "[INST] [/INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. "
        "You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output "
        "format, if specified.\n<</SYS>>\n\n{instruction} {input} [/INST] [/INST]"
    ),
    "sysinst_ex": (
        "[INST] <<SYS>>\nYou are an AI assistant that specializes in analyzing and reasoning over structured information. "
        "You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output "
        "format, if specified.\n<</SYS>>\n\n{instruction} {{example}} {input} [/INST]"
    ),
    "base": (
        "[INST] {instruction}\n\n### Input:\n{input}\n\n### Response: [/INST]"
    ),
    "base_1shot": (
        "[INST] Follow the example to complete the following task. {instruction}\n\n{{example}}\n\n### Input:\n{input}\n\n### Response: [/INST]"
    ),
    "inst_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
    "task_label": (
        "Task: {task_label}\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_instruction": (
        "### Input:\n{input}\n\n### Response:\n"
    ),
    "uskg_prompt": (
    "{instruction}; structured knowledge: {input}"
    )
}
all_datasets = ['alpaca', 'tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa']

DATASET_LEVEL_SPLIT = {
    # "wi_we_5shot": {
    #     "train": [],
    #     "test": ["wikitabletext", "webqsp"]
    # },
    # "held_in_inst_orca": {
    #     "train": ['orca', 'tabmwp', 'totto' , 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    # },
    # "held_in_inst_mix": {
    #     "train": ['alpaca', 'tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    # },
    # "all_base_1shot":{
    #     "train": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    # },
    # "orig_held_in_inst_split":{
    #     "train": ['tabmwp', 'totto', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']
    # },
    # "held_in_inst": {
    #     "train": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']
    # },
    # "held_out_inst": {
    #     "train": ['bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    #     "test": ['bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa']
    # },
    # "uskg_multitask": {
    #     "train": ["tabmwp", "cosql", "sqa", "infotabs", "wikitabletext", "finqa"],
    #     "test": ["tabmwp", "cosql", "sqa", "infotabs", "wikitabletext", "finqa"]
    # },
    # "uskg_cosql": {
    #     "train": ["cosql"],
    #     "test": ["cosql"]
    # },
    # "uskg_sqa": {
    #    "train": ["sqa"],
    #     "test": ["sqa"]
    # },
    "all_inst" : {
        "train": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
        "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    },
    # "held_in_tlabels": {
    #     "train": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']
    # },
    # "held_out_tlabels": {
    #     "train": ['bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    #     "test": ['bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa']
    # },
    # "all_tlabels" : {
    #     "train": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    # },
    # # we use fact verification as our testing set because training on it yields the sparsets information. 
    # "table_diff_task_tlabels": {
    #     "train": ['fetaqa', 'hybridqa', 'wikitq', 'tabmwp', 'totto', 'mmqa', 'wikisql', 'kvret', 'tab_fact', 'feverous', 'infotabs'],
    #     "test": ['tab_fact', 'feverous', 'infotabs']
    # },
    # "table_diff_task_inst": {
    #     "train": ['fetaqa', 'hybridqa', 'wikitq', 'tabmwp', 'totto', 'mmqa', 'wikisql', 'kvret'],
    #     "test": ['tab_fact', 'feverous', 'infotabs']
    # },
    # # we use logic 2 text generation to test how well we understand schema, and logic forms
    # "db_diff_task_tlabels": {
    #     "train": ['spider', 'sparc', 'logic2text'],
    #     "test": ['logic2text'],
    # },
    # "db_diff_task_inst": { #knowledge triples
    #     "train": ['spider', 'sparc'],
    #     "test": ['logic2text']
    # },
    # # we choose dart as the summarization task to test understanding of the structure of knowledge triples
    # "kt_diff_task_tlabels":{
    #     "train": ['compwebq', 'webqsp', 'grailqa', 'dart'],
    #     "test": ['dart']
    # },
    # "kt_diff_task_inst": {
    #     "train": ['compwebq', 'webqsp', 'grailqa'],
    #     "test": ['dart']
    # },
    # # we choose dart as the summarization task to test summarization generalization for data to texts
    # "summarization_table_kt_tlabels": {
    #     "train": ['totto', 'dart'],
    #     "test": ['dart']
    # },
    # "summarization_table_kt_inst": {
    #     "train": ['totto'],
    #     "test": ['dart']
    # },
    # # we study wikisql performance because it focuses on extractive answers, like compwebq, which is more comparable.
    # "qa_table_kt_tlabels": {
    #     "train": ['compwebq', 'wikisql'],
    #     "test": ['wikisql']
    # },
    # "qa_table_kt_inst": {
    #     "train": ['compwebq'],
    #     "test": ['wikisql']
    # }
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

def apply_formatting(arg_path2spec, arg_path2name, args, split, batch):
    formatted_strings = []
    for i, d in enumerate(batch):
        spec = arg_path2spec[d['arg_path']]
        if type(spec['instruction']) == list:
            if split == "train":
                instruction = random.choice(spec['instruction'])
            else:
                instruction = spec['instruction'][0] # for test split, we pick one instruction to evaluate the whole dataset.
        else:
            instruction = spec['instruction']
        prompt = PROMPT_DICT[args.prompt_type]
        format_str = prompt.format(instruction=instruction, input=spec['input_format'], task_label = arg_path2name[d['arg_path']])

        # the following line replaces struct_in or text_in as needed, and does not error if the format 
        # string does not contain one of struct_in/text_in. In that case, the non-existent field is simply ignored
        # NOTE the space is extremely important for the llama family of prompts to work correctly
        if split == "train":
            example = ""
        else:
            example = args.examples[d['arg_path']]
        formatted_input = format_str.format(struct_in=d['struct_in'], text_in=d["text_in"], example = example) + " " + d['seq_out']
        formatted_strings.append(formatted_input)
    return formatted_strings

def process_batch(batch, arg_path2spec, arg_path2name, tokenizer, args, split):
    """
    1. format the strings
    2. tokenize the batch
    3. truncate struct if necessary
    """
    # the following line takes the majority of the time.

    formatted_strings = apply_formatting(arg_path2spec, arg_path2name, args, split, batch)

    inp_tokens = tokenizer([s for s in formatted_strings])

    processed_batch = []
    for i, d in enumerate(batch):
        final_formatted_input = formatted_strings[i][:-len(d['seq_out'])]
        if len(d['struct_in']) > 0:
            # if there is a struct, we may need to truncate it
            formatted_input = formatted_strings[i]
            # get the char index of when the struct starts and ends
            struct_start = formatted_input.find(d['struct_in'])
            struct_end = struct_start + len(d['struct_in'])
            input_end = len(formatted_input) - len(" " + d['seq_out'])
            post_struct_token = inp_tokens.char_to_token(i, struct_end)
            start_struct_token = inp_tokens.char_to_token(i, struct_start)
            input_end_token = inp_tokens.char_to_token(i, input_end)
            if input_end_token is None:
                print(f"Warning, output sequence empty {split} split, {d['arg_path']}, omitting this example")
                continue
                # input_end_token = len(inp_tokens.input_ids[i])

            # truncate if the struct is longer than the max allowed length
            # truncation logic:
            # if train split,
            # the formatted input + sequence output must fit within the model_max_length
            # If we exceed the model max length, we truncate the struct portion of the input to fit within the model max length
            # if test split, we don't consider the length of the output sequence, just truncate the input to fit within the model_max_length
            if split == "train":
                diff = max(len(inp_tokens.input_ids[i]) - args.model_max_length, 0)
            else:
                diff = max(input_end_token - args.model_max_length, 0)
            # diff = max(post_struct_token - start_struct_token - args.max_input_length, 0)
            
            # Assume that there is no instance where the amount we need to truncate is at least the length of the struct
            assert diff < post_struct_token - start_struct_token
            struct_end_token = post_struct_token - diff
            # get the strings corresponding to the truncated input_ids
            struct_end = inp_tokens.token_to_chars(i, struct_end_token)[0]
            post_struct = inp_tokens.token_to_chars(i, post_struct_token)[0]
            input_end = inp_tokens.token_to_chars(i, input_end_token)[0]
            new_input = formatted_input[:struct_end] + formatted_input[post_struct:input_end]
            # input_ids = inp_tokens.input_ids[i][:struct_end_token] + inp_tokens.input_ids[i][post_struct_token:input_end_token]
            assert struct_end_token + input_end_token - post_struct_token <= args.model_max_length, f"input_ids length {len(input_ids)} exceeds model max length {args.model_max_length}"
            # remake string
            final_formatted_input = new_input

        if len(d['seq_out']) == 0:
            print(f"Warning, output sequence empty {split} split, {d['arg_path']}, omitting this example")
            continue

        if split == 'train':
            # in train split, we only need formatted_input and output
            datapoint = {
                'formatted_input': final_formatted_input,
                'seq_out': " " + d['seq_out'], # NOTE add space to the beginning of the output sequence
                'arg_path': d['arg_path'],
                'truncated': True if len(d['struct_in']) > 0 and diff > 0 else False
            }
        elif split == 'test':
            # in test split, we need access to the rest of the fields in each data point
            # for use in the uskg evaluators later on. 
            d['formatted_input'] = final_formatted_input
            d['truncated'] = True if len(d['struct_in']) > 0 and diff > 0 else False
            datapoint = d
        processed_batch.append(datapoint)
    return processed_batch


def write_out_splits(args, train_data, test_data):
    
    print("total train dataset size", len(train_data))
    # count the number of times each source appears
    cnts = defaultdict(int)
    truncated = 0
    for d in train_data:
        cnts[d['arg_path']] += 1
        if d['truncated']:
            truncated += 1
    print(cnts)
    print("number of examples structs truncated", truncated)


    print("total test dataset size", len(test_data))

    cnts = defaultdict(int)
    truncated = 0
    for d in test_data:
        cnts[d['arg_path']] += 1
        if d['truncated']:
            truncated += 1
    print(cnts)
    print("number of examples structs truncated", truncated)

    # dump_all_data to a json file
    with open(f'./processed_data/{args.out}_{args.dataset_split}_train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open(f'./processed_data/{args.out}_{args.dataset_split}_test.json', 'w') as f:
        json.dump(test_data, f, indent=4)


def main(args, data):

    # in data, the arg_path field contains the meta_tuning path for the dataset cfg
    # in our instruction tuning spec, each dataset name will be a subset of at least one of
    # the arg_path names. We will try to come up with a mapping from arg_paths to dataset names
    # to look up the prompt format and instructions that we want
    with open(args.held_in_spec) as f:
        held_in_spec = json.load(f)
    with open(args.held_out_spec) as f:
        held_out_spec = json.load(f)

    arg_path2trainspec = {}
    arg_path2evalspec = {}
    arg_path2name = {}
    arg_paths = set()
    for d in data[0]:
        arg_paths.add(d['arg_path'])
    for dname in held_out_spec: # held out spec contains all possible dnames
        matches = 0
        for arg_path in arg_paths:
            if dname in arg_path:
                if dname in held_in_spec:
                    arg_path2trainspec[arg_path] = held_in_spec[dname]
                arg_path2evalspec[arg_path] = held_out_spec[dname]
                arg_path2name[arg_path] = dname
                matches += 1
        assert matches <= 1 # arg_paths and dataset names need to have a one-to-one correspondance.

    # parallelize mapping over the dataset in data[0] using process_batch

    # combine the training and evaluation specs. by adding datasets not in the training spec but are contained in the eval spec
    # this should be harmless, because the held in/held out data filters are done by the DATASET_LEVEL_SPLIT variable.
    for arg_path, spec in arg_path2evalspec.items():
        if arg_path not in arg_path2trainspec:
            arg_path2trainspec[arg_path] = spec


    train_batches = [data[0][i: i+1000] for i in range(0, len(data[0]), 1000)]
    test_batches = [data[2][i: i+1000] for i in range(0, len(data[2]), 1000)]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True, add_eos_token=False)

    partial_process_train_data = partial(process_batch, arg_path2spec=arg_path2trainspec, arg_path2name=arg_path2name, tokenizer=tokenizer, args=args, split="train")
    partial_process_test_data = partial(process_batch, arg_path2spec=arg_path2evalspec, arg_path2name=arg_path2name, tokenizer=tokenizer, args=args, split="test")
    print("starting imap...")
    with multiprocessing.Pool(processes=32) as pool:
        # Use tqdm to create a progress bar
        if len(train_batches) > 0:
            print("creating train dataset")
            train_data = list(tqdm(pool.imap(partial_process_train_data, train_batches), total=len(train_batches)))
        if len(test_batches) > 0:
            print("creating test dataset")
            test_data = list(tqdm(pool.imap(partial_process_test_data, test_batches), total=len(test_batches)))
    # # # do the equivalent as a bove without multiprocessing
    # batch = [batch for batch in train_batches if 'grailqa' in batch[0]['arg_path']][0]
    # process_batch(batch, arg_path2trainspec, arg_path2name, tokenizer, args, split="train")
    # train_data = [process_batch(batch, arg_path2trainspec, arg_path2name, tokenizer, args, split="train") for batch in train_batches]
    # test_data = [process_batch(batch, arg_path2evalspec, arg_path2name, tokenizer, args, split="test") for batch in test_batches]

    # unpack train_data and test_data
    train_data = [item for sublist in train_data for item in sublist]
    test_data = [item for sublist in test_data for item in sublist]

    return train_data, test_data

if __name__ == '__main__':
    # argparse for output file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="tmp/skg_dataset_v12_full.pkl", help="path to the pickle file containing the data split generated by uskg_gen_dataset.py")
    parser.add_argument('--out', type=str, default='v13')
    parser.add_argument('--prompt_type', type=str, default='prompt_input')
    parser.add_argument('--held_in_spec', type=str, default='./prompts/instuning_format_spec.json')
    parser.add_argument('--held_out_spec', type=str, default='./prompts/instuning_format_spec_eval.json')
    parser.add_argument('--tokenizer_path', type=str, default='/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/models/codellama-7b-instruct-hf')
    parser.add_argument('--model_max_length', type=int, default=2048) # determines the max token length of the struct.
    parser.add_argument('--max_output_length', type=int, default=512) # unused for data generation
    parser.add_argument('--examples_file', type=str, default='./prompts/one_shot_gpt_examples.json')
    parser.add_argument('--dataset_split', type=str, default="all", help="dataset level s`plit type to generate jsons for")

    args = parser.parse_args()
    random.seed(1)


    print('reading pkl file')
    data = pickle.load(open(args.data, "rb"))

    # read examples file
    with open(args.examples_file) as f:
        examples = json.load(f)
    # package it into the args object for downstream use
    args.examples = examples

    # with open('./processed_data/formatted_alpaca_cleaned.json') as f:
    #     formatted_alpaca_cleaned = json.load(f)
    with open('./processed_data/formatted_slim_orca.json') as f:
        formatted_slim_orca = json.load(f)
    
    data_lookup = {}
    for prompt_type in ["sysinst"]: #["sysinst", "task_label", "base_1shot"]:
        args.prompt_type = prompt_type
        full_train_data, full_test_data = main(args, data)
        data_lookup[prompt_type] = {"train": defaultdict(list), "test": defaultdict(list)}
        for d in full_train_data:
            data_lookup[prompt_type]["train"][d['arg_path']].append(d)
        # if prompt_type == "inst":
        #     data_lookup[prompt_type]["train"]["alpaca-cleaned"] = formatted_alpaca_cleaned
        if prompt_type == "sysinst":
            data_lookup[prompt_type]["train"]["orca"] = formatted_slim_orca
        for d in full_test_data:
            data_lookup[prompt_type]["test"][d['arg_path']].append(d)



    # args.prompt_type = "inst"
    # full_train_data_inst, full_test_data_inst = main(args, data)

    # args.prompt_type = "task_label"
    # full_train_data_tlab, full_test_data_tlab = main(args, data)

    name2arg_path = {k: None for k in all_datasets}
    # name2arg_path['alpaca'] = 'alpaca-cleaned'
    name2arg_path['orca'] = 'orca'
    arg_paths = set(data_lookup[args.prompt_type]['train'].keys())
    for name in name2arg_path:
        for arg_path in arg_paths:
            if name in arg_path:
                name2arg_path[name] = arg_path
                break

    # tmp = {'all': DATASET_LEVEL_SPLIT['all'], 'held_out': DATASET_LEVEL_SPLIT['held_out']}
    # for split, v in tmp.items():
    for split, v in tqdm(DATASET_LEVEL_SPLIT.items()):
        # gather data for that split, and write it out

        args.dataset_split = split

        train_data = []
        test_data = []
        if "sysinst" in split:
            args.prompt_type = "sysinst"
        elif "tlabels" in split:
            args.prompt_type = "task_label"
        elif "base_1shot" in split:
            args.prompt_type = "base_1shot"
        elif "shot" in split:
            args.prompt_type = "sysinst_ex"
        
        for dname in v['train']:
            train_data.extend(data_lookup[args.prompt_type]['train'][name2arg_path[dname]])
        for dname in v['test']:
            test_data.extend(data_lookup[args.prompt_type]['test'][name2arg_path[dname]])

        write_out_splits(args, train_data, test_data)

    # try:
    #     main(args)
    # except:
    #     import pdb; pdb.post_mortem()
