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
from build_llama_dataset import write_out_splits, apply_formatting, process_batch, main


IGNORE_INDEX = -100
# WARNING: only llama family of prompts is valid atm.
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
}
all_datasets = ['alpaca', 'tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa']

DATASET_LEVEL_SPLIT = {
    # "held_in_inst_orca_ratio_2": {
    #     "train": ['tabmwp', 'totto' , 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    # },
    # "held_in_inst_orca_ratio_5": {
    #     "train": ['tabmwp', 'totto' , 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
    # },
    # "held_in_inst_orca_ratio_10": {
    #     "train": ['tabmwp', 'totto' , 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop'],
    #     "test": ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa'],
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
    # "table_diff_task_inst": {
    #     "train": ['fetaqa', 'hybridqa', 'wikitq', 'tabmwp', 'totto', 'mmqa', 'wikisql', 'kvret'],
    #     "test": ['tab_fact', 'feverous', 'infotabs']
    # },
    # "db_diff_task_inst": {
    #     "train": ['spider', 'sparc'],
    #     "test": ['logic2text']
    # },
    # "kt_diff_task_inst": {
    #     "train": ['compwebq', 'webqsp', 'grailqa'],
    #     "test": ['dart']
    # },
    # "summarization_table_kt_inst": {
    #     "train": ['totto'],
    #     "test": ['dart']
    # },
    # "qa_table_kt_inst": {
    #     "train": ['compwebq'],
    #     "test": ['wikisql']
    # }
    "table_diff_task_cotrain": {
        "train": ['fetaqa', 'hybridqa', 'wikitq', 'tabmwp', 'totto', 'mmqa', 'wikisql', 'kvret', 'tab_fact', 'feverous', 'infotabs'],
        "test": ['tab_fact', 'feverous', 'infotabs']
    },
    # we use logic 2 text generation to test how well we understand schema, and logic forms
    "db_diff_task_cotrain": {
        "train": ['spider', 'sparc', 'logic2text'],
        "test": ['logic2text'],
    },
    # # we choose dart as the summarization task to test understanding of the structure of knowledge triples
    "kt_diff_task_cotrain":{
        "train": ['compwebq', 'webqsp', 'grailqa', 'dart'],
        "test": ['dart']
    },
    # # we choose dart as the summarization task to test summarization generalization for data to texts
    "summarization_table_kt_cotrain": {
        "train": ['totto', 'dart'],
        "test": ['dart']
    },
    # # we study wikisql performance because it focuses on extractive answers, like compwebq, which is more comparable.
    "qa_table_kt_cotrain": {
        "train": ['compwebq', 'wikisql'],
        "test": ['wikisql']
    },
    'tabmwp_single_task_ft': {'train': ['tabmwp'], 'test': ['tabmwp']},
    'totto_single_task_ft': {'train': ['totto'], 'test': ['totto']},
    'grailqa_single_task_ft': {'train': ['grailqa'], 'test': ['grailqa']},
    'sql2text_single_task_ft': {'train': ['sql2text'], 'test': ['sql2text']},
    'mmqa_single_task_ft': {'train': ['mmqa'], 'test': ['mmqa']},
    'spider_single_task_ft': {'train': ['spider'], 'test': ['spider']},
    'kvret_single_task_ft': {'train': ['kvret'], 'test': ['kvret']},
    'hybridqa_single_task_ft': {'train': ['hybridqa'], 'test': ['hybridqa']},
    'sparc_single_task_ft': {'train': ['sparc'], 'test': ['sparc']},
    'compwebq_single_task_ft': {'train': ['compwebq'], 'test': ['compwebq']},
    'tab_fact_single_task_ft': {'train': ['tab_fact'], 'test': ['tab_fact']},
    'wikitq_single_task_ft': {'train': ['wikitq'], 'test': ['wikitq']},
    'wikisql_single_task_ft': {'train': ['wikisql'], 'test': ['wikisql']},
    'fetaqa_single_task_ft': {'train': ['fetaqa'], 'test': ['fetaqa']},
    'feverous_single_task_ft': {'train': ['feverous'], 'test': ['feverous']},
    'multiwoz_single_task_ft': {'train': ['multiwoz'], 'test': ['multiwoz']},
    'dart_single_task_ft': {'train': ['dart'], 'test': ['dart']},
    'logic2text_single_task_ft': {'train': ['logic2text'], 'test': ['logic2text']},
    'mtop_single_task_ft': {'train': ['mtop'], 'test': ['mtop']},
    'bird_single_task_ft': {'train': ['bird'], 'test': ['bird']},
    'cosql_single_task_ft': {'train': ['cosql'], 'test': ['cosql']},
    'sqa_single_task_ft': {'train': ['sqa'], 'test': ['sqa']},
    'webqsp_single_task_ft': {'train': ['webqsp'], 'test': ['webqsp']},
    'infotabs_single_task_ft': {'train': ['infotabs'], 'test': ['infotabs']},
    'wikitabletext_single_task_ft': {'train': ['wikitabletext'], 'test': ['wikitabletext']},
    'finqa_single_task_ft': {'train': ['finqa'], 'test': ['finqa']}
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

if __name__ == '__main__':
    # argparse for output file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help="path to the pickle file containing the data split generated by uskg_gen_dataset.py")
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
        if "all_inst" == split:
            continue
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
        

        for dname in v['train']:
            # select a random number of orca examples equal to the number of held_in examples
            train_data.extend(data_lookup[args.prompt_type]['train'][name2arg_path[dname]])
        if "orca_ratio" in split:
            orca = random.sample(data_lookup[args.prompt_type]['train'][name2arg_path['orca']], len(train_data)//int(split.split("_")[-1]))
        else:
            orca = random.sample(data_lookup[args.prompt_type]['train'][name2arg_path['orca']], len(train_data))
        if "orca" in split:
            print(f"number of orca examples {len(orca)}")
            train_data.extend(orca)
        print(f"number of train examples {len(train_data)}")

        for dname in v['test']:
            test_data.extend(data_lookup[args.prompt_type]['test'][name2arg_path[dname]])

        write_out_splits(args, train_data, test_data)

    # try:
    #     main(args)
    # except:
    #     import pdb; pdb.post_mortem()