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
import numpy as np
import math
from transformers import LlamaTokenizer
from copy import deepcopy
from random import shuffle


def compute_upsampling_weights(datasets, temp=2):
    # we follow unifiedskg's approach to compute upsampling 
    # https://github.com/HKUNLP/UnifiedSKG/blob/main/seq2seq_construction/meta_tuning.py#L55
    # we also follow https://huggingface.co/hkunlp/T5_large_prefix_all_tasks_2upsample2 to have an upsample temp of 2

    dataset_sizes = {}
    for path, data in datasets.items():
        dataset_sizes[path] = len(data)

    # Compute resampling weights.
    dataset_upsampling_factors = {}
    sum_tau_size = sum([np.exp(np.log(size) / temp) for size in dataset_sizes.values()])
    sum_size = sum(dataset_sizes.values())
    for path, size in dataset_sizes.items():
        tau_size = np.exp(np.log(size) / temp)
        dataset_upsampling_factors[path] = tau_size / sum_tau_size * sum_size / size

    # Compute upsampling weights.
    largest_path, _ = max(dataset_sizes.items(), key=lambda x: x[1])
    norm_coef = dataset_upsampling_factors[largest_path]
    for path in dataset_upsampling_factors.keys():
        dataset_upsampling_factors[path] = dataset_upsampling_factors[path] / norm_coef
    
    return dataset_upsampling_factors

def upsample(data, weight):
    # also following UnifiedSKG
    n_data = len(data)
    assert weight >= 1

    integral = list(range(n_data)) * int(math.floor(weight))
    residual = list(range(n_data))
    shuffle(residual)
    residual = residual[:int(n_data * (weight - int(math.floor(weight))))]
    return [deepcopy(data[idx]) for idx in integral + residual]


def build_input(strut, text, tokenizer):
    return tokenizer(strut)['input_ids'] + tokenizer(text)['input_ids']

def main(args):
    with open('./instuning_format_spec_rs.json') as f:
        spec = json.load(f)
    held_out_datasets = ['cosql', 'sqa', 'bird', 'grailqa', 'webqsp', 'compwebq']
    # alpaca_data_path = './alpaca_data.json'
    dataset_json_path = [(f"./ukg_data/{key}_train.json", v)  for key, v in spec.items() if key not in held_out_datasets]

    # load alpaca data
    # with open(alpaca_data_path) as f:
    #     alpaca_data = json.load(f)

    all_data = []
    random.seed(1)

    #tokenizer = LlamaTokenizer.from_pretrained('/mnt/tjena/alex/llama/Llama-2-7b-hf')

    datasets = {}
    dataset_spec = {}

    # list number of examples in each dataset train split
    for path, spec in tqdm(dataset_json_path):
        with open(path) as f:
            data = json.load(f)
        datasets[path] = data
        dataset_spec[path] = spec

    upsampling_weights = compute_upsampling_weights(datasets)

    for path, data in tqdm(datasets.items()):
        # randomly select 5k examples
        # random.shuffle(data)
        cur_dataset = []
        spec = dataset_spec[path]
        for i, d in enumerate(data):
            if type(spec['instruction']) == list:
                # choose a random instruction
                instruction = random.choice(spec['instruction'])
            else:
                instruction = spec['instruction']
            # if (len(d['struct_in']) == 0):
            #     # if we there is no input strcture, we don't care about this example. We only want to learn how to do inference based on structured data.
            #     continue
            datapoint = {
                'instruction': instruction,
                'input_format': spec['input_format'],
                "struct_in" : d['struct_in'],
                "text_in" : d['text_in'],
                'output': d['seq_out'],
                'source': path,
                'id': f"{path}_{i}",
            }
            # datapoint['input_len'] = len(tokenizer(datapoint['input'])['input_ids'])
            # if (datapoint['input_len'] > 512):
            #     continue
            cur_dataset.append(datapoint)

        upsampled_dataset = upsample(cur_dataset, upsampling_weights[path])
        all_data += upsampled_dataset

        print(f"{path}: {len(cur_dataset)}")
        print(f"upsampled by weight {upsampling_weights[path]} to {len(upsampled_dataset)}")
    
    # # shuffle alpaca data
    # random.shuffle(alpaca_data)
    # for i, d in enumerate(alpaca_data[:10000]):
    #     all_data.append(d)
    #     d['source'] = './alpaca_data.json'
    #     d['id'] = f"alpaca_{i}"

    print("total dataset size", len(all_data))

    # dump_all_data to a json file
    with open(f'./{args.out}', 'w') as f:
        json.dump(all_data, f, indent=4)


if __name__ == '__main__':
    # argparse for output file name
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='llama_data.json')
    args = parser.parse_args()

    try:
        main(args)
    except:
        import pdb; pdb.post_mortem()