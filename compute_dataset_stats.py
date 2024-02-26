import json
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing
from functools import partial
import argparse
import pandas as pd
# load v11_all_prompt_input_test.json


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
args = parser.parse_args()

with open(f'processed_data/{args.name}_train.json') as f:
#with open('processed_data/v11_all_inst_train.json') as f:
    train_data = json.load(f)

# load v11_all_prompt_input_train.json
with open(f'processed_data/{args.name}_test.json') as f:
# with open('processed_data/v11_all_inst_test.json') as f:
    test_data = json.load(f)

tokenizer = AutoTokenizer.from_pretrained("/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/models/codellama-7b-instruct-hf")

# super fast parallelized tokenization of all inputs and outputs
# batched

train_inp_tokenized = []
train_out_tokenized = []
test_inp_tokenized = []
test_out_tokenized = []

train_batches = [train_data[i: i+1000] for i in range(0, len(train_data), 1000)]
test_batches = [test_data[i: i+1000] for i in range(0, len(test_data), 1000)]

# for batch in tqdm(train_batches):
#     train_inp_tokenized.extend(tokenizer([d['formatted_input'] for d in batch])['input_ids'])
#     train_out_tokenized.extend(tokenizer([d['seq_out'] for d in batch])['input_ids'])

def process_batch(batch, tokenizer):
    inp_batch = tokenizer([d['formatted_input'] for d in batch])['input_ids']
    out_batch = tokenizer([d['seq_out'] for d in batch])['input_ids']
    return inp_batch, out_batch

process_wrapper = partial(process_batch, tokenizer=tokenizer)

with multiprocessing.Pool(processes=16) as pool:
        # Use tqdm to create a progress bar
        # [[inp_batch, out_batch], [inp_batch, out_batch], ...
        train_res = list(tqdm(pool.imap(process_wrapper, train_batches), total=len(train_batches)))
        test_res = list(tqdm(pool.imap(process_wrapper, test_batches), total=len(test_batches)))

train_inp_tokenized, train_out_tokenized = zip(*train_res)
# transform to single list of examples
train_inp_tokenized = [item for sublist in train_inp_tokenized for item in sublist]
train_out_tokenized = [item for sublist in train_out_tokenized for item in sublist]

test_inp_tokenized, test_out_tokenized = zip(*test_res)

test_inp_tokenized = [item for sublist in test_inp_tokenized for item in sublist]
test_out_tokenized = [item for sublist in test_out_tokenized for item in sublist]

print(len(train_inp_tokenized))
print(len(test_inp_tokenized))

print("TRAINING DATASET STATS")
print("====================================")

dataset_names = set([d['arg_path'] for d in train_data])
dstats = {dname: {'maxlen_input': 0, 'maxlen_output': 0, 'avg_inp': 0, 'avg_out': 0, 'num': 0, 'num_exceeding': 0, 'truncated': 0, 'max_inp_example': None, 'max_out_example': None} for dname in dataset_names}

for i, d in tqdm(enumerate(train_data)):
    # log the following 
    # - number of items
    # - max input length
    # - max output length
    # - max input length example
    # - max output length example
    dname = d['arg_path']

    dstats[dname]['num'] += 1

    num_in_tokens = len(train_inp_tokenized[i])
    num_out_tokens = len(train_out_tokenized[i])
    dstats[dname]['avg_inp'] = (dstats[dname]['avg_inp'] * (dstats[dname]['num']-1) + num_in_tokens) / dstats[dname]['num']
    dstats[dname]['avg_out'] = (dstats[dname]['avg_out'] * (dstats[dname]['num']-1) + num_out_tokens) / dstats[dname]['num']
    
    if num_out_tokens > 256:
        dstats[dname]['num_exceeding'] += 1

    if d['truncated']:
        dstats[dname]['truncated'] += 1

    if num_in_tokens > dstats[dname]['maxlen_input']:
        dstats[dname]['maxlen_input'] = num_in_tokens
        dstats[dname]['max_in_example'] = d['formatted_input']

    if num_out_tokens > dstats[dname]['maxlen_output']:
        dstats[dname]['maxlen_output'] = num_out_tokens
        dstats[dname]['max_out_example'] = d['seq_out']

# for dname, stats in dstats.items():
#     print(dname)
#     print("Number of items: {}".format(stats['num']))
#     print("Max input length: {}".format(stats['maxlen_input']))
#     print("Max output length: {}".format(stats['maxlen_output']))
#     print("Num exceeding: {}".format(stats['num_exceeding']))
#     print("% > 256 : {}".format(stats['num_exceeding']/stats['num']))
#     # print("Max input example: {}".format(stats['example']))
#     # print("Max output example: {}".format(stats['max_out_example']))
#     print("====================================")

print("TEST DATASET STATS")
print("====================================")

train_dstats = dstats
dataset_names = set([d['arg_path'] for d in test_data])
dstats = {dname: {'maxlen_input': 0, 'maxlen_output': 0, 'avg_inp': 0, 'avg_out': 0, 'num': 0, 'num_exceeding': 0, 'truncated': 0, 'max_inp_example': None, 'max_out_example': None} for dname in dataset_names}

for i, d in tqdm(enumerate(test_data)):
    dname = d['arg_path']

    dstats[dname]['num'] += 1

    num_in_tokens = len(test_inp_tokenized[i])
    num_out_tokens = len(test_out_tokenized[i])
    dstats[dname]['avg_inp'] = (dstats[dname]['avg_inp'] * (dstats[dname]['num']-1) + num_in_tokens) / dstats[dname]['num']
    dstats[dname]['avg_out'] = (dstats[dname]['avg_out'] * (dstats[dname]['num']-1) + num_out_tokens) / dstats[dname]['num']
    
    if num_out_tokens > 256:
        dstats[dname]['num_exceeding'] += 1

    if d['truncated']:
        dstats[dname]['truncated'] += 1

    if num_in_tokens > dstats[dname]['maxlen_input']:
        dstats[dname]['maxlen_input'] = num_in_tokens
        dstats[dname]['max_in_example'] = d['formatted_input']

    if num_out_tokens > dstats[dname]['maxlen_output']:
        dstats[dname]['maxlen_output'] = num_out_tokens
        dstats[dname]['max_out_example'] = d['seq_out']

# for dname, stats in dstats.items():
#     print(dname)
#     print("Number of items: {}".format(stats['num']))
#     print("Max input length: {}".format(stats['maxlen_input']))
#     print("Max output length: {}".format(stats['maxlen_output']))
#     print("% > 256 : {}".format(stats['num_exceeding']/stats['num']))
#     print("Num exceeding: {}".format(stats['num_exceeding']))
#     # print("Max input example: {}".format(stats['example']))
#     # print("Max output example: {}".format(stats['max_out_example']))
#     print("====================================")

test_dstats = dstats

all_datasets = set(list(train_dstats.keys()) + list(test_dstats.keys()))
# make csv with a row for each dataset in the train set
# columns: dataset_name, avg_inp_len, avg_output_len, train_num_items, train_max_input_length, train_max_output_length, train_pct_truncated, test_num_items, test_max_input_length, test_max_output_length, test_pct_truncated

stats = []
for dataset_name in all_datasets:
    v = train_dstats[dataset_name]
    if dataset_name not in train_dstats:
        v = {'maxlen_input': 0, 'maxlen_output': 0, 'avg_inp': 0, 'avg_out': 0, 'num': 0, 'num_exceeding': 0, 'truncated': 0, 'max_inp_example': None, 'max_out_example': None}
    if dataset_name not in test_dstats:
        test_dstats[dataset_name] = {'maxlen_input': 0, 'maxlen_output': 0, 'avg_inp': 0, 'avg_out': 0, 'num': 0, 'num_exceeding': 0, 'truncated': 0, 'max_inp_example': None, 'max_out_example': None}
    # train_trunc_pct = (v['truncated']/v['num'])*100 if v['num'] > 0 else 0
    # test_trunc_pct = (test_dstats[dataset_name]['truncated']/test_dstats[dataset_name]['num'])*100 if test_dstats[dataset_name]['num'] > 0 else 0
    stats.append([dataset_name, v['avg_inp'], v['avg_out'], v['num'], v['maxlen_input'], v['maxlen_output'], v['truncated'], 
                    test_dstats[dataset_name]['num'], test_dstats[dataset_name]['maxlen_input'], 
                    test_dstats[dataset_name]['maxlen_output'], test_dstats[dataset_name]['truncated']])
df = pd.DataFrame(stats, columns=['dataset_name', 'avg_inp_len', 'avg_output_len', 'train_num_items', 'train_max_input_length', 'train_max_output_length', 'train_pct_truncated', 'test_num_items', 'test_max_input_length', 'test_max_output_length', 'test_pct_truncated'])

df.to_csv(f'processed_data/{args.name}_stats.csv', index=False)
    
