import datasets
import argparse
import os
from utils.configue import Configure
import utils.tool
from utils.dataset import TokenizedDataset, TokenizedLlamaDataset
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from utils.trainer import LlamaSeq2SeqTrainer



# Argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, help='dataset name')
parser.add_argument('--run_name', type=str, help='run name')
parser.add_argument('--cache_dir', type=str, default="cache", help='path to seq2seq cfg')
training_args = parser.parse_args()

# Load dataset
dataset_raw_split = datasets.load_dataset(path = f"./tasks/{training_args.dataset_name}.py", cache_dir = "./data")

#
args = Configure.Get(f"Salesforce/{training_args.run_name}_{training_args.dataset_name}.cfg")

cache_root = os.path.join('output', training_args.cache_dir)
os.makedirs(cache_root, exist_ok=True)
meta_tuning_data = {}
for task, arg_path in args.arg_paths:
    task_args = Configure.Get(arg_path)
    task_args.bert = args.bert
    task_args.add_newline = args.add_newline
    print('task_args.bert.location:', task_args.bert.location)
    task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
        path=task_args.dataset.loader_path,
        cache_dir=task_args.dataset.data_store_path)
    task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
        to_seq2seq(task_raw_datasets_split, cache_root)

    meta_tuning_data[arg_path] = task_seq2seq_dataset_split

seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
    to_seq2seq(meta_tuning_data)

print("dataset constructed")
import pdb; pdb.set_trace()

# evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
# model = utils.tool.get_model(args.model.name)(args)
# model_tokenizer = model.tokenizer
# if not model_tokenizer.pad_token:
#     model_tokenizer.pad_token = model_tokenizer.eos_token

# seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
# if len(seq2seq_dataset_split) == 1:
#     seq2seq_eval_dataset = seq2seq_dataset_split[0]
# if len(seq2seq_dataset_split) == 2:
#     seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
# elif len(seq2seq_dataset_split) == 3:
#     seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
# else:
#     raise ValueError("Other split not support yet.")

# # We wrap the "string" seq2seq data into "tokenized tensor".
# train_dataset = TokenizedLlamaDataset(args, training_args, model_tokenizer,
#                                     seq2seq_train_dataset) if seq2seq_train_dataset else None

# eval_dataset = TokenizedLlamaDataset(args, training_args, model_tokenizer,
#                                 seq2seq_eval_dataset) if seq2seq_eval_dataset else None
# test_dataset = TokenizedLlamaDataset(args, training_args, model_tokenizer,
#                                 seq2seq_test_dataset) if seq2seq_test_dataset else None

# # Initialize our Trainer
# early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 5)
# trainer = LlamaSeq2SeqTrainer(
#     args=training_args,
#     model=model,
#     evaluator=evaluator,
#     # We name it "evaluator" while the hugging face call it "Metric",
#     # they are all f(predictions: List, references: List of dict) = eval_result: dict
#     tokenizer=model_tokenizer,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     eval_examples=seq2seq_eval_dataset,
#     callbacks=[early_stopping_callback],
# )
