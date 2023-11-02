import logging
import os
import time

import torch
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from collections import OrderedDict
import utils.tool
from utils.configue import Configure
from utils.dataset import TokenizedDataset, TokenizedLlamaDataset
from utils.trainer import LlamaSeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)

# class with a getitem
class DummyDataset():
    def __getitem__(self, index):
        return {}

def main() -> None:
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    torch.use_deterministic_algorithms(True)
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)
    args.llama.url = f"http://localhost:{training_args.port}"

    # The inputs will be train, dev, test or train, dev now.
    # We deprecate the k-fold cross-valid function since it causes too many avoidable troubles.

    cache_root = os.path.join('output', 'cache')
    os.makedirs(cache_root, exist_ok=True)
    # find all files in the cache root that start with the dataset name
    splits = [f for f in os.listdir(cache_root) if f.startswith(args.prompt_spec.dataset_name)]
    if len(splits) == 3:
        placeholder = {'train':DummyDataset(), 'validation':DummyDataset(), 'test':DummyDataset()}
    else:
        placeholder = {'train':DummyDataset(), 'validation':DummyDataset()}
    meta_tuning_data = {}
    for task, arg_path in args.arg_paths:
        task_args = Configure.Get(arg_path)
        task_args.bert = args.bert
        # add newline for tables can come from training args or task (experiment) cfg args
        task_args.add_newline = args.add_newline
        if training_args.add_newline:
            task_args.add_newline = True
        print('task_args.bert.location:', task_args.bert.location)
        task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
            to_seq2seq(placeholder, cache_root)

        meta_tuning_data[arg_path] = task_seq2seq_dataset_split

    seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
        to_seq2seq(meta_tuning_data)

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    model = utils.tool.get_model(args.model.name)(args)
    model_tokenizer = model.tokenizer
    if not model_tokenizer.pad_token:
        model_tokenizer.pad_token = model_tokenizer.eos_token

    seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = None, None, None
    if len(seq2seq_dataset_split) == 2:
        seq2seq_train_dataset, seq2seq_eval_dataset = seq2seq_dataset_split
    elif len(seq2seq_dataset_split) == 3:
        seq2seq_train_dataset, seq2seq_eval_dataset, seq2seq_test_dataset = seq2seq_dataset_split
    else:
        raise ValueError("Other split not support yet.")

    # We wrap the "string" seq2seq data into "tokenized tensor".
    train_dataset = TokenizedLlamaDataset(args, training_args, model_tokenizer,
                                     seq2seq_train_dataset, args.prompt_spec.few_shot_path) if seq2seq_train_dataset else None
    eval_dataset = TokenizedLlamaDataset(args, training_args, model_tokenizer,
                                    seq2seq_eval_dataset, args.prompt_spec.few_shot_path) if seq2seq_eval_dataset else None
    test_dataset = TokenizedLlamaDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset, args.prompt_spec.few_shot_path) if seq2seq_test_dataset else None
    
    # Initialize our Trainer
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=args.seq2seq.patience if args.seq2seq.patience else 5)
    trainer = LlamaSeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
        tokenizer=model_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=seq2seq_eval_dataset,
        callbacks=[early_stopping_callback],
    )
    print('Trainer build successfully.')


    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            test_dataset=test_dataset if test_dataset else eval_dataset,
            test_examples=seq2seq_test_dataset if seq2seq_test_dataset else seq2seq_eval_dataset,
            metric_key_prefix="predict"
        )

if __name__ == "__main__":
    main()