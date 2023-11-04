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
from utils.dataset import TokenizedDataset
from utils.trainer import EvaluateFriendlySeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm


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

    from filelock import FileLock
    import nltk
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=False)
        nltk.download("stopwords", quiet=False)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)

    os.makedirs(training_args.output_dir, exist_ok=True)

    # The inputs will be train, dev, test or train, dev now.
    # We deprecate the k-fold cross-valid function since it causes too many avoidable troubles.
    
    # check if train, eval, test already exist
    print("starting datset construction")
    if not args.arg_paths:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(path=args.dataset.loader_path,
                                                                        cache_dir=args.dataset.data_store_path)
        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).to_seq2seq(
            raw_datasets_split, cache_root)
    else:
        cache_root = os.path.join('output', 'cache')
        os.makedirs(cache_root, exist_ok=True)
        meta_tuning_data = {}
        for task, arg_path in tqdm(args.arg_paths):
            task_args = Configure.Get(arg_path)
            task_args.bert = args.bert
            print('task_args.bert.location:', task_args.bert.location)
            splits = [f for f in os.listdir(cache_root) if f.startswith(task)]
            if len(splits) == 3:
                placeholder = {'train':DummyDataset(), 'validation':DummyDataset(), 'test':DummyDataset()}
            else:
                placeholder = {'train':DummyDataset(), 'validation':DummyDataset()}

            # note, must have a cache at this point
            task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
                to_seq2seq(placeholder, cache_root)

            meta_tuning_data[arg_path] = task_seq2seq_dataset_split

        seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
            to_seq2seq(meta_tuning_data)
    # save it all into one pickle file
    import pickle
    if training_args.do_train:
        with open(os.path.join("./tmp/skg_dataset_v11_full.pkl"), "wb") as f:
            pickle.dump(seq2seq_dataset_split, f)

if __name__ == "__main__":
    main()
