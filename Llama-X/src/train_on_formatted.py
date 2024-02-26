#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import torch.distributed
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from datasets import load_dataset
import os
import pickle
from tqdm import tqdm
from bisect import bisect_right
import utils

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    pkl_path: str = field(default="data_v4_input_ids_labels.pkl", 
                          metadata={"help": "Path to the pickled version of tokenized data. Loading from this is faster. Must be in same directory as data_path"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn_2: bool = field(
        default=False,
        metadata={"help": "Whether to use flash attn 2"}   
    )

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def preprocess_fixed(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    input_ids, labels = [], []
    for s, t in zip(sources, targets):
        assert len(t) > 0, f"Empty target for source: {s}"
        tokenized_txt = tokenizer(
            s + t,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        # This line helps us avoid scenarios where s + t != tok(s) + tok(t)
        post_source = tokenized_txt.char_to_token(len(s))
        input_ids.append(tokenized_txt['input_ids'][0])
        label = copy.deepcopy(tokenized_txt['input_ids'][0])
        label[:post_source] = IGNORE_INDEX
        labels.append(label)
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def formatted_train_tokenize_function(examples, tokenizer):
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['seq_out']]
    data_dict = preprocess_fixed(examples['formatted_input'], targets, tokenizer)     
    return data_dict
              
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
        use_flash_attention_2 = training_args.use_flash_attn_2
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    # IMPORTANT NOTE: Do this for models in the llama family.
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )
    base_path = os.path.dirname(os.path.dirname(data_args.data_path))
    full_pkl_path = os.path.join(base_path, data_args.pkl_path)
    # if there is already a tokenized data file, then just load that
    if os.path.exists(full_pkl_path):
        logging.warning("Found tokenized data file, loading...")
        with open(full_pkl_path, "rb") as f:
            train_dataset = pickle.load(f)
            #train_dataset.set_format(type="pt")
    else:
        if training_args.local_rank > 0 or (torch.distributed.is_initialized() and torch.distributed.get_rank() > 0):
            torch.distributed.barrier()
        raw_train_datasets = load_dataset('json', data_files=data_args.data_path, split="train", cache_dir=training_args.cache_dir)

        # batch in 3000 examples
        # try:
        # exampleslst = [each for each in raw_train_datasets]
        # batches = [exampleslst[i:i + 3000] for i in range(0, len(exampleslst), 3000)]
        # train_dataset = []
        # for batch in tqdm(batches):
        #         import pdb; pdb.set_trace()
        #         train_dataset.extend(formatted_train_tokenize_function(batch, tokenizer))
        # # except:
        # #     import pdb; pdb.post_mortem()
        train_dataset = raw_train_datasets.map(
            formatted_train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True, # not args.overwrite_cache
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer}
        )
        
        # save the tokenized dataset to the pickle file
        with open(full_pkl_path, "wb") as f:
            print(f"Saving tokenized dataset to pickle file to {full_pkl_path}")
            pickle.dump(train_dataset, f)

        #train_dataset.set_format(type="pt")

        if training_args.local_rank == 0 and (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
            torch.distributed.barrier()
    
    if training_args.local_rank == 0:
        print(len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(f"Sample {index} of the training set: {train_dataset[index]}.")
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

    #Tell Trainer not to attempt DataParallel
    model.is_parallelizable = True
    model.model_parallel = True

    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    model.config.use_cache = False

    # trainer.predict(train_dataset)
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
