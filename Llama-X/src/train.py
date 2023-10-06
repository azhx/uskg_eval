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
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"
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
    "prompt_no_instruction": (
        "### Input:\n{input}\n\n### Response:\n"
    )
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    pkl_path: str = field(default="data_v4_input_ids_labels.pkl", 
                          metadata={"help": "Path to the pickled version of tokenized data. Loading from this is faster. Must be in same directory as data_path"})
    has_instruction: bool = field(default=True, metadata={"help": "Whether we should use instructions for this dataset."})
    dataset_type: str = field(default="llama", metadata={"help": "Type of dataset. [llama, skg]"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
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

def skg_preprocess_and_tokenize(example, format_str, tokenizer):
    if example["text_in"]:
        pre_truncation = format_str.format(struct_in=example['struct_in'], text_in=example["text_in"]) + example["output"]
    else:
        pre_truncation = format_str.format(struct_in=example['struct_in']) + example["output"]

    # get the char index of when the struct starts and ends
    struct_start = pre_truncation.find(example['struct_in']) 
    struct_end = struct_start + len(example['struct_in'])
    output_start = pre_truncation.find(example['output'])

    if "xgen" in str(type(tokenizer)):            # if we are using xgen, we need to manually calculate the offsets
        seq_in = tokenizer(pre_truncation, return_tensors="pt")
        _, offsets = tokenizer.decode_with_offsets(seq_in['input_ids'][0])
        output_start_token = bisect_right(offsets, output_start)
        post_struct_token = bisect_right(offsets, struct_end)
        start_struct_token = bisect_right(offsets, struct_start)
    else:
        seq_in = tokenizer(pre_truncation, return_tensors="pt")
        output_start_token = seq_in.char_to_token(output_start)
        post_struct_token = seq_in.char_to_token(struct_end)
        start_struct_token = seq_in.char_to_token(struct_start)
        assert seq_in.input_ids[:, -1] == tokenizer.eos_token_id


    diff = max(seq_in.input_ids.shape[1] - tokenizer.model_max_length, 0)
    
    if (post_struct_token - start_struct_token <= diff and diff > 0):
        # if there is a struct and this is the case, we would have completely truncated the struct! 
        # If this is the case, we should just completely throw away the example
        # or, there is no struct, but the output is too long, so we should throw away the example
        return None
    
    struct_end_token = post_struct_token - diff
    input_ids = torch.concat((seq_in.input_ids[0,:struct_end_token], seq_in.input_ids[0,post_struct_token:]), dim=0)
    labels = input_ids.clone() # will modify this soon
    # truncating the struct by an offset of diff means the output will be shifted diff tokens earlier
    labels[:output_start_token - diff] = IGNORE_INDEX        
    assert len(input_ids) == len(labels) <= tokenizer.model_max_length
    return dict(input_ids = input_ids, labels = labels)

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

def train_tokenize_function(examples, tokenizer):
    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    if 'input' in examples:
        sources = [
            prompt_input.format_map(dict(instruction=instruction, input=input)) if input != "" \
            else prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction, input in zip(examples['instruction'], examples['input']) 
        ]
    else:
        sources = [
            prompt_no_input.format_map(dict(instruction=instruction)) \
            for instruction in examples['instruction']
        ]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def skg_train_tokenize_function(examples, tokenizer, has_instruction):
    prompt_input, prompt_no_input, prompt_no_instr = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"], PROMPT_DICT["prompt_no_instruction"]
    input_ids = []
    labels = []
    # convert dict of lists to a list of dicts
    batch = [dict(zip(examples, t)) for t in zip(*examples.values())]
    for i, example in enumerate(batch):
        if has_instruction:
            format_str = prompt_input.format(instruction=example['instruction'], input=example['input_format'])
            data_dict = skg_preprocess_and_tokenize(example, format_str, tokenizer)
        else:
            # In SKG, we always have inputs, but sometimes no instructions. In Alpaca, 
            # we may have instructions with no input but just response
            # if we are using prompt_no_instr, we are tuning exclusively on SKG data, which always has inputs
            format_str = prompt_no_instr.format(input=example['input_format'])
            data_dict = skg_preprocess_and_tokenize(example, format_str, tokenizer)
        if data_dict is None:
            # warning
            logging.warning(f"Example {i} would have had its whole struct truncated, skipping...")
        else:
            input_ids.append(data_dict['input_ids'])
            labels.append(data_dict['labels'])
    if len(input_ids) == 0:
        return None
    data_dict = dict(input_ids=input_ids, labels=labels)
    return data_dict
              
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        add_eos_token=True,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )
    data_path_dir = os.path.dirname(data_args.data_path)
    full_pkl_path = os.path.join(data_path_dir, data_args.pkl_path)
    # if there is already a tokenized data file, then just load that
    if os.path.exists(full_pkl_path):
        logging.warning("Found tokenized data file, loading...")
        with open(full_pkl_path, "rb") as f:
            train_dataset = pickle.load(f)
            #train_dataset.set_format(type="pt")
    else:
        raw_train_datasets = load_dataset('json', data_files=data_args.data_path, split="train", cache_dir=training_args.cache_dir)
        if training_args.local_rank > 0: 
            torch.distributed.barrier()

        train_dataset = raw_train_datasets.map(
            skg_train_tokenize_function,
            batched=True,
            batch_size=3000,
            num_proc=32,
            remove_columns=raw_train_datasets.column_names,
            load_from_cache_file=True, # not args.overwrite_cache
            desc="Running tokenizer on train dataset",
            fn_kwargs={"tokenizer": tokenizer, "has_instruction": data_args.has_instruction}
        )

        train_dataset = train_dataset.filter(
            lambda example: example is not None
        )

        # save the tokenized dataset to the pickle file
        with open(full_pkl_path, "wb") as f:
            print(f"Saving tokenized dataset to pickle file to {full_pkl_path}")
            pickle.dump(train_dataset, f)

        #train_dataset.set_format(type="pt")

        if training_args.local_rank == 0:
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

    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
