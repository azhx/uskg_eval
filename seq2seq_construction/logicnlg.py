import os
from copy import deepcopy

import torch
from datasets import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.processor import get_default_processor


# The TabFact dataset is quiet special in the length.

class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


"""
{
    "id": datasets.Value("int32"),
    "table": {
        "id": datasets.Value("string"),
        "header": datasets.features.Sequence(datasets.Value("string")),
        "rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
        "caption": datasets.Value("string"),
    },
    "statement": datasets.Value("string"),
    "label": datasets.Value("int32"),
    "hardness": datasets.Value("string"),
    "small_test": datasets.Value("bool")
}
"""
# setting label=1 when it's entailed, label=0 when it's refuted.
label_id2label_str = {
    1: "entailed",
    0: "refuted"
}


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'logicnlg_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(args,max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            expansion = args.seq2seq.expansion if args.seq2seq.expansion else 1
            for expand_id in range(expansion):
                for raw_data in tqdm(self.raw_datasets):
                    extend_data = deepcopy(raw_data)
                    # This is important to change the question into lower case
                    # since the letter case is handled badly which inconsistency
                    # will cause the unwilling truncation.
                    table_dict = eval(extend_data['table'])
                    header = list(table_dict.keys())
                    rows = [[str(val) for val in each] for each in zip(*table_dict.values())]
                    table_context = {"header": header, "rows": rows}
                    # linearize a table into a string
                    linear_table = self.tab_processor.table_linearize_func.process_table(table_context)
                    extend_data.update({"struct_in": linear_table.lower(),
                                        "text_in": "",
                                        "seq_out": "|".join(extend_data['sentences'])})
                    self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'logicnlg_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.tab_processor = get_default_processor(args,max_cell_length=15,
                                                       tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                       max_input_length=args.seq2seq.table_truncation_max_length)

            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                # This is important to change the question into lower case
                # since the letter case is handled badly which inconsistency
                # will cause the unwilling truncation.
                table_dict = eval(extend_data['table'])
                header = list(table_dict.keys())
                rows = [[str(val) for val in each] for each in zip(*table_dict.values())]
                table_context = {"header": header, "rows": rows}
                # linearize a table into a string
                linear_table = self.tab_processor.table_linearize_func.process_table(table_context)

                extend_data.update({"struct_in": linear_table.lower(),
                                    "text_in": "",
                                    "seq_out": "|".join(extend_data['sentences'])})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets
        self.tab_processor = get_default_processor(args,max_cell_length=15,
                                                   tokenizer=AutoTokenizer.from_pretrained(args.bert.location, use_fast=False),
                                                   max_input_length=args.seq2seq.table_truncation_max_length)

        cache_path = os.path.join(cache_root, 'logicnlg_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                # This is important to change the question into lower case
                # since the letter case is handled badly which inconsistency
                # will cause the unwilling truncation.
                table_dict = eval(extend_data['table'])
                header = list(table_dict.keys())
                rows = [[str(val) for val in each] for each in zip(*table_dict.values())]
                table_context = {"header": header, "rows": rows}
                # linearize a table into a string
                linear_table = self.tab_processor.table_linearize_func.process_table(table_context)

                extend_data.update({"struct_in": linear_table.lower(),
                                    "text_in": "",
                                    "seq_out": "|".join(extend_data['sentences'])}) # HACK LEVEL: EXTREME
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)
