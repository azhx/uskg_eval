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

def linearize_table(table_dict):
    """
    linearize the table in the following format:
    key1 | value1
    key2 | value2
    ....
    """
    colnames = [key for key in table_dict.keys()]
    values = [", ".join(value) for value in table_dict.values()]
    colstr = "col : " + " | ".join(colnames)
    rowstr = "row 1: " + " | ".join(values)
    return colstr + " " + rowstr


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


class TrainDataset(Dataset):

    def __init__(self, args, raw_datasets, cache_root):
        # This tab processor is for table truncation and linearize.
        self.raw_datasets = raw_datasets

        cache_path = os.path.join(cache_root, 'infotabs_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                statement = extend_data["hypothesis"]
                # This is important to change the question into lower case
                # since the letter case is handled badly which inconsistency
                # will cause the unwilling truncation.\
                
                # modify a table internally

                # linearize a table into a string
                linear_table = linearize_table(eval(extend_data['table']))

                extend_data.update({"struct_in": linear_table,
                                    "text_in": statement,
                                    "seq_out": extend_data['label']})
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

        cache_path = os.path.join(cache_root, 'infotabs_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                statement = extend_data["hypothesis"]
                # This is important to change the question into lower case
                # since the letter case is handled badly which inconsistency
                # will cause the unwilling truncation.\
                
                # modify a table internally

                # linearize a table into a string
                linear_table = linearize_table(eval(extend_data['table']))

                extend_data.update({"struct_in": linear_table,
                                    "text_in": statement,
                                    "seq_out": extend_data['label']})
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

        cache_path = os.path.join(cache_root, 'infotabs_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                statement = extend_data["hypothesis"]
                # This is important to change the question into lower case
                # since the letter case is handled badly which inconsistency
                # will cause the unwilling truncation.\
                
                # modify a table internally

                # linearize a table into a string
                linear_table = linearize_table(eval(extend_data['table']))

                extend_data.update({"struct_in": linear_table,
                                    "text_in": statement,
                                    "seq_out": extend_data['label']})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)


    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)