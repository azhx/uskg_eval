import json
import torch
import os
from tqdm import tqdm

cache_dir = "./output/cache"
out_cache_dir = "./output/cache_nl"
dataset_names = ["bird", "logicnlg", "tabmwp", "infotabs", "finqa", "wikitq", "hybridqa", "spider", "fetaqa", "sql2text", "tab_fact", "wikisql", "feverous", "kvret", "sparc", "cosql", "sqa", "mmqa", "mtop", "logic2text", "multiwoz", "totto", "dart"]


# create the output cache dir if it doesn't exist
os.makedirs(out_cache_dir, exist_ok=True)

def replace_row_number(string):
    import re
    pattern = r"row \d+ :"
    new_string = re.sub(pattern, lambda m: "\n" + m.group(0), string)
    return new_string

for dataset_name in tqdm(dataset_names):
    print("Processing dataset", dataset_name)
    # load the caches that have that name as prefix (e.g. bird_train.cache)
    train, dev, test = None, None, None
    if os.path.exists(os.path.join(cache_dir, f"{dataset_name}_train.cache")) and not os.path.exists(os.path.join(out_cache_dir, f"{dataset_name}_train.cache")):
        train = torch.load(os.path.join(cache_dir, f"{dataset_name}_train.cache"))
    if os.path.exists(os.path.join(cache_dir, f"{dataset_name}_dev.cache")) and not os.path.exists(os.path.join(out_cache_dir, f"{dataset_name}_dev.cache")):
        dev = torch.load(os.path.join(cache_dir, f"{dataset_name}_dev.cache"))
    if os.path.exists(os.path.join(cache_dir, f"{dataset_name}_test.cache")) and not os.path.exists(os.path.join(out_cache_dir, f"{dataset_name}_test.cache")):
        test = torch.load(os.path.join(cache_dir, f"{dataset_name}_test.cache"))
    
    # load the dataset and modify the struct in field 
    for split in [train, dev, test]:
        if split is not None:
            for i, ex in enumerate(split):
                if "struct_in" in ex and len(ex["struct_in"]) > 0:
                    split[i]["struct_in"] = replace_row_number(ex["struct_in"])
    # write them back to the output cache dir
    if train is not None:
        torch.save(train, os.path.join(out_cache_dir, f"{dataset_name}_train.cache"))
    if dev is not None:
        torch.save(dev, os.path.join(out_cache_dir, f"{dataset_name}_dev.cache"))
    if test is not None:
        torch.save(test, os.path.join(out_cache_dir, f"{dataset_name}_test.cache"))
    