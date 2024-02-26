import os
from utils.configue import Configure
import utils
import datasets
cache_root = os.path.join('output', 'cache')
os.makedirs(cache_root, exist_ok=True)
meta_tuning_data = {}
# arg_paths = ['cosql_with_cell.cfg',                                                                                                                                            
#             'tab_fact.cfg',                                                                                                                                                                                                                                                                                                 
#             'sqa.cfg',                                                                                                                                                        
#             'sparc_with_cell.cfg',                                                                                                                                                       
#             'sql2text.cfg',            
#             'totto.cfg',               
#             'kvret.cfg',               
#             'wikitq.cfg',             
#             'hybridqa.cfg',         
#             'wikisql.cfg',                                                                                                                                                    .
#             'spider_with_cell.cfg', 
#             'dart.cfg',      
#             'logic2text.cfg',                                                                                                                                                 o
#             'feverous.cfg',         
#             'mtop.cfg',                                                                                                                                                       $
#             'multiwoz.cfg',                        
#             'mmqa.cfg',                            
#             'fetaqa.cfg']
arg_paths = ['finqa.cfg']



for task, arg_path in arg_paths:
    task_args = Configure.Get(arg_path)
    # task_args.add_newline = True
    task_args.bert = args.bert
    print('task_args.bert.location:', task_args.bert.location)
    task_raw_datasets_split: datasets.DatasetDict = datasets.load_dataset(
        path=task_args.dataset.loader_path,
        cache_dir=task_args.dataset.data_store_path)
    task_seq2seq_dataset_split: tuple = utils.tool.get_constructor(task_args.seq2seq.constructor)(task_args).\
        to_seq2seq(task_raw_datasets_split, cache_root)

    meta_tuning_data[arg_path] = task_seq2seq_dataset_split

seq2seq_dataset_split: tuple = utils.tool.get_constructor(args.seq2seq.constructor)(args).\
    to_seq2seq(meta_tuning_data)