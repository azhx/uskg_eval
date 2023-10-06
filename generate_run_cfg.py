# generate toml config file for running UnifiedSKG

import argparse
import os
import tomlkit

def generate_run_cfg(args):
    # generate yaml config file for running UnifiedSKG
    # args: arguments from command line
    
    # create config file
    config = {}

    # model
    config['model'] = {}
    config['model']['name'] = args.model_name
    config['model']['use_description'] = args.use_description
    config['model']['concatenate_description'] = args.concatenate_description
    config['model']['knowledge_usage'] = args.knowledge_usage

    # dataset
    config['dataset'] = {}
    config['dataset']['data_store_path'] = args.data_store_path
    
    # seq2seq
    config['seq2seq'] = {}
    config['seq2seq']['constructor'] = args.constructor
    config['seq2seq']['patience'] = args.patience

    # arg_paths
    config['arg_paths'] = {}
    config['arg_paths'][args.dataset_name] = args.arg_path

    # evaluate
    config['evaluate'] = {}
    config['evaluate']['tool'] = args.tool

    # special_tokens Note: legacy
    config['special_tokens'] = {}
    config['special_tokens']['less'] = ' <'
    config['special_tokens']['less_or_equal'] = ' <='

    # bert Note: legacy
    config['bert'] = {}
    config['bert']['location'] = 't5-3b'

    # llama
    config['llama'] = {}
    #config['llama']['url'] = args.api_url
    config['llama']['model_path'] = args.model_path

    # prompt_spec
    config['prompt_spec'] = {}
    config['prompt_spec']['dataset_name'] = args.dataset_name
    config['prompt_spec']['path'] = args.prompt_spec_path

    # debug
    config['debug'] = {}
    config['debug']['dump_preds'] = args.dump_preds

    # write config file
    with open(args.output_path, 'w') as f:
        tomlkit.dump(config, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate yaml config file for running UnifiedSKG')

    parser.add_argument('--output_path', type=str, default='run_cfg.yaml', help='output name')
    parser.add_argument('--run_name', type=str, default='llama2_v4e5', help='run name')

    parser.add_argument('--arg_path', type=str, default='META_TUNING/spider_with_cell.cfg', help='arg path')
    parser.add_argument('--dataset_name', type=str, default='spider', help='dataset name')
    #parser.add_argument('--api_url', type=str, default='http://localhost:8090', help='api url')
    parser.add_argument('--model_path', type=str, default='/mnt/tjena/alex/vaughan/ckpt7', help='api model path')


    parser.add_argument('--model_name', type=str, default='unified.llama_api', help='model name')
    parser.add_argument('--use_description', type=bool, default=False, help='use description')
    parser.add_argument('--concatenate_description', type=bool, default=False, help='concatenate description')
    parser.add_argument('--knowledge_usage', type=str, default='concatenate', help='knowledge usage')
    parser.add_argument('--data_store_path', type=str, default='./data', help='data store path')
    parser.add_argument('--constructor', type=str, default='seq2seq_construction.meta_tuning', help='constructor')
    parser.add_argument('--patience', type=int, default=200, help='patience')
    parser.add_argument('--tool', type=str, default='metrics.meta_tuning.no_evaluation', help='tool')
    parser.add_argument('--prompt_spec_path', type=str, default='/home/alex/v3-score/instuning_format_spec_eval_rs.json', help='path')
    parser.add_argument('--dump_preds', type=bool, default=True, help='dump preds')
    args = parser.parse_args()

    ### HARDCODING section
    #TODO undo this hack later if/when we need more general functionality
    dataset_names = ['bird', 'logicnlg', 'tabmwp', 'finqa', 'infotabs', 'totto', 'webqsp', 'sqa', 'sql2text', 'spider', 'cosql', 'kvret', 'hybridqa', 'sparc', 'grailqa', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'mmqa', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']
    datasets = ['bird', 'logicnlg', 'tabmwp', 'finqa', 'infotabs', 'totto', 'webqsp', 'sqa', 'sql2text', 'spider_with_cell', 'cosql_with_cell', 'kvret', 'hybridqa', 'sparc_with_cell', 'grailqa', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'mmqa', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']

    server1 = ["spider", "compwebq", "fetaqa", "hybridqa", "mmqa", "sql2text", "dart", "totto", "wikitq"]
    server2 = ["cosql", "sparc", "tab_fact", "wikisql", "feverous", "kvret", "mtop"]
    server3 = ["sqa", "logic2text", "multiwoz", "webqsp", "grailqa"]
    servers = [server1, server2, server3]
    servermap = {}
    for i, server in enumerate(servers):
        for j, dataset in enumerate(server):
            servermap[dataset] = i
            
    datasets.sort()
    dataset_names.sort()
    #api_urls = ["http://localhost:8090", "http://localhost:8091", "http://localhost:8092"]
    # add .cfg to the end
    datasets = [f"META_TUNING/{d}.cfg" for d in datasets]
    assert len(dataset_names) == len(datasets)

    for i, dataset in enumerate(datasets):
        args.dataset_name = dataset_names[i]
        args.arg_path = dataset
        #args.api_url = api_urls[servermap[args.dataset_name]]
        args.output_path = f"/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/configure/Salesforce/{args.run_name}_{args.dataset_name}.cfg"
        generate_run_cfg(args)

