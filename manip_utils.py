


def llama_dataset_summary(path):
    # read the path to a json file
    import json
    from collections import defaultdict
    with open(path, "r") as f:
        data = json.load(f)

    # summary should include
    # number of dataset, with the number of examples in each dataset
    # number of examples in total
    # printout of one example, 

    print("Name: ", path)
    print("Number of examples: ", len(data))
    cnts = defaultdict(int)
    for example in data:
        cnts[example['arg_path']] += 1
    print("Number of datasets: ", len(cnts))
    print("By Dataset: ")
    for k, v in cnts.items():
        print(k, v)
    print("Example: ")
    if data:
        print(data[0]['formatted_input'])
        print(data[0]['seq_out'])

def view_gpt_responses(path):
    import json
    # load jsonl file from path
    with open(path, "r") as f:
        data = [json.loads(each) for each in f.readlines()]
    # sort by the key "task_id"
    data = sorted(data, key=lambda x: x['task_id'])
    # printout the prompts and response
    for each in data:
        print(each['response'][0]['messages'][0]['content'])
        print(each['response'][1]['choices'][0]['message']['content'])
        print("\n\n\n")
        print("============================================================================")
        print("\n\n\n")

def predictions_summary(run_name, dataset):
    import json
    with open(f"./output/{run_name}/predictions_predict.json", "r") as f:
        preds = json.load(f)
    
    pass

def qa_dataset_examples(path):
    # read the path to a json file
    import json
    from collections import defaultdict
    with open(path, "r") as f:
        data = json.load(f)

    # summary should include
    # number of dataset, with the number of examples in each dataset
    # number of examples in total
    # printout of one example, 
    ds = {}
    for example in data:
        if example['arg_path'] not in ds:
            ds[example['arg_path']] = example
    
    for k, v in ds.items():
        print(k)
        print(v['formatted_input'] + v['seq_out'])
        print("\n\n\n")
        print("============================================================================")
        print("\n\n\n")


def get_n_examples(dataset, ds_name, n=10):

    desired_dataset = [each for each in dataset if ds_name in each['arg_path']]
    predictions_and_answers = [(each['prediction'], each['seq_out']) for each in desired_dataset]
    # select random index
    return predictions_and_answers

def start_qa_routine(path):
    import json
    from collections import defaultdict
    import random
    from tqdm import tqdm
    with open(path, "r") as f:
        data = json.load(f)
    datasets = defaultdict(list)
    for each in data:
        datasets[each['arg_path']].append(each)
    
    for k, v in tqdm(datasets.items()):
        # show 5 random examples
        examples = random.sample(v, 5)
        for each in examples:
            print("Dataset:", k)
            print("============================================================================")
            # wait for user input before moving to next one
            print(each['formatted_input'] + each['seq_out'])
            print("\n\n\n")
            input("Press Enter to continue...")

def token_qa_routine(path, tokenpath, tokenizer_path):
    import json
    import pickle
    from collections import defaultdict
    from transformers import AutoTokenizer
    import random

    tokens = pickle.load(open(tokenpath, 'rb'))
    with open(path, "r") as f:
        data = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    dataset = defaultdict(list)
    for i, each in enumerate(data):
        dataset[each['arg_path']].append((i, each))

    while True:
        ds_name = input("dataset name: ")
        for name, v in dataset.items():
            if ds_name in name:
                # choose random example
                ex = random.choice(v)
                ex_tokens = tokens[ex[0]]['input_ids']
                # map tokens to their word pieces
                word_pieces = []
                for each in ex_tokens:
                    word_pieces.append((tokenizer.decode(each), each))
                
                print("example", i)
                print("============================================================================")
                print(" ".join([each[0] + f"({each[1]})"for each in word_pieces]))
                break

def inference_qa_routine(model_path, dataset_path):
    # run inference on a random example in each of the 26 datasets to identify problems
    import json
    from collections import defaultdict
    import random
    from tqdm import tqdm
    import vllm
    model = vllm.LLM(model=model_path)
    sampling_params = vllm.SamplingParams(temperature=0, max_tokens=256, skip_special_tokens=False)


    with open(dataset_path, "r") as f:
        data = json.load(f)
    datasets = defaultdict(list)
    for each in data:
        datasets[each['arg_path']].append(each)

    for k, v in tqdm(datasets.items()):
        ex = random.choice(v)
        print("Dataset:", k)
        print(ex['formatted_input'])
        res = model.generate(prompts=[ex['formatted_input']], sampling_params=sampling_params, use_tqdm=False)
        print("predicted output:", res[0].outputs[0].text)
        print("Intended output:", ex['seq_out'])
        print("============================================================================")
    

def reformat_for_yi(dataset_path, output_path):
    import json
    with open(dataset_path, "r") as f:
        data = json.load(f)

    # yi format is a jsonlines file, where each line is
    #{"id": "0", "messages": [{"role": "user", "content": "formatted input"}, {"role": "assistant", "content": "seq out"}]}
    # we need to convert the dataset to this format

    template = {
        "id": "",
        "messages": [
            {
                "role": "user",
                "content": ""
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
    }
    lines = []

    for i, each in enumerate(data):
        line = template.copy()
        line['id'] = str(i)
        line['messages'][0]['content'] = each['formatted_input']
        line['messages'][1]['content'] = each['seq_out']
        lines.append(line)
    
    with open(output_path, "w") as f:
        for each in lines:
            f.write(json.dumps(each) + "\n")

def format_alpaca_dataset(dataset_path, output_path):
    import json
    from tqdm import tqdm
    from datasets import load_dataset

    data = load_dataset(dataset_path)['train']
    
    format_str = ("Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n### Input:\n{input}\n### Response:\n")

    formatted_data = []
    for each in tqdm(data):
        formatted_data.append({
            "formatted_input": format_str.format(instruction=each['instruction'], input=each['input']),
            "seq_out": each['output'],
            "arg_path": "alpaca-cleaned",
            "truncated": False
        })

    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)

def format_orca_dataset(dataset_path, output_path):
    import json
    from tqdm import tqdm
    from datasets import load_dataset
    import random
    from functools import partial
    import multiprocessing

    data = load_dataset(dataset_path)["train"]
    format_str = (
        "[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{instruction} [/INST] "
    )
    formatted_data = []
    for each in tqdm(data):
        formatted_data.append({
            "formatted_input": format_str.format(sys_prompt=each['system_prompt'], instruction=each['question']),
            "seq_out": each['response'],
            "arg_path": "orca",
            "truncated": False
        })
    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)

    # filter orca to n= 743297 examples , where n is the number of examples not originally in orca
    # n = 676630
    # random.seed(42)
    # formatted_data = random.sample(formatted_data, n)
    # with open(output_path.replace(".json", f"_reduced.json"), "w") as f:
    #     json.dump(formatted_data, f, indent=4)
def process_batch(batch, tokenizer):
        res_batch = []
        for each in batch:
            if len(tokenizer(each['formatted_input'] + each['seq_out'])['input_ids']) > 2048:
                continue
            else:
                res_batch.append(each)
        return res_batch

def format_slim_orca_dataset(dataset_path, output_path):
    import json
    from tqdm import tqdm
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from functools import partial
    import multiprocessing
    data = load_dataset(dataset_path)["train"]
    tokenizer = AutoTokenizer.from_pretrained("../models/codellama-7b-instruct-hf")
    format_str = (
        "[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{instruction} [/INST] "
    )
    formatted_data = []
    for each in tqdm(data):
        convo = each['conversations']
        if len(convo) == 3:
            assert convo[0]['from'] == "system"
            assert convo[1]['from'] == "human"
            assert convo[2]['from'] == "gpt"
            sys = convo[0]['value']
            question = convo[1]['value']
            response = convo[2]['value']
        else:
            assert convo[0]['from'] == "human"
            assert convo[1]['from'] == "gpt"
            sys = ""
            question = convo[0]['value']
            response = convo[1]['value']
        formatted_input = format_str.format(sys_prompt=sys, instruction=question)
        formatted_data.append({
            "formatted_input": formatted_input,
            "seq_out": response,
            "arg_path": "slimorca",
            "truncated": False
        })

    batches = [formatted_data[i: i+1000] for i in range(0, len(formatted_data), 1000)]
    process_wrapper = partial(process_batch, tokenizer=tokenizer)
    with multiprocessing.Pool(processes=16) as pool:
        res = list(tqdm(pool.imap(process_wrapper, batches), total=len(batches)))
    
    formatted_data = [item for sublist in res for item in sublist]

    print(f"Percentage truncated: {1 - len(formatted_data) / len(data)}")
    with open(output_path, "w") as f:
        json.dump(formatted_data, f, indent=4)

def create_results_csv(exp_names):
    import json
    import pandas as pd
    import datetime

    ref = pd.read_csv("uskgandsota.csv")

    important_metrics = ['META_TUNING/tabmwp.cfg/acc', 'META_TUNING/totto.cfg/sacrebleu', 'META_TUNING/grailqa.cfg/exact_match', 'META_TUNING/sql2text.cfg/blec', 'META_TUNING/mmqa.cfg/f1', 'META_TUNING/spider_with_cell.cfg/exact_match', 'META_TUNING/kvret.cfg/all_micro', 'META_TUNING/hybridqa.cfg/acc', 'META_TUNING/sparc_with_cell.cfg/exact_match', 'META_TUNING/compwebq.cfg/acc', 'META_TUNING/tab_fact.cfg/all', 'META_TUNING/wikitq.cfg/all_ex', 'META_TUNING/wikisql.cfg/all_ex', 'META_TUNING/fetaqa.cfg/sacrebleu', 'META_TUNING/feverous.cfg/all', 'META_TUNING/multiwoz.cfg/Joint Acc', 'META_TUNING/dart.cfg/sacrebleu', 'META_TUNING/logic2text.cfg/blec', 'META_TUNING/mtop.cfg/exact_match', 'META_TUNING/bird.cfg/acc', 'META_TUNING/cosql_with_cell.cfg/exact_match', 'META_TUNING/sqa.cfg/all_acc', 'META_TUNING/webqsp.cfg/F1', 'META_TUNING/infotabs.cfg/acc', 'META_TUNING/wikitabletext.cfg/sacrebleu', 'META_TUNING/finqa.cfg/acc']
    names = ['tabmwp', 'totto', 'grailqa', 'sql2text', 'mmqa', 'spider', 'kvret', 'hybridqa', 'sparc', 'compwebq', 'tab_fact', 'wikitq', 'wikisql','fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop', 'bird', 'cosql', 'sqa', 'webqsp', 'infotabs', 'wikitabletext', 'finqa']
    # make a data frame with rownames important_metrics and column names exp_names
    colnames = [each for each in exp_names if "single_task" not in each] + ["single_task_ft", "uskg", "sota"]
    df = pd.DataFrame(index=important_metrics, columns=colnames)
    for i, r in ref.iterrows():
        metric = r['Property']
        df.loc[metric, "uskg"] = r['UnifiedSKG Best']
        df.loc[metric, "sota"] = r['SOTA']
    for each in exp_names:
        try:
            with open(f"./output/{each}/summary.json", "r") as f:
                metrics = json.load(f)
        except:
            print(f"{each} results do not exist, skipping")
            continue
        for k, v in metrics.items():
            if k in important_metrics:
                # multiple by 100 and round to 1 decimal place if it is below 1
                if "single_task" in each:
                    each = "single_task_ft"
                if v < 1:
                    df.loc[k, each] = round(v * 100, 2)
                else:
                    df.loc[k, each] = round(v, 1)
    date = datetime.datetime.now().strftime("%m-%d")
    df.to_csv(f"results_{date}.csv")

def collect_dataset_statistics(dataset_path):
    # dataset Name, number of examples, training max length, % truncated, output max length, test max length, test output max length
    # output to pandas
    import json
    from collections import defaultdict
    import pandas as pd
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("codellama-7b-instruct-hf")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    datasets = defaultdict(list)
    for each in data:
        datasets[each['arg_path']].append(each)
    stats = []
    for k, v in datasets.items():
        max_input_len = max([len(each['formatted_input']) for each in v])
        max_output_len = max([len(each['seq_out']) for each in v])
        truncated = sum([each['truncated'] for each in v])
        stats.append({
            "dataset_name": k,
            "num_examples": len(v),
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "truncated": truncated,
            "truncated_percent": truncated / len(v)
        })


if __name__ == "__main__":
    # import os
    # from tqdm import tqdm
    # lst = sorted([each for each in os.listdir('processed_data/') if each.startswith("v11")])
    # for each in tqdm(lst):
    #     llama_dataset_summary(os.path.join('processed_data/', each))
    #     print("\n\n")
    #view_gpt_responses("output/gpt4_tiny/responses.jsonl")
    # qa_dataset_examples("processed_data/v13_held_in_inst_orca_train.json")
    # start_qa_routine("processed_data/v11_all_inst_train.json")

    # tokenizer_path = "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/models/ckpts/v11_held_in_inst"
    # tokenpath = "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/tmp/cl_2048_non_upsampled.pkl"
    # path = "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/llama_data_v7_non_upsampled.json"
    # token_qa_routine(path, tokenpath, tokenizer_path)

    # inference_qa_routine("/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/models/ckpts/v11_held_in_inst/checkpoint-3966", "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/v11_all_inst_test.json")

    # reformat_for_yi("/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/v11_held_in_inst_train.json", "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/yi_v11_held_in_inst_train.jsonl")

    # format_alpaca_dataset("../datasets/alpaca-cleaned", "processed_data/formatted_alpaca_cleaned.json")
    # format_orca_dataset("../datasets/orca", "processed_data/formatted_orca.json")
    # format_slim_orca_dataset("../datasets/slimorca", "processed_data/formatted_slim_orca.json")

    # collect_dataset_statistics("processed_data/v13_held_in_inst_orca_train.json")

    create_results_csv([
        # # "v12_all_inst_mix_e2",
        # "v13_all_inst_orca_e2",
        # "v13_held_in_inst_orca_ratio_2_e2",
        # "v13_held_in_inst_orca_ratio_5_e2",
        # "v13_held_in_inst_orca_ratio_10_e2",
    "v13_all_inst_orca_e2",
        "v13_all_inst_orca_e3",
        "v13_all_inst_orca_13b",
        "v13_all_inst_orca_13b_e3",
        # "v13_tabmwp_single_task_ft_e3",
        # "v13_totto_single_task_ft_e3",
        # "v13_grailqa_single_task_ft_e3",
        # "v13_sql2text_single_task_ft_e3",
        # "v13_mmqa_single_task_ft_e3",
        # "v13_spider_single_task_ft_e3",
        # "v13_kvret_single_task_ft_e3",
        # "v13_hybridqa_single_task_ft_e3",
        # "v13_sparc_single_task_ft_e3",
        # "v13_compwebq_single_task_ft_e3",
        # "v13_tab_fact_single_task_ft_e3",
        # "v13_wikitq_single_task_ft_e3",
        # "v13_wikisql_single_task_ft_e3",
        # "v13_fetaqa_single_task_ft_e3",
        # "v13_feverous_single_task_ft_e3",
        # "v13_multiwoz_single_task_ft_e3",
        # "v13_dart_single_task_ft_e3",
        # "v13_logic2text_single_task_ft_e3",
        # "v13_mtop_single_task_ft_e3",
        # "v13_bird_single_task_ft_e3",
        # "v13_cosql_single_task_ft_e3",
        # "v13_sqa_single_task_ft_e3",
        # "v13_webqsp_single_task_ft_e3",
        # "v13_infotabs_single_task_ft_e3",
        # "v13_wikitabletext_single_task_ft_e3",
        # "v13_finqa_single_task_ft_e3",
        # "v12_all_inst_llama2-7b_e1",
        # "v12_all_inst_llemma-7b_e1",
        # "v12_all_inst_b256_e1",
        # "v13_db_diff_task_cotrain_e3",
        # "v13_qa_table_kt_cotrain_e3",
        # "v13_kt_diff_task_cotrain_e3",
        # "v13_summarization_table_kt_cotrain_e3",
        # "v13_table_diff_task_cotrain_e3",
        # "v12_codellama_1shot_baseline",
        # "v12_all_inst_b256_e2",
        "gpt35_test",
        "v13_all_inst_orca_13b",
        "v13_all_inst_orca_33b",
        "v13_all_inst_orca_33b_e3",

    ])
