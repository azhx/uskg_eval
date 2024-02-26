# datasets = [
#     "v12_all_inst_test.json",
#     "v12_all_tlabels_test.json",
#     "v12_db_diff_task_inst_test.json",
#     "v12_db_diff_task_tlabels_test.json",
#     "v12_kt_diff_task_inst_test.json",
#     "v12_kt_diff_task_tlabels_test.json",
#     "v12_qa_table_kt_inst_test.json",
#     "v12_qa_table_kt_tlabels_test.json",
#     "v12_summarization_table_kt_inst_test.json",
#     "v12_summarization_table_kt_tlabels_test.json",
#     "v12_table_diff_task_inst_test.json",
#     "v12_table_diff_task_tlabels_test.json",
# ]

# llema, llama, and codellama for first, and then codellama only for the rest

runs = {
    # "v12_all_inst_llama2-7b": {
    #     "model": "v12_held_in_inst_llama2-7b", 
    #     "data": "v12_all_inst_test.json"
    # },
    # "v12_all_inst_llemma-7b": {
    #     "model": "v12_held_in_inst_llemma-7b", 
    #     "data": "v12_all_inst_test.json"
    # },
    # "v12_all_inst": {
    #     "model": "v12_held_in_inst", 
    #     "data": "v12_all_inst_test.json"
    # },
    # "v12_all_tlabels": {
    #     "model": "v12_held_in_tlabels",
    #     "data": "v12_all_tlabels_test.json",
    # },
    # "v12_held_out_tlabels": {
    #     "model": "v12_held_in_tlabels",
    #     "data": "v12_all_inst_test.json",
    # },
    # "v12_db_diff_task_inst": {
    #     "model": "v12_db_diff_task_inst",
    #     "data": "v12_db_diff_task_inst_test.json",
    # },
    # "v12_db_diff_task_tlabels": {
    #     "model": "v12_db_diff_task_tlabels",
    #     "data": "v12_db_diff_task_tlabels_test.json",
    # },
    # "v12_kt_diff_task_inst": {
    #     "model": "v12_kt_diff_task_inst",
    #     "data": "v12_kt_diff_task_inst_test.json",
    # },
    # "v12_kt_diff_task_tlabels": {
    #     "model": "v12_kt_diff_task_tlabels",
    #     "data": "v12_kt_diff_task_tlabels_test.json",
    # },
    # "v12_qa_table_kt_inst": {
    #     "model": "v12_qa_table_kt_inst",
    #     "data": "v12_qa_table_kt_inst_test.json",
    # },
    # "v12_qa_table_kt_tlabels": {
    #     "model": "v12_qa_table_kt_tlabels",
    #     "data": "v12_qa_table_kt_tlabels_test.json",
    # },
    # "v12_summarization_table_kt_inst": {
    #     "model": "v12_summarization_table_kt_inst",
    #     "data": "v12_summarization_table_kt_inst_test.json",
    # },
    # "v12_summarization_table_kt_tlabels": {
    #     "model": "v12_summarization_table_kt_tlabels",
    #     "data": "v12_summarization_table_kt_tlabels_test.json",
    # },
    # "v12_table_diff_task_inst": {
    #     "model": "v12_table_diff_task_inst",
    #     "data": "v12_table_diff_task_inst_test.json",
    # },
    # "v12_table_diff_task_tlabels": {
    #     "model": "v12_table_diff_task_tlabels",
    #     "data": "v12_table_diff_task_tlabels_test.json",
    # },
    # "v12_all_inst_b128":{
    #     "model": "v12_held_in_inst_b128",
    #     "data": "v12_all_inst_test.json",
    # },
    # "v12_all_inst_b256":{
    #     "model": "v12_held_in_inst_b256",
    #     "data": "v12_all_inst_test.json",
    # }
    # "v13_held_in_inst_orca_ratio_2": {
    #     "model": "v13_held_in_inst_orca_ratio_2",
    #     "data": "v13_all_inst_test.json",
    # },
    # "v13_held_in_inst_orca_ratio_5": {
    #     "model": "v13_held_in_inst_orca_ratio_5",
    #     "data": "v13_all_inst_test.json",
    # },
    # "v13_held_in_inst_orca_ratio_10": {
    #     "model": "v13_held_in_inst_orca_ratio_10",
    #     "data": "v13_all_inst_test.json",
    # },
    "v13_table_diff_task_cotrain": {
        "model": "v13_table_diff_task_cotrain",
        "data": "v13_table_diff_task_cotrain_test.json",
    },
    "v13_summarization_table_kt_cotrain": {
        "model": "v13_summarization_table_kt_cotrain",
        "data": "v13_summarization_table_kt_cotrain_test.json",
    },
    "v13_qa_table_kt_cotrain": {
        "model": "v13_qa_table_kt_cotrain",
        "data": "v13_qa_table_kt_cotrain_test.json",
    },
    "v13_db_diff_task_cotrain": {
        "model": "v13_db_diff_task_cotrain",
        "data": "v13_db_diff_task_cotrain_test.json",
    },
    "v13_kt_diff_task_cotrain": {
        "model": "v13_kt_diff_task_cotrain",
        "data": "v13_kt_diff_task_cotrain_test.json",
    },
    'v13_tabmwp_single_task_ft': {'model': 'v13_tabmwp_single_task_ft','data': 'v13_tabmwp_single_task_ft_test.json'},                                             
    'v13_totto_single_task_ft': {'model': 'v13_totto_single_task_ft','data': 'v13_totto_single_task_ft_test.json'},                                              
    'v13_grailqa_single_task_ft': {'model': 'v13_grailqa_single_task_ft',    'data': 'v13_grailqa_single_task_ft_test.json'},                                            
    'v13_sql2text_single_task_ft': {'model': 'v13_sql2text_single_task_ft',    'data': 'v13_sql2text_single_task_ft_test.json'},                                           
    'v13_mmqa_single_task_ft': {'model': 'v13_mmqa_single_task_ft','data': 'v13_mmqa_single_task_ft_test.json'},                                               
    'v13_spider_single_task_ft': {'model': 'v13_spider_single_task_ft','data': 'v13_spider_single_task_ft_test.json'},                                             
    'v13_kvret_single_task_ft': {'model': 'v13_kvret_single_task_ft', 'data': 'v13_kvret_single_task_ft_test.json'},
    'v13_hybridqa_single_task_ft': {'model': 'v13_hybridqa_single_task_ft','data': 'v13_hybridqa_single_task_ft_test.json'},                                           
    'v13_sparc_single_task_ft': {'model': 'v13_sparc_single_task_ft','data': 'v13_sparc_single_task_ft_test.json'},                                              
    'v13_compwebq_single_task_ft': {'model': 'v13_compwebq_single_task_ft','data': 'v13_compwebq_single_task_ft_test.json'},                                           
    'v13_tab_fact_single_task_ft': {'model': 'v13_tab_fact_single_task_ft',    'data': 'v13_tab_fact_single_task_ft_test.json'},                                           
    'v13_wikitq_single_task_ft': {'model': 'v13_wikitq_single_task_ft',    'data': 'v13_wikitq_single_task_ft_test.json'},                                             
    'v13_wikisql_single_task_ft': {'model': 'v13_wikisql_single_task_ft',    'data': 'v13_wikisql_single_task_ft_test.json'},                                            
    'v13_fetaqa_single_task_ft': {'model': 'v13_fetaqa_single_task_ft',    'data': 'v13_fetaqa_single_task_ft_test.json'},                                             
    'v13_feverous_single_task_ft': {'model': 'v13_feverous_single_task_ft',    'data': 'v13_feverous_single_task_ft_test.json'},                                           
    'v13_multiwoz_single_task_ft': {'model': 'v13_multiwoz_single_task_ft',    'data': 'v13_multiwoz_single_task_ft_test.json'},                                           
    'v13_dart_single_task_ft': {'model': 'v13_dart_single_task_ft',    'data': 'v13_dart_single_task_ft_test.json'},                                               
    'v13_logic2text_single_task_ft': {'model': 'v13_logic2text_single_task_ft',    'data': 'v13_logic2text_single_task_ft_test.json'},                                         
    'v13_mtop_single_task_ft': {'model': 'v13_mtop_single_task_ft',    'data': 'v13_mtop_single_task_ft_test.json'},                                               
    'v13_bird_single_task_ft': {'model': 'v13_bird_single_task_ft',    'data': 'v13_bird_single_task_ft_test.json'},                                               
    'v13_cosql_single_task_ft': {'model': 'v13_cosql_single_task_ft',    'data': 'v13_cosql_single_task_ft_test.json'},                                              
    'v13_sqa_single_task_ft': {'model': 'v13_sqa_single_task_ft','data': 'v13_sqa_single_task_ft_test.json'},                                                
    'v13_webqsp_single_task_ft': {'model': 'v13_webqsp_single_task_ft',    'data': 'v13_webqsp_single_task_ft_test.json'},                                             
    'v13_infotabs_single_task_ft': {'model': 'v13_infotabs_single_task_ft',    'data': 'v13_infotabs_single_task_ft_test.json'},                                           
    'v13_wikitabletext_single_task_ft': {'model': 'v13_wikitabletext_single_task_ft',    'data': 'v13_wikitabletext_single_task_ft_test.json'},                                      
    'v13_finqa_single_task_ft': {'model': 'v13_finqa_single_task_ft',    'data': 'v13_finqa_single_task_ft_test.json'},

    "v13_held_out_orca_few_shot": {
        "model": "v13_held_in_inst_orca",
        "data": "v13_wi_we_5shot_test.json",
    }
}
# for every run name, make configure/eval/{run_name}_{epoch_number}/{run_name}.cfg
cfg_format = """[model]
name = "unified.vllm"
path = "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/models/ckpts/{model_dir}/{ckpt}"

[dataset]
test_split_json = "./processed_data/{data_file}"

[seq2seq]
constructor = "seq2seq_construction.meta_tuning"

[evaluate]
tool = "metrics.meta_tuning.no_evaluation"
"""
import os
from tqdm import tqdm

commands = []
for run_name, v in tqdm(runs.items()):

    # look in the ckpt directory for the ckpt steps
    ckpts = sorted([name for name in os.listdir(f"models/ckpts/{v['model']}/") if "checkpoint" in name])

    for i in range(3):
        epoch = i+1
        # make the directory first, even if it exists
        os.makedirs(f"configure/eval/{run_name}_e{epoch}", exist_ok=True)
        commands.append(f"launchers/run_test_eval.sh {run_name}_e{epoch}")
        with open(
            f"configure/eval/{run_name}_e{epoch}/{run_name}_e{epoch}.cfg", "w"
        ) as f:
            f.write(
                cfg_format.format(
                    model_dir=v["model"],
                    ckpt=ckpts[i],
                    data_file=v["data"],
                )
            )

# os.makedirs(f"configure/eval/{codellama}_{baseline}", exist_ok=True)
print("\n".join(commands))
