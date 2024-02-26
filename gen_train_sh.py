

TEMPLATE = """#!/bin/bash

#SBATCH --job-name=codellama_llamax_finetune
#SBATCH --mem=200G
#SBATCH -c 10
#SBATCH --partition=a100
#SBATCH --qos=a100_wenhuchen
#SBATCH -w gpu185
#SBATCH --output=%x.%j.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

## must export blow enviroment variables
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_DEBUG=INFO

export NCCL_DEBUG=DEBUG
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_1:1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
 

__conda_setup="$('/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

export HF_HOME=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/.cache/huggingface
cd /ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/Llama-X/src
conda activate SKGLM

export WANDB_DISABLED=True

MODEL_DIR=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/models/{model_path} # /ML-A100/team/mm/zhangge/gezhangmv/SKGLM/models/llama2-7b-hf 
DATA_PATH=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/{data_path} #/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/llama_data_v11_kg.json
OUTPUT_DIR=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/models/ckpts/{ckpt_path} #/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/models/ckpts/v11_llama2_base

# NOTE : CONFIGURED TO RUN ON 16 GPUS

#LAUNCHER="python"
# LAUNCHER="deepspeed"
LAUNCHER="torchrun"
LAUNCHER_ARGS=(--master_addr=${MLP_WORKER_0_HOST} \\
    --master_port=${MLP_WORKER_0_PORT} \\
    --nnodes=${MLP_WORKER_NUM} \\
    --node_rank=${MLP_ROLE_INDEX} \\
    --nproc_per_node=${MLP_WORKER_GPU} \\
)
SCRIPT="train_on_formatted.py"
SCRIPT_ARGS=(--model_name_or_path ${MODEL_DIR} \\
    --data_path "${DATA_PATH}" \\
    --output_dir ${OUTPUT_DIR} \\
    --pkl_path "tmp/{ckpt_path}.pkl" \\
    --use_flash_attn_2 True \\
    --num_train_epochs 3 \\
    --model_max_length 2048 \\
    --per_device_train_batch_size 16 \\
    --per_device_eval_batch_size 1 \\
    --gradient_accumulation_steps 2 \\
    --evaluation_strategy no \\
    --save_total_limit 3 \\
    --save_strategy "epoch" \\
    --learning_rate 2e-5 \\
    --warmup_ratio 0.01 \\
    --logging_steps 2 \\
    --lr_scheduler_type cosine \\
    --report_to "tensorboard" \\
    --gradient_checkpointing True \\
    --deepspeed configs/deepspeed_config_transformers4.31.json \\
    --bf16 True
)

echo 'address:'${MASTER_ADDR},'SURM_JOBID:'${SLURM_PROCID}
echo $LAUNCHER "${LAUNCHER_ARGS[@]}" $SCRIPT "${SCRIPT_ARGS[@]}"

$LAUNCHER "${LAUNCHER_ARGS[@]}" $SCRIPT "${SCRIPT_ARGS[@]}"
"""

# files_to_make = [
#  #'v11_all_prompt_input_train.json',
#  #'v11_all_task_label_train.json',
#  'v11_db_diff_task_inst_prompt_input_train.json',
#  'v11_db_diff_task_tlabels_task_label_train.json',
#  'v11_held_in_prompt_input_train.json',
#  'v11_held_in_task_label_train.json',
#  #'v11_held_out_prompt_input_train.json',
#  #'v11_held_out_task_label_train.json',
#  'v11_kt_diff_task_inst_prompt_input_train.json',
#  'v11_kt_diff_task_tlabels_task_label_train.json',
#  'v11_qa_table_kt_inst_prompt_input_train.json',
#  'v11_qa_table_kt_tlabels_task_label_train.json',
#  'v11_summarization_table_kt_inst_prompt_input_train.json',
#  'v11_summarization_table_kt_tlabels_task_label_train.json',
#  'v11_table_diff_task_inst_prompt_input_train.json',
#  'v11_table_diff_task_tlabels_task_label_train.json']


files_to_make = [
#  'v11_all_inst_train.json',
#  'v11_all_tlabels_train.json',
#  'v12_db_diff_task_inst_train.json',
#  'v12_db_diff_task_tlabels_train.json',
#  'v12_held_in_inst_train.json',
#  'v12_held_in_tlabels_train.json',
# #  'v12_held_out_inst_train.json',
# #  'v12_held_out_tlabels_train.json',
#  'v12_kt_diff_task_inst_train.json',
#  'v12_kt_diff_task_tlabels_train.json',
#  'v12_qa_table_kt_inst_train.json',
#  'v12_qa_table_kt_tlabels_train.json',
#  'v12_summarization_table_kt_inst_train.json',
#  'v12_summarization_table_kt_tlabels_train.json',
#  'v12_table_diff_task_inst_train.json',
#  'v12_table_diff_task_tlabels_train.json']
    # "v13_held_in_inst_orca_ratio_2_train.json",
    # "v13_held_in_inst_orca_ratio_5_train.json",
    # "v13_held_in_inst_orca_ratio_10_train.json",
    # "v13_table_diff_task_inst_train.json",
    # "v13_db_diff_task_inst_train.json",
    # "v13_kt_diff_task_inst_train.json",
    # "v13_summarization_table_kt_inst_train.json",
    # "v13_qa_table_kt_inst_train.json",
    "v13_table_diff_task_cotrain_train.json",
    "v13_db_diff_task_cotrain_train.json",
    "v13_kt_diff_task_cotrain_train.json",
    "v13_summarization_table_kt_cotrain_train.json",
    "v13_qa_table_kt_cotrain_train.json",
    "v13_tabmwp_single_task_ft_train.json",
    "v13_totto_single_task_ft_train.json",
    "v13_grailqa_single_task_ft_train.json",
    "v13_sql2text_single_task_ft_train.json",
    "v13_mmqa_single_task_ft_train.json",
    "v13_spider_single_task_ft_train.json",
    "v13_kvret_single_task_ft_train.json",
    "v13_hybridqa_single_task_ft_train.json",
    "v13_sparc_single_task_ft_train.json",
    "v13_compwebq_single_task_ft_train.json",
    "v13_tab_fact_single_task_ft_train.json",
    "v13_wikitq_single_task_ft_train.json",
    "v13_wikisql_single_task_ft_train.json",
    "v13_fetaqa_single_task_ft_train.json",
    "v13_feverous_single_task_ft_train.json",
    "v13_multiwoz_single_task_ft_train.json",
    "v13_dart_single_task_ft_train.json",
    "v13_logic2text_single_task_ft_train.json",
    "v13_mtop_single_task_ft_train.json",
    "v13_bird_single_task_ft_train.json",
    "v13_cosql_single_task_ft_train.json",
    "v13_sqa_single_task_ft_train.json",
    "v13_webqsp_single_task_ft_train.json",
    "v13_infotabs_single_task_ft_train.json",
    "v13_wikitabletext_single_task_ft_train.json",
    "v13_finqa_single_task_ft_train.json"
    ]



# make a codellama 7b run for each of the files_to_make


sh_lst = []
from functools import partial
for f in files_to_make:
    model_path = "codellama-7b-instruct-hf"
    data_path = f
    # remove "train" from the file path
    ckpt_path = f.split('.')[0].replace("_train", "")
    sh_lst.append(f"finetune_{ckpt_path}.sh")
    with open(f"Llama-X/src/finetune_{ckpt_path}.sh", "w") as f:
        f.write(TEMPLATE.replace("{model_path}", model_path).replace("{data_path}", data_path).replace("{ckpt_path}",ckpt_path))

# make a llema run for just the held in instruction dataset

# model_path = "llemma-7b"
# data_path = "v12_held_in_inst_train.json"
# ckpt_path = "v12_held_in_inst_llemma-7b"
# sh_lst.append(f"finetune_{ckpt_path}.sh")
# with open(f"Llama-X/src/finetune_{ckpt_path}.sh", "w") as f:
#     f.write(TEMPLATE.replace("{model_path}", model_path).replace("{data_path}", data_path).replace("{ckpt_path}",ckpt_path))

# # make a llama2 base run for just the held in instruction dataset
# model_path = "llama2-7b-hf"
# data_path = "v12_held_in_inst_train.json"
# ckpt_path = "v12_held_in_inst_llama2-7b"
# sh_lst.append(f"finetune_{ckpt_path}.sh")
# with open(f"Llama-X/src/finetune_{ckpt_path}.sh", "w") as f:
#     f.write(TEMPLATE.replace("{model_path}", model_path).replace("{data_path}", data_path).replace("{ckpt_path}",ckpt_path))

for each in sh_lst:
    print(each.replace(".sh", ""))
for each in sh_lst:
    print(f"/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/Llama-X/src/{each}")
