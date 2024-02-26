#!/bin/bash

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

MODEL_DIR=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/models/codellama-7b-instruct-hf # /ML-A100/team/mm/zhangge/gezhangmv/SKGLM/models/llama2-7b-hf 
DATA_PATH=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/v12_held_in_tlabels_train.json #/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/processed_data/llama_data_v11_kg.json
OUTPUT_DIR=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/models/ckpts/v12_held_in_tlabels #/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/models/ckpts/v11_llama2_base

# NOTE : CONFIGURED TO RUN ON 16 GPUS

#LAUNCHER="python"
# LAUNCHER="deepspeed"
LAUNCHER="torchrun"
LAUNCHER_ARGS=(--master_addr=${MLP_WORKER_0_HOST} \
    --master_port=${MLP_WORKER_0_PORT} \
    --nnodes=${MLP_WORKER_NUM} \
    --node_rank=${MLP_ROLE_INDEX} \
    --nproc_per_node=${MLP_WORKER_GPU} \
)
SCRIPT="train_on_formatted.py"
SCRIPT_ARGS=(--model_name_or_path ${MODEL_DIR} \
    --data_path "${DATA_PATH}" \
    --output_dir ${OUTPUT_DIR} \
    --pkl_path "tmp/v12_held_in_tlabels.pkl" \
    --use_flash_attn_2 True \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_total_limit 3 \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.01 \
    --logging_steps 2 \
    --lr_scheduler_type cosine \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config_transformers4.31.json \
    --bf16 True
)

echo 'address:'${MASTER_ADDR},'SURM_JOBID:'${SLURM_PROCID}
echo $LAUNCHER "${LAUNCHER_ARGS[@]}" $SCRIPT "${SCRIPT_ARGS[@]}"

$LAUNCHER "${LAUNCHER_ARGS[@]}" $SCRIPT "${SCRIPT_ARGS[@]}"
