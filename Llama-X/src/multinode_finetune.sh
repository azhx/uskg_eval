#!/bin/bash

#SBATCH --job-name=dist_codellamax
#SBATCH --mem=200G
#SBATCH -c 10
#SBATCH --partition=a100
#SBATCH --qos=a100_wenhuchen
#SBATCH --nodelist=gpu182,gpu185
#SBATCH --output=%x.%j.log
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1

export WANDB_DISABLED=True

#MODEL_DIR=/ssd005/projects/waterloo_nlp/alex/Llama-2-7b-hf/
#MODEL_DIR=/ssd005/projects/waterloo_nlp/alex/llama2_8_18/before_restart/checkpoint-15380
MODEL_DIR=/h/wenhuchen/STORAGE/alex/codellama/CodeLlama-7b-Instruct-hf
#MODEL_DIR=/h/wenhuchen/STORAGE/alex/codellama_finetune/checkpoint-25630
DATA_PATH=/ssd005/projects/waterloo_nlp/alex/llama_data_v5_upsampled.json
OUTPUT_DIR=/ssd005/projects/waterloo_nlp/alex/codellama_llamax_finetune


#LAUNCHER="python"
LAUNCHER="deepspeed"
LAUNCHER_ARGS=(--num_gpus 4 \
    --num_nodes 2 \
    --master_addr "172.17.8.185" \
    --master_port 34545 \
    --hostfile configs/hostfile
)
SCRIPT="train.py"
SCRIPT_ARGS=(--model_name_or_path ${MODEL_DIR} \
    --data_path "${DATA_PATH}" \
    --output_dir ${OUTPUT_DIR} \
    --pkl_path "cl_2048_upsampled.pkl" \
    --has_instruction True \
    --dataset_type="skg" \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy no \
    --save_total_limit 3 \
    --save_strategy "epoch" \
    --learning_rate 2e-5 \
    --warmup_ratio 0.01 \
    --logging_steps 2 \
    --lr_scheduler_type cosine \
    --report_to "tensorboard" \
    --gradient_checkpointing True \
    --deepspeed configs/deepspeed_config.json \
    --fp16 True
)

echo 'address:'${MASTER_ADDR},'SURM_JOBID:'${SLURM_PROCID}
echo $LAUNCHER ${LAUNCHER_ARGS[@]} $SCRIPT "${SCRIPT_ARGS[@]}"

$LAUNCHER ${LAUNCHER_ARGS[@]} $SCRIPT "${SCRIPT_ARGS[@]}"
