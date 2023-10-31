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

export WANDB_DISABLED=True

#MODEL_DIR=/ssd005/projects/waterloo_nlp/alex/Llama-2-7b-hf/
#MODEL_DIR=/ssd005/projects/waterloo_nlp/alex/llama2_8_18/before_restart/checkpoint-15380
# MODEL_DIR=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/codellama/CodeLlama-7b-Instruct-hf
# #MODEL_DIR=/h/wenhuchen/STORAGE/alex/codellama_finetune/checkpoint-25630
# DATA_PATH=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/llama_data_v6_upsampled_rcs.json
# OUTPUT_DIR=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v6_rcs

#MODEL_DIR=/cpfs/29cd2992fe666f2a/shared/public/hub/models--codellama--CodeLlama-13b-Instruct-hf/snapshots/ff0983bc4267bb98ead4fb5168fe2f049b442787
MODEL_DIR=/cpfs/29cd2992fe666f2a/shared/public/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/6114dd1e16f69e0765ccbd7a64d33d04b265fbd2/
#MODEL_DIR=/h/wenhuchen/STORAGE/alex/codellama_finetune/checkpoint-25630
DATA_PATH=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/llama_data_v10_nommqa.json
OUTPUT_DIR=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_nu_prompt_fixed



#LAUNCHER="python"
LAUNCHER="deepspeed"
SCRIPT="train.py"
SCRIPT_ARGS=(--model_name_or_path ${MODEL_DIR} \
    --data_path "${DATA_PATH}" \
    --output_dir ${OUTPUT_DIR} \
    --pkl_path "cl_2048_nommqa_noinst.pkl" \
    --has_instruction False \
    --dataset_type="skg" \
    --num_train_epochs 3 \
    --model_max_length 2048 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
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
    --fp16 True
)

echo 'address:'${MASTER_ADDR},'SURM_JOBID:'${SLURM_PROCID}
echo $LAUNCHER $SCRIPT "${SCRIPT_ARGS[@]}"

$LAUNCHER $SCRIPT "${SCRIPT_ARGS[@]}"
