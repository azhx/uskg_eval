# CUDA_VISIBLE_DEVICES=6 python llama_service.py \
#     --llama_path '/mnt/tjena/alex/llama2_base_no_instr_7_26' \
#     --port 8090\

    #--llama_path '/mnt/tjena/alex/llama_6_29' \
# -m pdb -c "c"
    
#--model_path '/mnt/tjena/alex/vaughan/checkpoint-15380' \
#--model_path '/mnt/tjena/alex/llama2_7_26' \
#--model_path '/mnt/tjena/alex/vaughan/xgen/checkpoint-10253' \
#--model_path '/mnt/tjena/alex/vaughan/codellamae5' \
CUDA_VISIBLE_DEVICES=2 python llama_service.py \
    --model_path '/mnt/tjena/alex/vaughan/llamax_e2' \
    --port 8092 \
    #--model_path '/mnt/tjena/alex/llama2_chat_7_31' \

