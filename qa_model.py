'''
Author: ygjin11 1633504509@qq.com
Date: 2024-02-13 08:05:21
LastEditors: ygjin11 1633504509@qq.com
LastEditTime: 2024-02-14 10:37:00
FilePath: /uskg_eval/qa_model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import argparse
import os
from utils.configue import Configure
from transformers import AutoTokenizer

# argparse the model path
parser = argparse.ArgumentParser()
# read the first argument into the model path
parser.add_argument("model_ckpt", type=str)
args = parser.parse_args()

# check if the ckpt path is a real path
if not os.path.exists(args.model_ckpt):
    assert os.path.exists(f"configure/eval/{args.model_ckpt}/{args.model_ckpt}.cfg"), f"model checkpoint {args.model_ckpt} not found as an eval cfg or a real path"
    # load the config file at /configure/eval/{run_name}/{run_name}.cfg
    cfgargs = Configure.Get(f"eval/{args.model_ckpt}/{args.model_ckpt}.cfg")
    # get the model checkpoint path
    args.model_ckpt = cfgargs.model.path

# load the model
import vllm
model = vllm.LLM(model=args.model_ckpt, tensor_parallel_size=1)
# import pdb; pdb.set_trace()
sampling_params = vllm.SamplingParams(temperature=0, max_tokens=256, skip_special_tokens=True)
# get the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_ckpt, use_fast=True, add_eos_token=False, padding_side="left")

try:
    while True:
        prompt = input("Enter a prompt:\n")
        prompt = prompt.replace("\\n", "\n")

        # take the prompt literally by converting 
        
        tokens = tokenizer(prompt)
        # generate
        generations = model.generate(prompt_token_ids=[tokens.input_ids], sampling_params=sampling_params, use_tqdm=False)

        # print the generations
        print("======\nOutput:\n")
        for each in generations:
            print(each.outputs[0].text)
except:
    import pdb; pdb.post_mortem()