# the goal of this file is to load a flask server that loads llama independently, 
# so that we can save time doing prompt editing 

from flask import Flask, jsonify, request
from tqdm import tqdm
import json
import os
import argparse
import transformers
import torch
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
app = Flask(__name__)


class TokenStop(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids, scores):
        # if the last n tokens are stop tokens, stop
        #import pdb; pdb.set_trace()
        if len(self.stop_token_ids) == 0:
            return False
        for i, tok_id in enumerate(input_ids[0][-len(self.stop_token_ids):]):
            if tok_id != self.stop_token_ids[i]:
                return False
        return True
        
global tokenizer
global model
global model_path

@app.route('/generate', methods=['POST'])
def generate():
    req = request.get_json()
    batch = req.get("batch")
    # get stop tokens
    stop_ids = req.get("stop_ids")
    # get the ids using the tokenizer
    criteria = StoppingCriteriaList([TokenStop(stop_ids)])
    batch = tokenizer(
        batch,
        return_tensors="pt",
        add_special_tokens=False,
    )
    batch = {k: v.cuda() for k, v in batch.items()}

    # supply the object as kwargs to the generation config
    generation_config = GenerationConfig(**req.get("generation_config"))
    generation_output = model.generate(
        input_ids=batch['input_ids'],
        generation_config=generation_config,
        use_cache = True
        #stopping_criteria=criteria,
    )
    outputs = tokenizer.batch_decode(generation_output)
    return jsonify({"generations": outputs, "tokens": generation_output.tolist()})


@app.route('/generate_from_tokens', methods=['POST'])
def generate_from_tokens():
    try:
        res = request.get_json()
        input_ids = torch.tensor(res.get("input_ids")).cuda()
        generation_config = GenerationConfig(**res.get("generation_config"))
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            use_cache = True
        )
        # remove the input tokens from the output
        generation_output = generation_output[:, input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(generation_output)
        # print the name of the model that we are using
        print(f"model path: {model_path}")
        return jsonify({"generations": outputs, "tokens": generation_output.tolist()})
    except:
        import pdb; pdb.post_mortem()

@app.route('/verify_model', methods=['POST'])
def verify_model():
    # request will have a model path, check it against the
    # model path that we have
    req = request.get_json()
    req_model_path = req.get("model_path") # must be a full absolute path
    if model_path == req_model_path:
        return jsonify({"verified": True})
    else:
        return jsonify({"verified": False})

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    model_path = args.model_path #TODO not necessarily llama model in the future
    print(f"llama path: {model_path}")
    print("loading llama")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    #import pdb; pdb.set_trace()

    if "xgen" in model_path:
        model.config.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    model = model.eval()
    model = torch.compile(model)

    print("llama loading complete")
    app.run(host='0.0.0.0', port=args.port)

