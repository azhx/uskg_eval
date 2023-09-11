#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from transformers import LlamaTokenizerFast
import requests
import torch


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(self.args.llama.model_path)
        self.tokenizer.paddding_side = "left"

        # post a request to the port to verify its the model that we want
        res = requests.post(
            f"{self.args.llama.url}/verify_model",
            json={
                "model_path": self.args.llama.model_path
            }
        ).json()
    
        if not res['verified']:
            raise ValueError("Model path does not match the model path on the llama server")
        else:
            print(f"Model path verified to be {self.args.llama.model_path}")

    def forward(self, input_ids, attention_mask, labels):
        return {'loss': 0} # we don't need to compute loss for this model

    def generate(self, input_ids, attention_mask, **kwargs):

        generation_config = {
            "num_beams": 1,
            "num_return_sequences": 1,
        }

        # add kwargs to the generation config
        generation_config.update(kwargs)
        res = requests.post(
            f"{self.args.llama.url}/generate_from_tokens", #TODO remember to add an args for thiss
            json={
                # the batch of prompts to generate from
                "input_ids": input_ids.tolist(),
                # the generation config
                "generation_config": generation_config,
                "stop_ids": []
            }   
        ).json()

        # remake the tensor
        res['tokens'] = torch.tensor(res['tokens']).cuda()

        return res['tokens']
