#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
#from .base import PushToHubFriendlyModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model.path, use_fast=False)
        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model.path,
            device_map="auto",
        )
        state_dict = torch.load(args.model.path + "/pytorch_model.bin") # really bad hack
        state_dict = {k[len("pretrain_model."):] : v for k, v in state_dict.items()}
        print('all states recovered')
        self.pretrain_model.load_state_dict(state_dict)
        self.pretrain_model.to("cuda")
        print("model device", self.pretrain_model.device)
        self.config = self.pretrain_model.config
        self.save_pretrained = self.pretrain_model.save_pretrained
        

        if args.special_tokens:
            self.tokenizer.add_tokens([v for k, v in args.special_tokens])
            self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

    def forward(self, input_ids, attention_mask, labels):
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
        ).loss
        return {'loss': loss}

    def generate(self, inputs, generation_max_length, **kwargs):
        generated_ids = self.pretrain_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=generation_max_length,
            temperature=0,
            use_cache=True,
            **kwargs,
        )
        # print the input that is not masked
        # find the first zero in the attention mask
        # print(self.tokenizer.batch_decode(input_ids[:,:attention_mask.sum()]))
        # print(self.tokenizer.batch_decode(generated_ids))

        return generated_ids