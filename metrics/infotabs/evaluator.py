# encoding=utf8
import numpy as np

class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):

        total = 0
        correct = 0

        entail_synonyms = ['entail', 'support', 'E', 'reasonable']
        refute_synonyms = ['refute', 'contradict', 'C', 'wrong']
        neutral_synonyms = ['neutral', 'irrelevant', 'N', 'neither', "not necessarily either"]

        for pred, gold_item in zip(preds, golds): 
            # correct or not
            if any(synonym in pred.lower() for synonym in entail_synonyms):
                pred = "e"
            elif any(synonym in pred.lower() for synonym in refute_synonyms):
                pred = "c"
            elif any(synonym in pred.lower() for synonym in neutral_synonyms):
                pred = "n"
            if pred.lower() == gold_item['label'].lower():
                correct += 1
            total += 1
        print("Acc: ", correct/total)
        return {"acc": correct/total}
