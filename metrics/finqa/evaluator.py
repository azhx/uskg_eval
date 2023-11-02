# encoding=utf8
import numpy as np

class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):

        total = 0
        correct = 0
        for pred, gold_item in zip(preds, golds):
            if pred.lower().endswith(gold_item['final_res'].lower()):
                # for non numeric answers, just check if the answer is in the prediction
                correct += 1
            else:
                # first remove all percent signs and money signs from the answer
                pred = pred.replace('%', '').replace("$", '')
                # if it contains an equal sign, take the part before the equal sign
                if '=' in pred:
                    pred = pred.split('=')[0]

                # if gold is a percentage, remove the percent sign and express as a decimal
                if gold_item['final_res'].endswith('%'):
                    gold = float(gold_item['answer'].replace('%', '')) / 100
                # try to evaluate the expression
                else:
                    gold = float(eval(gold_item['final_res']))

                try:
                    pred = float(eval(pred))
                    # round to the same number of decimal places as the gold answer
                    pred = round(pred, len(str(gold).split('.')[1]))
                    # if the prediction is close enough to the gold answer, count as correct
                    if np.isclose(pred, gold, atol=0.001):
                        correct += 1
                except:
                    # count as incorrect
                    pass
            total += 1
        print("Acc: ", correct/total)
        return {"acc": correct/total}
