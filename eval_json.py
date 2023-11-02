import argparse
import importlib
import os
import json
from utils.configue import Configure

def main(args):
    # use import lib to import EvaluateTool from metrics.{args.dataset_name}.evaluator
    output_path= f"./output/{args.run_name}_{args.dataset_name}"
    predictions_path = os.path.join(output_path,"predictions_predict.json")
    config_path = f"Salesforce/non_upsampled_{args.dataset_name}.cfg"
    run_args = Configure.Get(config_path)
    meta_tuning_path = getattr(run_args.arg_paths, args.dataset_name)
    meta_args = Configure.Get(meta_tuning_path)
    evaluator = importlib.import_module(meta_args.evaluate.tool).EvaluateTool(run_args)


    with open(predictions_path, "rb") as f:
        data = json.load(f)
    preds = [item['prediction'] for item in data]
    labs = data
    summary = evaluator.evaluate(preds, labs, "test")
    print(summary)
    with open(os.path.join(output_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

if __name__=="__main__":
    # args: name of the data, name of the run, 
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="wikisql")
    parser.add_argument("--run_name", type=str, default="run1")
    args = parser.parse_args()
    main(args)