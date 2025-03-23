import glob, os
from tqdm import tqdm
current_path = os.getcwd()

import re
import pandas as pd

from freeEvalLM._lib._df import read_file, save_file
from freeEvalLM.src.Evaluator import Evaluator
from freeEvalLM.tasks.ifeval.utils import process_results_new, agg_inst_level_acc

def convert_booleans(lst):

    return [ int(x) for x in lst]


def convert_booleans_list(lst):
    all = []
    for x in lst:
        for _x in x:
            all.append(int(_x))
    return all


class ifeval(Evaluator):
    def __init__(self, 
        task, 
        result_dir
        ):
        self.task = task
        self.result_dir = result_dir


    def evaluate(self):
        all_names = []
        all_finals = []
        for subtask, data_path in zip(self.all_dfs,  self.subtasks_data_path):
            name = subtask["subtask_name"]
            all_names.append(name)
            data = subtask["subtask_data"]


            scores = []
            answers = []
            final = []

            print(data)
            docs = data.to_dict(orient="records")

            answers = data["response"]

            for doc, result in zip(docs, answers):
                res = process_results_new(doc, result)         
                scores.append(res)


            df = pd.DataFrame({
                'filtered_answer': answers,
                'score': scores
            })
            
            data = data.join(df)
            # print(os.path.dirname(data_path))
            save_file(data, os.path.join(os.path.dirname(data_path), name+".json"))
            
        prompt_level_strict_acc = []
        inst_level_strict_acc = []
        prompt_level_loose_acc = []
        inst_level_loose_acc = []


        for score in scores:
            prompt_level_strict_acc.append(score["prompt_level_strict_acc"])
            inst_level_strict_acc.append(score["inst_level_strict_acc"])
            prompt_level_loose_acc.append(score["prompt_level_loose_acc"])
            inst_level_loose_acc.append(score["inst_level_loose_acc"])

        prompt_level_strict_acc = convert_booleans(prompt_level_strict_acc)
        prompt_level_loose_acc = convert_booleans(prompt_level_loose_acc)
        inst_level_strict_acc = convert_booleans_list(inst_level_strict_acc)
        inst_level_loose_acc = convert_booleans_list(inst_level_loose_acc)
        
        prompt_level_strict_acc = sum(prompt_level_strict_acc)/len(prompt_level_strict_acc)
        prompt_level_loose_acc = sum(prompt_level_loose_acc)/len(prompt_level_loose_acc)
        inst_level_strict_acc = sum(inst_level_strict_acc)/len(inst_level_strict_acc)
        inst_level_loose_acc = sum(inst_level_loose_acc)/len(inst_level_loose_acc)

        print(prompt_level_strict_acc)
        print(prompt_level_loose_acc)
        print(inst_level_strict_acc)
        print(inst_level_loose_acc)

        df = pd.DataFrame({
            'prompt_level_strict_acc': [prompt_level_strict_acc],
            'prompt_level_loose_acc': [prompt_level_loose_acc],
            'inst_level_strict_acc': [inst_level_strict_acc],
            'inst_level_loose_acc': [inst_level_loose_acc]
        })    
        save_file(df, os.path.join(os.path.dirname(data_path), "Results.csv"))

            

if __name__ == "__main__":

    model_list = [
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Llama-70B",
        "OpenThinker-7B",
        "OpenThinker-32B",
        "s1.1-32B",
        "QwQ"
    ]

    sum_list = [
        "summary",
        "summary_plus"
    ]

    for model in model_list:
        for sum1 in sum_list:
            path = f""

            a = ifeval("livebench", path)

            a.load_results()
            a.evaluate()