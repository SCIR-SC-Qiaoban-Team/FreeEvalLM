import glob, os
from tqdm import tqdm
current_path = os.getcwd()

import re
import pandas as pd

from _lib._df import read_file, save_file


class Evaluator:
    def __init__(self, 
        task, 
        result_dir
        ):
        self.task = task
        self.result_dir = result_dir


    def load_results(self):
        print("loading results")
        print(self.result_dir)
        self.subtasks_data_path = glob.glob(os.path.join(self.result_dir, "*.json"))
        self.all_dfs = []
        for data_path in self.subtasks_data_path:
            print("loading subtask results: ", os.path.basename(data_path).split(".")[0])
            subtask_name = os.path.basename(data_path).split(".")[0]
            subtask_data = read_file(data_path)
            
            if "filtered_answer" in subtask_data.columns:
                subtask_data.drop(['filtered_answer', 'score'], axis=1, inplace=True)

            _dict = {
                "subtask_name": subtask_name,
                "subtask_data": subtask_data
            }

            self.all_dfs.append(_dict)


    def evaluate(self):
        all_names = []
        all_finals = []
        for subtask, data_path in zip(self.all_dfs,  self.subtasks_data_path):
            name = subtask["subtask_name"]
            all_names.append(name)
            data = subtask["subtask_data"]
            answers = self.filter_answer_subtask(data)
            scores = self.compare(answers, data["target"])
            final = self.count(scores)
            all_finals.append(final)
            df = pd.DataFrame({
                'filtered_answer': answers,
                'score': scores
            })
            
            data = data.join(df)
            print(os.path.dirname(data_path))
            save_file(data, os.path.join(os.path.dirname(data_path), name+".json"))
            
        all_names += ["FINAL"]
        all_finals += [sum(all_finals)/len(all_finals)]
        
        df = pd.DataFrame({
            'subtask': all_names,
            'score': all_finals
        })    
        save_file(df, os.path.join(os.path.dirname(data_path), "Results.csv"))
            


if __name__ == "__main__":
    a = Evaluator("mmlu_pro", "/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250303_mmlu_pro/share/home/wxzhao/gjh_ws/Downloads/LLMs/DeepSeek-R1-Distill-Llama-8B")
    a.load_results()
    a.evaluate()
