import glob, os
from tqdm import tqdm
current_path = os.getcwd()

import re
import pandas as pd

from freeEvalLM._lib._df import read_file, save_file
from freeEvalLM.src.Evaluator import Evaluator

class mmlu_pro(Evaluator):
    def __init__(self, 
        task, 
        result_dir
        ):
        self.task = task
        self.result_dir = result_dir
        self.none = 0


    def count(self, scores):
        final = round(scores.count(1)/len(scores), 4)
        return final
                    
    def filter_answer_subtask(self, data):
        answers = []
        for index, row in data.iterrows():
            response = row["response"]
            try:
                reasoning = row["reasoning"]
            except:
                reasoning = row["response"]
            print(response)
            if response == None:
                self.none += 1
                answer = "None"
            else:
                answer = self.Choice_Matching(response, reasoning)
            answers.append(answer)
        return answers



    def Choice_Matching(self, response, reasoning):
        if isinstance(response, str):
            pass
        else:
            response = "None"
        if isinstance(reasoning, str):
            pass
        else:
            response = "None"
        pattern = r'answer is \(?([ABCDEFGHIJ])\)?'
        matches = re.findall(pattern, response,  re.IGNORECASE)
        # return matches[0] if matches else "None"
        if matches:
            return matches[0]
        else:
            matches = re.findall(pattern, reasoning,  re.IGNORECASE)
            return matches[0] if matches else "None"
    
    def compare(self, answers, targets):
        scores = []
        for answer, target in zip(answers, targets):
            if answer == target:
                score = 1
            else:
                score = 0
            scores.append(score)
        return scores


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
        all_finals += [round(sum(all_finals)/len(all_finals), 4)]
        
        df = pd.DataFrame({
            'subtask': all_names,
            'score': all_finals
        })    
        save_file(df, os.path.join(os.path.dirname(data_path), "Results.csv"))
            


if __name__ == "__main__":


   pass
