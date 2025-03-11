import glob, os
from tqdm import tqdm
current_path = os.getcwd()

import re
import pandas as pd

from _lib._df import read_file, save_file
from src.Evaluator import Evaluator

class mmlu_pro(Evaluator):
    def __init__(self, 
        task, 
        result_dir
        ):
        self.task = task
        self.result_dir = result_dir
        self.none = 0


    def count(self, scores):
        final = scores.count(1)/len(scores)
        return final
                    
    def filter_answer_subtask(self, data):
        answers = []
        for index, row in data.iterrows():
            response = row["response"]
            if response == None:
                self.none += 1
                answer = "None"
            else:
                answer = self.Choice_Matching(response)
            answers.append(answer)
        return answers

    def Choice_Matching(self, input):
        pattern = r'answer is \(?([ABCDEFGHIJ])\)?'
        matches = re.findall(pattern, input,  re.IGNORECASE)
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


if __name__ == "__main__":
    a = mmlu_pro("mmlu_pro", "/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/results/250303_mmlu_pro/share/home/wxzhao/gjh_ws/Downloads/LLMs/OpenThinker-32B")
    a.load_results()
    a.evaluate()
    print(a.none)

    # def Choice_Matching(input):
    #     pattern = r'answer is \(?([ABCDEFGHIJ])\)?'
    #     matches = re.findall(pattern, input,  re.IGNORECASE)
    #     return matches[0] if matches else "None"

    # print(Choice_Matching("\n\nThus, the answer is B.\n\n"))