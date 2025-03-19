import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import time
import os
import re
import glob

import nltk
import numpy as np
from tqdm import tqdm
import math

# from freeEvalLM.tasks.livebench.model.api_models import get_model
from freeEvalLM.tasks.livebench.process_results.data_analysis.tablereformat.utils import table_process_results
from freeEvalLM.tasks.livebench.process_results.data_analysis.cta.utils import cta_process_results
from freeEvalLM.tasks.livebench.process_results.data_analysis.tablejoin.utils import joinmap_process_results
# from freeEvalLM.tasks.livebench.process_results.reasoning.web_of_lies_v2.utils import web_of_lies_process_results
# from freeEvalLM.tasks.livebench.process_results.reasoning.house_traversal.utils import house_traversal_process_results
# from freeEvalLM.tasks.livebench.process_results.reasoning.zebra_puzzle.utils import get_zebra_puzzle_evaluator
# from freeEvalLM.tasks.livebench.process_results.reasoning.spatial.utils import spatial_process_results
# from freeEvalLM.tasks.livebench.process_results.math.math_competitions.utils import mathcontest_process_results,aime_process_results 
# from freeEvalLM.tasks.livebench.process_results.math.olympiad.utils import proof_rearrangement_process_results
# from freeEvalLM.tasks.livebench.process_results.math.AMPS_Hard.utils import amps_hard_process_results 
from freeEvalLM.tasks.livebench.process_results.writing.plot_unscrambling.utils import plot_unscrambling_process_results
from freeEvalLM.tasks.livebench.process_results.writing.typos.utils import typos_process_results
from freeEvalLM.tasks.livebench.process_results.writing.connections.utils import get_connections_puzzle_evaluator
# from freeEvalLM.tasks.livebench.process_results.coding.utils import LCB_generation_process_results
from freeEvalLM.tasks.livebench.process_results.instruction_following.utils import instruction_following_process_results


from freeEvalLM.tasks.livebench.common import (
    LIVE_BENCH_RELEASES,
    load_questions,
    load_questions_jsonl,
    load_model_answers,
    check_data,
    get_model_list,
    make_match_single,
    MatchSingle,
    get_categories_tasks,
    LIVE_BENCH_DATA_SUPER_PATH
)

def play_a_match_gt(match: MatchSingle, output_file: str, debug=False):
    """
    Evaluate a model's answer to a question.

    Args:
        match: An object containing the question, model name, and model answer.
        output_file: The path to which the judgment should be outputted.
    
    Returns:
        result: The judgment, containing the question id, task name, model name, score, turn, timestamp, and category name
    """
    question, model, answer = (
        match.question,
        match.model,
        match.answer,
    )
    coding_test_case_tasks = ["coding_completion", "LCB_generation"]
    if "ground_truth" not in question and "reference" not in question and question["task"] not in coding_test_case_tasks and question["category"] != "instruction_following":
        # aside from coding and instruction following tasks, all questions should contain the ground truth answer
        raise ValueError("Questions must have ground_truth to run gen_ground_truth_judgment.")

    task = question["task"]
    print("task is")
    print(task)
    task_or_subtask = question["subtask"] if "subtask" in question.keys() else question["task"]
    question_text = question["turns"][0]
    ground_truth = question.get("ground_truth", None)
    # llm_answer = answer['choices'][0]['turns'][-1]
    # llm_answer = re.sub(f"<think>.*?<\/think>", "", llm_answer, flags=re.DOTALL)


    llm_answer = answer
    if llm_answer:
        if isinstance(llm_answer, str):
            pass
        else:
            llm_answer = "None"
    else:
        llm_answer = "None"

    print("llm_answer ###############", llm_answer)
    score = 0
    category = None

    # todo: find a better solution than a long if statement.
    splits = task_or_subtask.split('_')
    try:
        if splits[0] in ["amc", "smc"] or (len(splits) > 1 and splits[1] == "amc"):
            score = mathcontest_process_results(ground_truth, llm_answer, question_text, debug)
            category = "math"
        elif splits[0] == "aime":
            score = aime_process_results(ground_truth, llm_answer, debug)
            category = "math"
        elif splits[0] in ["imo", "usamo"]:
            score = proof_rearrangement_process_results(ground_truth, llm_answer, edit_distance=True, debug=debug)
            category = "math"
        elif task_or_subtask == "cta":
            score = cta_process_results(ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif task_or_subtask == "tablereformat":
            score = table_process_results(question_text, ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif task_or_subtask == "tablejoin":
            score = joinmap_process_results(question_text, ground_truth, llm_answer, debug)
            category = "data_analysis"
        elif "amps_hard" in task_or_subtask:
            score = amps_hard_process_results(ground_truth, llm_answer, debug)
            category = "math"
        elif task_or_subtask == "web_of_lies_v2":
            score = web_of_lies_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "house_traversal":
            score = house_traversal_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "zebra_puzzle":
            zebra_evaluator = get_zebra_puzzle_evaluator(question["livebench_release_date"])
            score = zebra_evaluator(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == "spatial":
            score = spatial_process_results(ground_truth, llm_answer, debug)
            category = "reasoning"
        elif task_or_subtask == 'typos':
            score = typos_process_results(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask == "connections":
            connections_evaluator = get_connections_puzzle_evaluator(question["livebench_release_date"])
            score = connections_evaluator(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask == "plot_unscrambling":
            score = plot_unscrambling_process_results(ground_truth, llm_answer, debug)
            category = "language"
        elif task_or_subtask in coding_test_case_tasks:
            # use entire question object, because there are test cases inside.
            score = LCB_generation_process_results(question, llm_answer, debug)
            category = "coding"
        else:
            raise NotImplementedError(f"This task ({task_or_subtask}) has not been implemented yet.")
    except Exception as e:
        raise RuntimeError(f"Error occurred evaluating question {question['question_id']}") from e

    if not category:
        raise NotImplementedError(f"A category must be assigned to each task")
    question_id = question["question_id"]
    turn = 1
    result = {
        "question_id": question_id,
        "task": task,
        "model": model,
        "score": score,
        "turn": turn,
        "tstamp": time.time(),
        "category": category,
    }
    if "subtask" in question.keys():
        result["subtask"] = question["subtask"]
    print(
        f"question: {question_id}, turn: {turn}, model: {model}, "
        f"score: {score}, "
       
    )

    # if output_file:
    #     if '/' in output_file:
    #         os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #     with open(output_file, "a") as fout:
    #         fout.write(json.dumps(result) + "\n")

    return score





import glob, os
from tqdm import tqdm
current_path = os.getcwd()

import re
import pandas as pd

from freeEvalLM._lib._df import read_file, save_file
from freeEvalLM.src.Evaluator import Evaluator


def convert_booleans_list(lst):
    all = []
    for x in lst:
        for _x in x:
            all.append(int(_x))
    return all



def process_result(docs, output_file):

    task_name = docs[0]["task"]
    print("task name")
    print(task_name)
    if task_name in ["cta", "tablejoin", "tablereformat", "connections", "plot_unscrambling", "typos"]:
        scores = []
        for doc in tqdm(docs):
            # print(doc)
            match = MatchSingle(question=doc, model="modelname", answer=doc["response"])
            scores.append(play_a_match_gt(match, output_file=output_file))
        # # print(scores)
        # all_res = []
        # for score in scores:
        #     all_res.append(score["score"])
        # return all_res
        return scores
        
    else:
        questions = []
        model_answers = []

        for doc in docs:
            if doc["response"]:
                pass
            else:
                doc["response"] = "None"
            model_answers.append(
                {

                        
                            "question_id": doc["question_id"],
                            "model_id": "model_id",
                            "choices": [{"index":0, "turns": [doc["response"]]}]
                        
                }
            )

        model_id = "model_id"
        debug = False

        # print(docs)
        # print(model_answers)
        scores = instruction_following_process_results(docs,  model_answers, task_name, model_id, debug)
        # print(scores)
        all_res = []
        for score in scores:
            all_res.append(score["score"])
        return all_res





class livebench(Evaluator):
    def __init__(self, 
        task, 
        result_dir
        ):
        self.task = task
        self.result_dir = result_dir
        self.none = 0


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

            # print(data)
            docs = data.to_dict(orient="records")
            
            answers = data["response"]

            # print(answers[0])

            # for doc in docs:
            res = process_result(docs, os.path.join(os.path.dirname(data_path), name+"_process.json") )
            if isinstance(res, list):
                print("#########")
                print(res)
                scores = res
            else:
                scores = res 


            # print(scores)
            final_ = 0
            for score in scores:
                final_ += score
            df = pd.DataFrame({
                'filtered_answer': answers,
                'score': scores
            })
            # all_finals.append(sum(scores)/len(scores)*100)
            all_finals.append(final_/len(scores)*100)
            
            data = data.join(df)
            # print(os.path.dirname(data_path))
            save_file(data, os.path.join(os.path.dirname(data_path), name+".json"))

        data_analysis = 0
        instruction_following = 0
        language = 0
        for name, final in zip(all_names, all_finals):
            if name in ["cta", "tablejoin", "tablereformat"]:
                data_analysis += final
            elif name in ["connections", "typos", "plot_unscrambling"]:
                language += final
            else:
                instruction_following += final


        

        all_names += ["","data_analysis", "instruction_following", "language", "", "FINAL"]
        all_finals += ["",data_analysis/3, instruction_following/4, language/3,"", (data_analysis/3 + instruction_following/4 + language/3)/3]



        df = pd.DataFrame({
            'subtask': all_names,
            'score': all_finals
        })    

        save_file(df, os.path.join(os.path.dirname(data_path), "Results.csv"))


if __name__ == "__main__":

    pass




