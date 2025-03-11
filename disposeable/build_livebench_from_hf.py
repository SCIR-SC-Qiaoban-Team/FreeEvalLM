import os, sys
import glob
import sys
from datasets import load_dataset, Dataset, load_from_disk

import freeEvalLM
from freeEvalLM._lib._json import *
from freeEvalLM._lib._pkl import *




LIVE_BENCH_CATEGORIES = [
    "data_analysis",
    "instruction_following",
    "language",
]

task_lists = [
    ["cta", "tablejoin", "tablereformat"],
    ["paraphrase", "simplify", "story_generation", "summarize"],
    ["connections", "plot_unscrambling", "typos"]
]




for _LIVE_BENCH_CATEGORIE, task_list in zip(LIVE_BENCH_CATEGORIES, task_lists):
    all_data = load_from_disk(f"/share/home/wxzhao/gjh_ws/Code/LM-eval/resources/livebench/{_LIVE_BENCH_CATEGORIE}")["test"]
    for task in task_list:
        task_spec = []
        for item in all_data:
            if item["task"] == task:
                item["livebench_release_date"] = str(item["livebench_release_date"])
                item["input"] =  str(item["turns"][0])
                task_spec.append(item)
        # print(task_spec)
        print(len(task_spec))
        print(task_spec[0])
        save_to_json(task_spec, os.path.join(f"/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/datasets/livebench/{_LIVE_BENCH_CATEGORIE}_{task}.json"))
