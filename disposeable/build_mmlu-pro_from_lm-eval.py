import os, sys
import glob
import sys

import freeEvalLM
from freeEvalLM._lib._json import *
from freeEvalLM._lib._pkl import *



lmevalLogSamplePath = "/share/home/wxzhao/gjh_ws/Code/LM-eval/results/250219/qwen25_7b_real0_4096/__share__home__wxzhao__gjh_ws__Downloads__LLMs__Qwen2.5-7B-Instruct"

sample_jsons = glob.glob(os.path.join(lmevalLogSamplePath, "sample*.jsonl"))
# print(sample_jsons)


for sample_json in sample_jsons:
    all_items = []
    sample = load_jsonl(sample_json)
    category = sample[0]["doc"]["category"]
    for line in sample:
        _question_id = line["doc"]["question_id"]
        _input = line["arguments"]["gen_args_0"]["arg_0"]
        _target = line["target"]

        _dict = {
            "input": _input,
            "question_id": _question_id,
            "target": _target
        }
        
        # print(_dict)
        # exit()
        all_items.append(_dict)

    save_to_json(all_items, os.path.join("/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/datasets/mmlu_pro", category + ".json"))


