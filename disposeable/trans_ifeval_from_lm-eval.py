import os, sys
import glob
import sys

import freeEvalLM
from freeEvalLM._lib._json import *
from freeEvalLM._lib._pkl import *



lmevalLogSamplePath = "/share/home/wxzhao/gjh_ws/Code/LM-eval/results/250302_IFEval/qwen25_7b_ins"

sample_jsons = glob.glob(os.path.join(lmevalLogSamplePath, "sample*.jsonl"))


for sample_json in sample_jsons:
    all_items = []
    sample = load_jsonl(sample_json)
    # category = sample[0]["doc"]["category"]
    for line in sample:
        _key = line["doc"]["key"]
        _input = line["arguments"]["gen_args_0"]["arg_0"]
        _target = line["target"]
        _kwargs = line["doc"]["kwargs"]
        _instruction_id_list = line["doc"]["instruction_id_list"]
        _response = line["filtered_resps"]

        _dict = {
            "input": _input,
            "key": _key,
            "target": _target,
            "kwargs": _kwargs,
            "instruction_id_list": _instruction_id_list,
            "response": _response
        }
        
        # print(_dict)
        # exit()
        all_items.append(_dict)

    save_to_json(all_items, os.path.join("/share/home/wxzhao/gjh_ws/Code/FreeEvalLM/scripts/_examples/caozhi.json"))


