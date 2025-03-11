import os
from typing import Annotated, List, Optional, Literal, Union

import fire
import torch
import pandas as pd

from easy_vllm import InferenceModel, ChatParam, ChatExtraParam, GenParam, GenExtraParam
from _lib._df import read_file, save_file
from Task import TaskLoader


def decode_query(model: InferenceModel, query_list: Union[list[str], list[list[str]]], system: str=None, threads: int=20, reasoning_max_retry: int=10, param: ChatParam = None, ext_param: ChatExtraParam = None) -> Union[list[str], list[list[str]]]:
    msg_list = [[] for _ in query_list]
    if system:
        msg_list = [[{'role': 'system', 'content': system}] for _ in query_list]

    if type(query_list[0]) == str:
        msg_list = [msg + [{'role': 'user', 'content': d}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat(msg_list, threads, reasoning_max_retry=reasoning_max_retry, param=param, ext_param=ext_param)
        return responses
    
    response_list = []
    for i in range(len(query_list[0])):
        msg_list = [msg + [{'role': 'user', 'content': d[i]}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat(msg_list, threads, reasoning_max_retry=reasoning_max_retry, param=param, ext_param=ext_param)
        response_list.append(responses)
        if type(query_list[0]) == str:
            msg_list = [msg + [{'role': 'assistant', 'content': r}] for msg, r in zip(msg_list, responses)]
        else:
            msg_list = [msg + [{'role': 'assistant', 'content': r[1]}] for msg, r in zip(msg_list, responses)]
    response_list = [list(i) for i in zip(*response_list)]
    return response_list


def decode_query_reasoning_ctrl(model: InferenceModel, query_list: Union[list[str], list[list[str]]], system: str=None, threads: int=20, reasoning_max_retry=10, add_reasoning_prompt: bool = False, enable_length_ctrl: bool = False, reasoning_max_len: int=None, reasoning_min_len: int=0, reasoning_scale: float=None, cut_by_sentence=False, param: ChatParam = None, ext_param: ChatExtraParam = None) -> Union[list[str], list[list[str]]]:
    msg_list = [[] for _ in query_list]
    if system:
        msg_list = [[{'role': 'system', 'content': system}] for _ in query_list]

    if type(query_list[0]) == str:
        msg_list = [msg + [{'role': 'user', 'content': d}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat_custom(msg_list, threads, reasoning_max_retry, add_reasoning_prompt, enable_length_ctrl, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        return responses
    
    response_list = []
    for i in range(len(query_list[0])):
        msg_list = [msg + [{'role': 'user', 'content': d[i]}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat_custom(msg_list, threads, reasoning_max_retry, add_reasoning_prompt, enable_length_ctrl, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        response_list.append(responses)
        msg_list = [msg + [{'role': 'assistant', 'content': r[1]}] for msg, r in zip(msg_list, responses)]
    response_list = [list(i) for i in zip(*response_list)]
    return response_list


def decode_query_force_reasoning_content(model: InferenceModel, query_list: Union[list[str], list[list[str]]], reasoning_content_lines: Union[list[str], list[list[str]]], system: str=None, threads: int=20, reasoning_scale: float=None, cut_by_sentence: bool = False, param: ChatParam = None, ext_param: ChatExtraParam = None) -> Union[list[str], list[list[str]]]:
    msg_list = [[] for _ in query_list]
    if system:
        msg_list = [[{'role': 'system', 'content': system}] for _ in query_list]

    if type(query_list[0]) == str:
        assert type(reasoning_content_lines[0]) == str
        msg_list = [msg + [{'role': 'user', 'content': d}] for msg, d in zip(msg_list, query_list)]
        responses = model.parallel_chat_force_reasoning_content(msg_list, reasoning_content_lines, threads, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        return responses
    
    response_list = []
    for i in range(len(query_list[0])):
        msg_list = [msg + [{'role': 'user', 'content': d[i]}] for msg, d in zip(msg_list, query_list)]
        reasoning_content = [r[i] for r in reasoning_content_lines]
        responses = model.parallel_chat_force_reasoning_content(msg_list, reasoning_content, threads, reasoning_scale, cut_by_sentence, param=param, ext_param=ext_param)
        response_list.append(responses)
        msg_list = [msg + [{'role': 'assistant', 'content': r}] for msg, r in zip(msg_list, responses)]
    response_list = [list(i) for i in zip(*response_list)]
    return response_list



def main(
        model_path,
        save_path,
        decode_type: Literal['query', 'query_reasoning_ctrl', 'query_force_reasoning_content'],
        file_path: str = None,
        task: str = None,
        sample: int = -1,
        query_keys: str = None,
        response_keys: str = None,
        reasoning_keys: str = None,
        tensor_parallel_size: int = 1,
        model_num: int = None,
        port: int = 50000,
        max_model_len: int = None,
        show_log: bool = True,
        timeout: int = 30,
        threads: int=20,
        enable_reasoning: bool = False,
        reasoning_parser: str = 'deepseek_r1',
        system_prompt_file: str = None,
        chat_template_file: str = None,
        max_new_tokens = 8192,
        device_ids: str = None,
        reasoning_max_retry: int = 10,
        add_reasoning_prompt: bool = False,
        enable_length_ctrl: bool = False,
        reasoning_max_len: int = None,
        reasoning_min_len: int = 0,
        reasoning_scale: float = None,
        cut_by_sentence: bool = False,
        force_reasoning_content_keys: str = None,
        overwrite: bool = False,
):


    print("decode")
    print("enable_reasoning", enable_reasoning)
    assert query_keys or decode_type != 'query', 'set `query_keys` when use query decode'
    
    all_dfs = []
    
    if file_path == None:
        print(f"Using predefined tasks: [{task}]")
        taskLoader = TaskLoader(task, save_path, sample)
        all_dfs = taskLoader.load_dataset()
        evaluator = taskLoader.load_evaluator()
    else:
        df = read_file(file_path)
        all_dfs = [{
            "subtask_name": "file_path",
            "subtask_data": df
        }]

    system = None
    if system_prompt_file:
        with open(system_prompt_file) as f:
            system = f.read()

    if device_ids:
        if type(device_ids) != tuple:
            device_ids = [device_ids]
        device_ids = [int(d) for d in device_ids]
        assert model_num is None or len(device_ids) >= tensor_parallel_size * model_num
        assert len(device_ids) >= tensor_parallel_size
        if model_num:
            device_ids = device_ids[:tensor_parallel_size * model_num]
    else:
        if not model_num:
            model_num = torch.cuda.device_count() // tensor_parallel_size
        device_ids = list(range(min(torch.cuda.device_count(), model_num*tensor_parallel_size)))
    model = InferenceModel(model_path, device_ids, tensor_parallel_size, port, max_model_len, show_log, timeout, enable_reasoning, reasoning_parser, chat_template_file)

    query_keys = [query_keys] if type(query_keys) != tuple else list(query_keys)
    response_keys = ([response_keys] if type(response_keys) != tuple else list(response_keys)) if response_keys else ['resp_'+q for q in query_keys]
    reasoning_keys = ([reasoning_keys] if type(reasoning_keys) != tuple else list(reasoning_keys)) if reasoning_keys else ['reas_'+q for q in query_keys]
    
    
    for item in all_dfs:

        df = item["subtask_data"]
        subtask_name = item["subtask_name"]
        print(f"running on {subtask_name}")


        if decode_type in ['query', 'query_reasoning_ctrl', 'query_force_reasoning_content']:

            query_lines = df[query_keys].values.tolist()
            if decode_type == 'query':
                responses = decode_query(model, query_lines, system, threads*model_num, reasoning_max_retry, param=ChatParam(max_completion_tokens=max_new_tokens))
            elif decode_type == 'query_reasoning_ctrl':
                assert enable_reasoning
                responses = decode_query_reasoning_ctrl(model, query_lines, system, threads*model_num, reasoning_max_retry, add_reasoning_prompt, enable_length_ctrl, reasoning_max_len, reasoning_min_len, reasoning_scale, cut_by_sentence, param=GenParam(max_tokens=max_new_tokens))
            elif decode_type == 'query_force_reasoning_content':
                assert force_reasoning_content_keys
                assert enable_reasoning
                force_reasoning_content_keys = [force_reasoning_content_keys] if type(force_reasoning_content_keys) != tuple else list(force_reasoning_content_keys)
                reasoning_keys = force_reasoning_content_keys
                
                reasoning_content_lines = df[force_reasoning_content_keys].values.tolist()
                responses = decode_query_force_reasoning_content(model, query_lines, reasoning_content_lines, system, threads*model_num, reasoning_scale, cut_by_sentence, param=GenParam(max_tokens=max_new_tokens))
            
            if enable_reasoning:
                if not isinstance(responses[0], list):
                    reasonings = [r[0] for r in responses]
                    responses = [r[1] for r in responses]
                else:
                    reasonings = [[i[0] for i in r] for r in responses]
                    responses = [[i[1] for i in r] for r in responses]
                resp_df = pd.DataFrame(responses, columns=response_keys)
                resn_df = pd.DataFrame(reasonings, columns=reasoning_keys)
                
                if overwrite:
                    df = df.drop(response_keys + reasoning_keys, axis=1, errors='ignore')
                df = df.join(resp_df).join(resn_df)
            else:
                resp_df = pd.DataFrame(responses, columns=response_keys)
                if overwrite:
                    df = df.drop(response_keys, axis=1, errors='ignore')
                df = df.join(resp_df)
            
        
        result_path = os.path.join(save_path, subtask_name + ".json")
        print("saving to ", result_path)
        os.makedirs(os.path.split(result_path)[0], exist_ok=True)
        save_file(df, result_path)


    evaluator.load_results()
    evaluator.evaluate()



if __name__ == "__main__":
    fire.Fire(main)

