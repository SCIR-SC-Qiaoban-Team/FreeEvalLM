import json
import os
import pickle

def load_json(file_path):

    with open(file_path) as f:
        data_dict_list = json.load(f)

    return data_dict_list

def load_jsonl(file_path):
    json_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                json_list.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"jsonl error: {e}")
    return json_list

def save_to_json(data_dict, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, "w") as f:
        json.dump(data_dict, f, indent=4, ensure_ascii=False)

