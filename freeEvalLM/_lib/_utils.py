import os
import json
import time
import functools
import concurrent.futures

from openai import OpenAI
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()


def read_file(file_name):
    _, file_extension = os.path.splitext(file_name)
    
    if file_extension == '.csv':
        df = pd.read_csv(file_name)
    if file_extension == '.tsv':
        df = pd.read_csv(file_name, sep='\t')
    elif file_extension == '.json':
        df = pd.read_json(file_name)
    elif file_extension == '.jsonl':
        df = pd.read_json(file_name, lines=True)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    return df

def save_file(df: pd.DataFrame, file_path):
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.csv':
        df.to_csv(file_path, index=False)
        print(f"DataFrame successfully saved as CSV: {file_path}")
    
    elif file_extension == '.json':
        df.to_json(file_path, orient='records', lines=False, indent=2, force_ascii=False)
        print(f"DataFrame successfully saved as JSON: {file_path}")
    
    elif file_extension == '.jsonl':
        df.to_json(file_path, orient='records', lines=True, force_ascii=False)
        print(f"DataFrame successfully saved as JSONL: {file_path}")
    
    elif file_extension == '.xlsx' or file_extension == '.xls':
        df.to_excel(file_path, index=False)
        print(f"DataFrame successfully saved as Excel: {file_path}")
    
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

openai = OpenAI()

def generate(query, system=None, usage=False):
    global openai
    if openai.is_closed():
        openai = OpenAI()
    msg = [{"role": "user", "content": query}]
    if system:
        msg.insert(0, {"role": "system", "content": system})
    chat_completion = openai.chat.completions.create(
        # model="Qwen/Qwen2.5-72B-Instruct",
        model="gpt-4o-mini",
        messages=msg,
        stream=False,
    )
    if usage:
        return chat_completion.choices[0].message.content, chat_completion.usage.prompt_tokens, chat_completion.usage.completion_tokens
    return chat_completion.choices[0].message.content

def retry(max_attempts=None, delay=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while max_attempts is None or attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print('err:', e, '\n')
                    last_exception = e
                    attempts += 1
                    if delay > 0:
                        time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

@retry()
def _gen(item):
    item['response'] = generate(item['query'], item['system'])
    return item

def generate_batch(query_list, sys_list, threads=20):
    assert len(query_list) == len(sys_list)
    futures = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        for i, (q, s) in enumerate(zip(query_list, sys_list)):
            futures.append(executor.submit(_gen, {'query': q, 'system': s, 'idx': i}))
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    results = sorted(results, key=lambda x:x['idx'])
    return [r['response'] for r in results]


