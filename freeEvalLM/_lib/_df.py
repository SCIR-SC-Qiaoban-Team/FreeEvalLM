import os

import pandas as pd


def read_file(file_name):
    _, file_extension = os.path.splitext(file_name)
    
    if file_extension == '.csv':
        df = pd.read_csv(file_name)
    elif file_extension == '.json':
        df = pd.read_json(file_name)
    elif file_extension == '.jsonl':
        df = pd.read_json(file_name, lines=True)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    return df

def save_file(df: pd.DataFrame, file_path):
    _, file_extension = os.path.splitext(file_path)
    print(os.path.splitext(file_path))
    
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