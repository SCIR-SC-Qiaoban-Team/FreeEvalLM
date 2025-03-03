import json
import os
import pickle



def load_from_pickle(file_path):
    if not os.path.exists(file_path):
        return None
    # logging.info(f"Loading hidden states from {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_to_pickle(data, file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    # logging.info(f"Saving hidden states to {file_path}")
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
        


