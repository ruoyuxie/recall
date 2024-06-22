from datasets import load_dataset
import os
import random
import numpy as np
import json
   
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for example in data:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

def create_dataset(dataset_name,sub_dataset_name, output_dir, num_shots):
    
    if dataset_name == "wikimia":
        dataset = load_dataset("swj0419/WikiMIA", split=f"WikiMIA_length{sub_dataset_name}")
        member_data = []
        nonmember_data = []
        for data in dataset:
            if data["label"] == 1:
                member_data.append(data["input"])
            elif data["label"] == 0:
                nonmember_data.append(data["input"])
        dataset_folder = os.path.join(output_dir, f"{sub_dataset_name}", "data")
        
        # shuffle the datasets
        random.shuffle(member_data)
        random.shuffle(nonmember_data)
        num_shots = int(num_shots)

        nonmember_prefix = nonmember_data[:num_shots]
        nonmember_data = nonmember_data[num_shots:]
        member_data_prefix = member_data[:num_shots]
        member_data = member_data[num_shots:]
        
        # # Create the output directory
        # os.makedirs(dataset_folder, exist_ok=True)

        # data_folder = os.path.join(dataset_folder,  "target_data")  
        # os.makedirs(data_folder, exist_ok=True)

        # prefix_folder_path = os.path.join(dataset_folder,  "prefixes")   
        # os.makedirs(prefix_folder_path, exist_ok=True)

        # # save the original datasets
        # member_data_path = os.path.join(data_folder, "train.jsonl")
        # nonmember_data_path = os.path.join(data_folder, "test.jsonl")\
            
        # save_jsonl(member_data, member_data_path)
        # save_jsonl(nonmember_data, nonmember_data_path)
        # save_jsonl(nonmember_prefix, os.path.join(prefix_folder_path, "nonmember_prefix.jsonl"))
        # save_jsonl(member_data_prefix, os.path.join(prefix_folder_path, "member_prefix.jsonl"))
        
    else: 
        raise ValueError(f"Unknown dataset: {dataset_name}. Please modify the code to include the dataset. Make sure the dataset is in the same format.")

    full_data = [] 
    # binary classification, the data need to be balanced. 
    for nm_data, m_data in zip(nonmember_data, member_data):
        full_data.append({"input": nm_data, "label": 0})
        full_data.append({"input": m_data, "label": 1})
    
    return full_data, nonmember_prefix, member_data_prefix
