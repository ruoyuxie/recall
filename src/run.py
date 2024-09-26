# This implementation is adapted from Min-K% and WikiMIA: https://github.com/swj0419/detect-pretrain-code 
import os
from tqdm import tqdm
from process_data import create_dataset
import torch 
from options import Options
import numpy as np
import torch
import zlib
from eval import *
import os
import torch.nn.functional as F
from visualization import analyze_final_results
from transformers import set_seed
import torch 
import random
import openai 
from accelerate import Accelerator
import torch 
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    MambaForCausalLM,
)

def load_model(name1, name2, use_float16=True):    
    accelerator = Accelerator()

    def load_specific_model(name, use_float16=False):
        if "mamba" in name:
            model = MambaForCausalLM.from_pretrained(
                name, return_dict=True, trust_remote_code=True,
                torch_dtype=torch.float16 if use_float16 else torch.float32,
                device_map="cuda:0"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                name, return_dict=True, trust_remote_code=True,
                torch_dtype=torch.float16 if use_float16 else torch.float32,
                device_map="auto"
            )
        return model

    # Load the first model
    model1 = load_specific_model(name1, use_float16)
    tokenizer1 = AutoTokenizer.from_pretrained(name1)
    # Load the second model with the same float16 setting as model1
    model2 = load_specific_model(name2, use_float16)
    tokenizer2 = AutoTokenizer.from_pretrained(name2)

    model1.eval()
    model2.eval()

    model1, model2 = accelerator.prepare(model1, model2)

    return model1, model2, tokenizer1, tokenizer2, accelerator

def api_key_setup(key_path):
    openai.api_key = open(key_path, "r").read()

def fix_seed(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def get_ll(sentence, model, tokenizer, device):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return get_all_prob(input_ids, loss, logits)

def get_conditional_ll(input_text, target_text, model, tokenizer, device):
    input_encodings = tokenizer(input_text, return_tensors="pt")
    target_encodings = tokenizer(target_text, return_tensors="pt")
    concat_ids = torch.cat((input_encodings.input_ids.to(device), target_encodings.input_ids.to(device)), dim=1)
    labels = concat_ids.clone()
    labels[:, : input_encodings.input_ids.size(1)] = -100
    with torch.no_grad():
        outputs = model(concat_ids, labels=labels)
    loss, logits = outputs[:2]
    return get_all_prob(labels, loss, logits)

def get_all_prob(input_ids, loss, logits):
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    ll = -loss.item()  # log-likelihood
    ppl = torch.exp(loss).item()
    prob = torch.exp(-loss).item()
    return prob, ll , ppl, all_prob, loss.item()

def inference(model1, model2, tokenizer1, tokenizer2, target_data, prefix, accelerator, num_shots, ex):
    pred = {}
    
    # unconditional log-likelihood
    ll = get_ll(target_data, model1, tokenizer1,accelerator.device)[1]
     
    # ReCaLL
    if int(num_shots) != 0:   
        # conditional log-likelihood with prefix     
        ll_nonmember = get_conditional_ll("".join(prefix), target_data, model1, tokenizer1, accelerator.device)[1]
        pred["recall"] = ll_nonmember / ll 

    # baselines 
    input_ids = torch.tensor(tokenizer1.encode(target_data)).unsqueeze(0).to(accelerator.device)
    with torch.no_grad():
        outputs = model1(input_ids, labels=input_ids)
    _, logits = outputs[:2]
    ll_ref = get_ll(target_data, model2, tokenizer2, accelerator.device)[1]

    # loss and zlib
    pred["ll"] = ll
    pred["ref"] = ll - ll_ref
    pred["zlib"] = ll / len(zlib.compress(bytes(target_data, "utf-8")))

    # For mink and mink++
    input_ids = input_ids[0][1:].unsqueeze(-1)
    probs = F.softmax(logits[0, :-1], dim=-1)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=input_ids).squeeze(-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)

    ## mink
    for ratio in [0.2]:
        k_length = int(len(token_log_probs) * ratio)
        topk = np.sort(token_log_probs.cpu())[:k_length]
        pred[f"mink_{ratio}"] = np.mean(topk).item()
        
    ## mink++
    mink_plus = (token_log_probs - mu) / sigma.sqrt()
    for ratio in [0.2]:
        k_length = int(len(mink_plus) * ratio)
        topk = np.sort(mink_plus.cpu())[:k_length]
        pred[f"mink++_{ratio}"] = np.mean(topk).item()

    ex["pred"] = pred
    return ex

def generate_prompt(example):
    return f"Generate a passage that is similar to the given text in length, domain, and style.\nGiven text:{example}\nPassage :"

def get_completion(prompt):
    message = [{"role": "user", "content": prompt}]
    responses = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=message,
        max_tokens=1024,
        temperature=1,
    )
    return responses.choices[0]["message"]["content"]

def gpt_synthetic_prefix (original_prefixes):
    # default generate synthetic prefix from non-member data
    synthetic_prefixes = []
    for original_prefix in original_prefixes:
        prompt = generate_prompt(original_prefix)
        response = get_completion(prompt)      
        synthetic_prefixes.append(response)
    return synthetic_prefixes

    
def process_prefix(target_model, prefix, avg_length, pass_window, total_shots):
    if pass_window:
        return prefix
    max_length = model1.config.max_position_embeddings if "mamba" not in target_model else 2048
    token_counts = [len(tokenizer1.encode(shot)) for shot in prefix]
    target_token_count = avg_length
    total_tokens = sum(token_counts) + target_token_count
    if total_tokens <= max_length:
        return prefix
    # Determine the maximum number of shots that can fit within the max_length
    max_shots = 0
    cumulative_tokens = target_token_count
    for count in token_counts:
        if cumulative_tokens + count <= max_length:
            max_shots += 1
            cumulative_tokens += count
        else:
            break
    # Truncate the prefix to include only the maximum number of shots
    truncated_prefix = prefix[-max_shots:]
    total_shots = max_shots
    return truncated_prefix

def evaluate_data(test_data, model1, model2, tokenizer1, tokenizer2, prefix, accelerator, total_shots, pass_window, synehtic_prefix):
    all_output = []
    if int(total_shots) != 0:   
        avg_length = int(np.mean([len(tokenizer1.encode(ex["input"])) for ex in test_data])) 
        prefix = process_prefix(target_model, prefix, avg_length, pass_window, total_shots) 
        if synehtic_prefix:
            prefix = gpt_synthetic_prefix(prefix)
    for ex in tqdm(test_data):
        new_ex = inference(model1, model2, tokenizer1, tokenizer2, ex["input"], prefix, accelerator, total_shots, ex)
        all_output.append(new_ex)
    return all_output

if __name__ == "__main__":
    fix_seed(42)
    args = Options()
    args = args.parser.parse_args()

    output_dir = args.output_dir
    dataset = args.dataset
    target_model = args.target_model
    ref_model = args.ref_model
    sub_dataset = args.sub_dataset
    num_shots = args.num_shots
    pass_window = args.pass_window
    synehtic_prefix = args.synehtic_prefix
    api_key_path = args.api_key_path

    if synehtic_prefix and api_key_path is not None:
        api_key_setup(api_key_path)

    # process and prepare the data
    full_data, nonmember_prefix, member_data_prefix = create_dataset(dataset, sub_dataset, output_dir, num_shots)

    # load models
    model1, model2, tokenizer1, tokenizer2, accelerator = load_model(target_model, ref_model)

    # evaluate the data
    all_output = evaluate_data(full_data, model1, model2, tokenizer1, tokenizer2, nonmember_prefix, accelerator, num_shots, pass_window, synehtic_prefix)
    
    # save the results
    all_output_path = os.path.join(output_dir, f"{dataset}", f"{target_model.split('/')[1]}_{ref_model.split('/')[1]}", f"{sub_dataset}", f"{num_shots}_shot_{sub_dataset}.json")
    os.makedirs(os.path.dirname(all_output_path), exist_ok=True)
    dump_jsonl(all_output, all_output_path)
    print(f"Saved results to {all_output_path}")
    
    # evaluate the results
    fig_fpr_tpr(all_output, all_output_path)        
    
    # result visualizations to show 0 to n shot results - make sure you have these results 
    analyze_final_results(os.path.join(output_dir, f"{dataset}", f"{target_model.split('/')[1]}_{ref_model.split('/')[1]}", f"{sub_dataset}"), show_values=True)
            