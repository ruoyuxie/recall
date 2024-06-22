# This implementation is adapted from Min-K% and WikiMIA: https://github.com/swj0419/detect-pretrain-code 
import json
import os
import random
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# plot data
def sweep(score, labels):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, ACC, TPR@10%FPR, and TPR@20%FPR.
    """
    fpr, tpr, _ = roc_curve(labels, score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    auc_roc = auc(fpr, tpr)
    return fpr, tpr, auc_roc, acc


def do_plot(prediction, answers, sweep_fn=sweep, metric="auc", legend=""):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, auc_value, acc = sweep_fn(
        np.array(prediction), np.array(answers, dtype=bool)
    )

    low01 = tpr[np.where(fpr < 0.001)[0][-1]]
    low1 = tpr[np.where(fpr < 0.01)[0][-1]]
    low5 = tpr[np.where(fpr < 0.05)[0][-1]]
    low10 = tpr[np.where(fpr < 0.1)[0][-1]]
    low20 = tpr[np.where(fpr < 0.2)[0][-1]]
    print('Attack %s AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f, TPR@1%%FPR of %.4f, TPR@5%%FPR of %.4f'%(legend, auc_value, acc, low01, low1, low5))
    
    metric_text = ""
    if metric == "auc":
        metric_text = "auc=%.3f" % auc_value
    elif metric == "acc":
        metric_text = "acc=%.3f" % acc
    plt.plot(fpr, tpr, label=legend + metric_text)
    return legend, auc_value, acc, low01, low1, low5, low10, low20


def fig_fpr_tpr(all_output, result_path):
    answers = []
    metric2predictions = defaultdict(list)
    for ex in all_output:
        answers.append(ex["label"])
        for metric in ex["pred"].keys():
            if ("raw" in metric) and ("clf" not in metric):
                continue
            metric2predictions[metric].append(ex["pred"][metric])
            # if the metric is all nan, then just append 0 to it to avoid error
            if np.isnan(ex["pred"][metric]).all():
                metric2predictions[metric][-1] = 0

    plt.figure(figsize=(4, 3))

    results = {}
    with open(result_path, "w") as f:
        average_results = {}

        for metric, predictions in metric2predictions.items():
            if "all_prob" in metric:
                continue
            predictions = np.clip(predictions, -np.finfo(np.float64).max, np.finfo(np.float64).max)
            # get the probability of the positive class based on label=1 in answers list
            prob_class_1 = [
                pred for pred, label in zip(predictions, answers) if label == 1
            ]
            # get the probability of the negative class based on label=0 in answers list
            prob_class_0 = [
                pred for pred, label in zip(predictions, answers) if label == 0
            ]

            # get the average probability of the positive class
            prob_class_1_avg = np.mean(prob_class_1)
            # get the average probability of the negative class
            prob_class_0_avg = np.mean(prob_class_0)

            average_results[metric] = {
                "prob_class_1_avg": prob_class_1_avg,
                "prob_class_0_avg": prob_class_0_avg,
                "prob_class_avg_1_0_diff": abs(prob_class_1_avg - prob_class_0_avg),
                "prob_class_1_std": np.std(prob_class_1),
                "prob_class_0_std": np.std(prob_class_0),
                "prob_class_1_median": np.median(prob_class_1),
                "prob_class_0_median": np.median(prob_class_0),
            }

            legend, auc_value, acc, low01, low1, low5, low10, low20 = do_plot(
                predictions, answers, legend=metric, metric="auc"
            )
            # f.write('%s AUC %.4f, Accuracy %.4f, TPR@0.01%%FPR of %.4f, TPR@0.1%%FPR of %.4f, TPR@0.5%%FPR of %.4f, TPR@1%%FPR of %.4f, TPR@2%%FPR of %.4f\n'%(legend, auc_value, acc, low01, low1, low5, low10, low20))
            results[legend] = {
                "AUC-ROC": round(auc_value, 3),
                "ACC": round(acc, 3),
                "TPR@0.1%FPR": round(low01, 3),
                "TPR@1%FPR": round(low1, 3),
                "TPR@5%FPR": round(low5, 3),
                "TPR@10%FPR": round(low10, 3),
                "TPR@20%FPR": round(low20, 3),
            }
            
        # combine all the results 
        results["average"] = average_results
        # save the results to the output file
        json.dump(results, f, indent=4)


def load_jsonl(input_path):
    with open(input_path, "r") as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data


def dump_jsonl(data, path):
    with open(path, "w") as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data


def load_and_convert_to_list_dic(data_folder_path, max_data_points):
    data_name = os.path.basename(data_folder_path)
    print(f"loading data from {data_name}")
    test_file = os.path.join(data_folder_path, "test.jsonl")
    train_file = os.path.join(data_folder_path, "train.jsonl")

    test_data = load_jsonl(test_file)
    train_data = load_jsonl(train_file)

    list_dic = []
    max_length = min(len(test_data), len(train_data))

    for test_ex, train_ex in zip(test_data[:max_length], train_data[:max_length]):
        list_dic.append({"input": test_ex, "label": 0})
        list_dic.append({"input": train_ex, "label": 1})

    return list_dic[:max_data_points]
