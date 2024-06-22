import os
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams.update({'font.size': 17})
plt.rcParams.update({'axes.labelsize': 20})
plt.rcParams.update({'xtick.labelsize': 17})
plt.rcParams.update({'ytick.labelsize': 17})
plt.rcParams.update({'legend.fontsize': 17})

def process_json_files(folder_path, show_values=False):
    attack_name_mapping = {
        "recall": "ReCaLL",
        "ll" : "Loss",
        "mink++_0.2": "Min-K++",
        "mink_0.2": "Min-K",
        "zlib": "Zlib",
        "ref": "Ref",
    }

    attack_colors = {
    "ReCaLL": "blue",
    "Loss": "red",
    "Min-K++": "green",
    "Min-K": "purple",
    "Zlib": "orange",
    "Ref": "brown",
    }

    results = []

    # Walk through the folder and process each JSON file
    for root, _, files in os.walk(folder_path):
        for file in files:
            if "_metrics" in file:
                continue
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_parts = file.split("_shot_")
                
                if len(file_parts) == 2:
                    num_shots = int(file_parts[0])
                    sub_data_category = file_parts[1].replace('.json', '')
                    
                    for key, values in data.items():
                        if key != "average":
                            if key == "recall":
                                result_row = [num_shots, sub_data_category, key, values["AUC-ROC"]*100]
                                results.append(result_row)
                            else:
                                result_row = [0, sub_data_category, key, values["AUC-ROC"]*100]
                                results.append(result_row)
    # Sort results by Sub-category, then Attack, then Shots
    results.sort(key=lambda x: (x[1], x[2], x[0]))

    # Convert results to a dictionary for easier plotting
    results_dict = {}
    for result in results:
        shots, sub_category, attack, auc_roc = result
        attack = attack_name_mapping.get(attack, attack)
        if sub_category not in results_dict:
            results_dict[sub_category] = {}
        if attack not in results_dict[sub_category]:
            results_dict[sub_category][attack] = {'shots': [], 'auc_roc': []}
        results_dict[sub_category][attack]['shots'].append(shots)
        results_dict[sub_category][attack]['auc_roc'].append(auc_roc)

    output_folder = os.path.join(folder_path, "graphs")
    os.makedirs(output_folder, exist_ok=True)

    for sub_category, attacks in results_dict.items():
        fig, ax = plt.subplots(figsize=(8, 5.5))   
        legend_labels = []
        legend_handles = []
        for attack, values in sorted(attacks.items()):
            shots = np.array(values['shots'])
            auc_roc = np.array(values['auc_roc'])
            color = attack_colors.get(attack, "gray")

            if attack == "ReCaLL":
                if len(shots) == 1:
                    scatter = ax.scatter(shots, auc_roc, color=color)
                    legend_handles.append(scatter)
                else:
                    try:
                        shots_new = np.linspace(shots.min(), shots.max(), 300)
                        spl = make_interp_spline(shots, auc_roc, k=3, bc_type='natural')
                        auc_roc_smooth = spl(shots_new)
                        line, = ax.plot(shots_new, auc_roc_smooth, linewidth=3.5, alpha=0.7, color=color)
                        legend_handles.append(line)
                    except np.linalg.LinAlgError:
                        print(f"Warning: Singular matrix encountered for attack '{attack}'. Skipping interpolation.")
                        line, = ax.plot(shots, auc_roc, linewidth=3.5, alpha=0.7, color=color)
                        legend_handles.append(line)
            else:
                if 0 in shots:
                    zero_shot_index = np.where(shots == 0)[0][0]
                    scatter = ax.scatter(0, auc_roc[zero_shot_index], color=color)
                    legend_handles.append(scatter)
                else:
                    scatter = ax.scatter(shots, auc_roc, color=color)
                    legend_handles.append(scatter)

            if show_values and attack == "ReCaLL":
                for x, y in zip(shots, auc_roc):
                    ax.text(x, y, f"{y:.1f}", fontsize=12, ha='right')
            max_auc_roc = max(auc_roc)
            max_shot = shots[auc_roc.argmax()]
            legend_labels.append((attack, max_auc_roc))
            ax.set_ylim([55, 95])
            ax.locator_params(axis='y', nbins=5)
            
        legend_labels, legend_handles = zip(*sorted(zip(legend_labels, legend_handles), key=lambda x: x[0][1], reverse=True))
        legend_labels = [f"{label[0]}: {label[1]:.1f}" for label in legend_labels]

        ax.set_xlabel('Number of Shots')
        ax.set_ylabel('AUC-ROC (%)')
        ax.set_title(sub_category)
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.legend(legend_handles, legend_labels, loc='lower center', fontsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{sub_category}.png'), dpi=300)
        plt.close()
        
        
def analyze_final_results(folder_path, show_values=False):
    process_json_files(folder_path, show_values)