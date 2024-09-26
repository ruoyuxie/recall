# ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods 🔍

[![Website](https://img.shields.io/badge/Website-Project%20Page-yellow)](https://royxie.com/recall-project-page)
[![arXiv](https://img.shields.io/badge/arXiv-2404.02936-b31b1b.svg)](https://arxiv.org/abs/2406.15968)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## 📝 Overview
This is the official repository for [ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods (EMNLP 2024)](https://arxiv.org/abs/2406.15968). The repo contains the original ReCaLL implementation on the WikiMIA benchmark dataset. Check out the project [website](https://royxie.com/recall-project-page/) for more information.

⭐ If you find our implementation and paper helpful, please consider citing our work ⭐ :


```bibtex
@misc{xie2024recall,
    title={ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods},
    author={Xie, Roy and Wang, Junlin and Huang, Ruomin and Zhang, Minxing and Ge, Rong and Pei, Jian and Gong, Neil Zhenqiang and Dhingra, Bhuwan},
    year={2024},
    eprint={2406.15968},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```


## 🛠 Installation
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run ReCaLL with the following command:

```bash
cd src
python run.py --target_model <TARGET_MODEL> --ref_model <REFERENCE_MODEL> --output_dir <OUTPUT_PATH> --dataset <DATASET> --sub_dataset <SUB_DATASET> --num_shots <NUM_SHOTS>
```

Example:
```bash
python run.py --target_model "EleutherAI/pythia-6.9b" --ref_model "EleutherAI/pythia-70m" --output_dir ./output --dataset "wikimia" --sub_dataset "128" --num_shots 7
```

### 🔧 Parameters:

| Parameter | Description                                                              |
|-----------|--------------------------------------------------------------------------|
| `--target_model` | Target model to evaluate (e.g., "EleutherAI/pythia-6.9b")                |
| `--ref_model` | Reference model for comparison (e.g., "EleutherAI/pythia-70m")           |
| `--output_dir` | Directory to save output files                                           |
| `--dataset` | Dataset to use ("wikimia")                                       |
| `--sub_dataset` | Subset of the dataset (e.g., "128" from wikimia dataset)                 |
| `--num_shots` | Number of shots for prefix                                    |
| `--pass_window` | (Optional) exceed the context window                                     |
| `--synthetic_prefix` | (Optional) Use synthetic prefixes generated by GPT-4o                    |
| `--api_key_path` | (Optional) Path to OpenAI API key file (required for synthetic prefixes) |

## 📊 Example Output

The script will output results in JSON format and generates visualizations for:

- ReCaLL score
- Loss
- Reference  
- Zlib 
- Min-k% 
- Min-k++ 

Example visualization from 1 - 28 shots:

<p align="center">
  <img src="/out/all_28_shots_example.png" width="80%" height="80%">
</p>


## 📬 Contact

For questions or issues, please open an [issue](https://github.com/ruoyuxie/recall/issues) on GitHub or [contact](https://royxie.com/) the authors directly.

