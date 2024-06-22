# ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods 🔍

[![Website](https://img.shields.io/badge/Website-Project%20Page-yellow)](https://royxie.com/recall-project-page)
[![arXiv](https://img.shields.io/badge/arXiv-2404.02936-b31b1b.svg)](https://arxiv.org/abs/2404.02936)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


## 📝 Overview
This is the official repository for ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods. The repo contains the original ReCaLL implementation on the WikiMIA benchmark dataset. Check out our [website](https://royxie.com/recall-project-page/) and [paper](https://arxiv.org/abs/2404.02936) for more information.

## 🛠 Installation
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run ReCaLL with the following command:

```bash
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
| `--dataset` | Dataset to use (default "wikimia")                                       |
| `--sub_dataset` | Subset of the dataset (e.g., "128" from wikimia dataset)                 |
| `--num_shots` | Number of shots for few-shot learning                                    |
| `--pass_window` | (Optional) exceed the context window                                     |
| `--synthetic_prefix` | (Optional) Use synthetic prefixes generated by GPT-4o                    |
| `--api_key_path` | (Optional) Path to OpenAI API key file (required for synthetic prefixes) |

## 📊 Example Output

The script will output results in JSON format and generates visualizations for:

- ReCaLL score
- Log-likelihood (LL)
- Reference model comparison
- Zlib compression ratio
- Min-k% probability
- Min-k++ probability

Example visualization:

<p align="center">
  <img src="/out/example.png" width="80%" height="80%">
</p>

## 🌟 Citation

If you use ReCaLL in your research, please cite our paper:

```bibtex
@article{xie2024recall,
    title={ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods},
    author={Xie, Roy and Wang, Junlin and Huang, Ruomin and Zhang, Minxing and Ge, Rong and Pei, Jian and Gong, Neil Zhenqiang and Dhingra, Bhuwan},
    journal={arXiv preprint arXiv:1234},
    year={2024}
}
```

## 📬 Contact

For questions or issues, please open an [issue](https://github.com/ruoyuxie/recall/issues) on GitHub or [contact](https://royxie.com/) the authors directly.

