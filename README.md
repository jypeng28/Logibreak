# LogiBreak

This repository contains an official implement of [**LogiBreak**](https://arxiv.org/abs/2505.13527) accepted by ACL'26 on language models across multiple languages. The framework consists of three main components: reformulation, jailbreak, and evaluation.

## Overview

The project implements a systematic approach to:
1. Reformulate potentially harmful requests into formal logical forms
2. Attempt jailbreaks using the reformulated requests
3. Evaluate the success of jailbreak attempts using multiple judges

## Components

### 1. Reformulation (`reformulate_*.py`)
- Reformulates potentially harmful requests into formal logical forms
- Available for multiple languages:
  - English (`reformulate_en.py`)
  - Chinese (`reformulate_zh.py`)
  - Dutch (`reformulate_du.py`)
- Uses GPT-3.5-turbo by default for reformulation
- Supports multiple restarts for each request

### 2. Jailbreak (`jailbreak_*.py`)
- Attempts to jailbreak target models using reformulated requests
- Available for multiple languages:
  - English (`jailbreak_en.py`)
  - Chinese (`jailbreak_zh.py`)
  - Dutch (`jailbreak_du.py`)
- Uses a formal semantics approach to generate jailbreak attempts
- Supports parallel processing with multiple restarts

### 3. Evaluation (`evaluate_*.py`)
- Evaluates jailbreak attempts using multiple judges:
  - Rule-based evaluation
  - GPT-4 evaluation
  - Llama3-70b evaluation
- Available for multiple languages:
  - English (`evaluate_en.py`)
  - Chinese (`evaluate_zh.py`)
  - Dutch (`evaluate_du.py`)
- Generates comprehensive evaluation results

## Usage


### Running the Pipeline

1. **Reformulation**:
```bash
python reformulate_en.py --reformulate_model gpt-3.5-turbo --n_restarts 5
```

2. **Jailbreak**:
```bash
python jailbreak_en.py --target_model gpt-3.5-turbo --input_path <path_to_reformulated_queries> --n_restarts 5
```

3. **Evaluation**:
```bash
python evaluate_en.py --evaluate_llama3 False --evaluate_gpt True --input_path <path_to_jailbreak_output> --n_restarts 5
```

### Output Files
- Reformulated queries are saved in `./output/reformulated_queries/`
- Jailbreak attempts are saved in `./output/jailbreak_output/`
- Evaluation results are saved alongside the input files with an `-evaluation_result.json` suffix

## Project Structure
```
.
├── api.py                 # API interface for language models
├── judges.py             # Evaluation judges implementation
├── reformulate_*.py      # Reformulation scripts for different languages
├── jailbreak_*.py     # Jailbreak scripts for different languages
├── evaluate_*.py         # Evaluation scripts for different languages
└── output/               # Output directory for results
    ├── reformulated_queries/
    └── jailbreak_output/
```


## Citation

If you feel our work is insightful and want to use the code or cite our paper, please add the following citation to your paper references.
```
@article{peng2025logic,
  title={Logic jailbreak: Efficiently unlocking llm safety restrictions through formal logical expression},
  author={Peng, Jingyu and Wang, Maolin and Wang, Nan and Li, Jiatong and Li, Yuchen and Ye, Yuyang and Wang, Wanyu and Jia, Pengyue and Zhang, Kai and Zhao, Xiangyu},
  journal={arXiv preprint arXiv:2505.13527},
  year={2025}
}
```