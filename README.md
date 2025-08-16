# BERT LoRA & Adapter Fine-Tuning

This repository contains the implementation of our experiments on parameter-efficient fine-tuning (PEFT) methods for adapting BERT to downstream task.
[Read the paper](paper/bert_peft_paper.pdf)

## Research Questions

In this project, we investigated:

1. Which PEFT method performs best (LoRA, Houlsby adapters, or Adapter+) in terms of **accuracy** and parameter efficiency on a binary classification task (CoLA dataset)?
2. Do all transformer layers need adapters? Specifically, what happens if we remove adapters from lower layers?

## Key Takeaways

* Adapter+ is the most effective PEFT method for this binary classification task.
* LoRA is stable but underperforms compared to adapters, even with similar parameter counts.
* Layer ablation suggests that most gains come from higher transformer layers.

# Running Instructions

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Hugging Face Authentication:
   - Get your Hugging Face token from: https://huggingface.co/settings/tokens
   - In `src/train.py`, replace `login()` with `login("your_token_here")`
   - Or use my token (contact me for the token)

## 1. Train Models

### For Figure 4:
```bash
python train.py --figure figure4
```

### For Figure 6:
```bash
python train.py --figure figure6
```

Output: Results saved to `results_figure4/` or `results_figure6/` directories with JSON files containing validation accuracy, parameters, and configuration details.

## 2. Generate Plots

### For Figure 4:
```bash
python figure4.py
```

### For Figure 6:
```bash
python figure6.py
```

Output: Displays plots showing parameter efficiency comparison and layer ablation study results.



