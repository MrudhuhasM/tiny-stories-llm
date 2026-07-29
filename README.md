# TinyStories LLM

A decoder-only transformer trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The repository is structured as an experiment lab for comparing architectural choices while keeping the data pipeline, training loop, prompts, and evaluation procedure fixed.

## What is implemented

- Byte-level BPE tokenizer trained from the selected TinyStories training split
- GPT-style causal transformer with configurable depth, width, heads, dropout, and normalization order
- PyTorch scaled dot-product attention
- Pre-LayerNorm and Post-LayerNorm experiment baselines
- Mixed-precision training, `torch.compile`, AdamW, gradient clipping, warmup, and cosine decay
- Validation loss and perplexity tracking
- Fixed qualitative generation prompts after each epoch
- Optional Weights & Biases logging
- Best-checkpoint export and optional Hugging Face Hub upload

## Baseline configuration

| Setting | Value |
| --- | ---: |
| Layers | 6 |
| Attention heads | 8 |
| Model width | 512 |
| Feed-forward width | 2048 |
| Context length | 512 |
| Training stories | 200,000 |
| Validation stories | 10,000 |
| Epochs | 3 |

Two checked-in experiments isolate the normalization change:

- `exp01_post_norm.yaml`: Post-LayerNorm baseline
- `exp02_pre_norm.yaml`: same model and training settings with Pre-LayerNorm

## Setup

Requirements:

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/)
- PyTorch 2.x; a CUDA-capable GPU is strongly recommended

```bash
uv sync
uv pip install torch
```

Choose the PyTorch build appropriate for your operating system and CUDA version when GPU acceleration is required.

## Run an experiment

Run the baseline without external experiment logging:

```bash
uv run python train.py \
  --exp experiments/exp01_post_norm.yaml \
  --no-wandb
```

Run with Weights & Biases:

```bash
wandb login
uv run python train.py --exp experiments/exp02_pre_norm.yaml
```

On the first run, the script downloads TinyStories and builds the tokenizer. Checkpoints are written to:

```text
checkpoints/<experiment-name>/
```

If `hub_repo_id` is set in an experiment file, each improved checkpoint is also uploaded to that Hugging Face repository. Remove or change that field before running under a different account.

## Experiment design

Each experiment is a YAML file containing separate `model` and `train` sections. To make comparisons interpretable, copy the nearest experiment and change only the variable being tested.

```bash
uv run python train.py --exp experiments/expXX_name.yaml
```

The training script uses the same four generation prompts across runs, making qualitative changes easier to compare alongside validation loss and perplexity. See [`experiments/README.md`](experiments/README.md) for the experiment log and planned comparisons.

## Repository layout

```text
.
├── train.py
├── experiments/
│   ├── exp01_post_norm.yaml
│   ├── exp02_pre_norm.yaml
│   └── README.md
├── src/story_llm/
│   ├── config.py
│   ├── data.py
│   └── model.py
├── tokenizer/
│   └── tiny_stories_tokenizer.json
└── notebooks/
    └── 01-test.ipynb
```

## Current scope

This repository focuses on controlled training experiments rather than serving or optimized inference. Reproducible benchmark results should be added to the experiment log as runs are completed.
