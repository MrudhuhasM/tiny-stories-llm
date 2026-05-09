from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelConfig:
    vocab_size: int = 10000
    block_size: int = 512
    d_model: int = 512
    d_ffn: int = 2048
    n_layers: int = 6
    n_heads: int = 8
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    bias: bool = True
    pre_norm: bool = False


@dataclass
class TrainConfig:
    num_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 1e-4
    train_size: int = 200_000
    test_size: int = 10_000
    max_len: int = 512
    stride: int = 128
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0
    log_interval: int = 100
    num_workers: int = 4
    seed: int = 42
