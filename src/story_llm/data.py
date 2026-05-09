from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

TOKENIZER_FILENAME = "tiny_stories_tokenizer.json"


def build_tokenizer(
    texts,
    tokenizer_dir: Path,
    vocab_size: int = 10000,
    min_frequency: int = 2,
) -> Tokenizer:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<s>", "<pad>", "</s>"],
        min_frequency=min_frequency,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    tokenizer.decoder = ByteLevelDecoder()
    save_path = tokenizer_dir / TOKENIZER_FILENAME
    tokenizer.save(str(save_path))
    print(f"Tokenizer saved → {save_path}")
    return tokenizer


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_file(str(tokenizer_dir / TOKENIZER_FILENAME))


def tokenize_and_concat(texts, tokenizer: Tokenizer) -> Tensor:
    all_ids = []
    for text in texts:
        all_ids.extend(tokenizer.encode(text).ids)
    return torch.tensor(all_ids, dtype=torch.long)


class TinyStoriesDataset(Dataset):
    def __init__(self, texts, tokenizer: Tokenizer, max_len: int, stride: int):
        self.max_len = max_len
        self.stride = stride
        self.data = tokenize_and_concat(texts, tokenizer)

    def __len__(self) -> int:
        return (len(self.data) - self.max_len) // self.stride

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        start = idx * self.stride
        x = self.data[start : start + self.max_len]
        y = self.data[start + 1 : start + self.max_len + 1]
        return x, y


def get_dataloaders(
    data_dir: Path,
    tokenizer_dir: Path,
    train_size: int,
    test_size: int,
    max_len: int,
    stride: int,
    batch_size: int,
    seed: int = 42,
    num_workers: int = 4,
) -> Tuple[Tokenizer, DataLoader, DataLoader]:
    tokenizer_path = tokenizer_dir / TOKENIZER_FILENAME

    print("Loading TinyStories from HuggingFace cache / hub...")
    train_raw = load_dataset(
        "roneneldan/TinyStories", split="train", cache_dir=str(data_dir)
    )
    val_raw = load_dataset(
        "roneneldan/TinyStories", split="validation", cache_dir=str(data_dir)
    )

    train_raw = train_raw.shuffle(seed=seed).select(range(train_size))
    val_raw = val_raw.shuffle(seed=seed).select(range(test_size))

    if not tokenizer_path.exists():
        print("Building BPE tokenizer from training texts...")
        tokenizer = build_tokenizer(train_raw["text"], tokenizer_dir)
    else:
        print(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_dir)

    print("Tokenising train set...")
    train_dataset = TinyStoriesDataset(train_raw["text"], tokenizer, max_len, stride)
    print("Tokenising validation set...")
    val_dataset = TinyStoriesDataset(val_raw["text"], tokenizer, max_len, stride)

    print(
        f"Train  → {len(train_dataset.data):,} tokens | {len(train_dataset):,} samples\n"
        f"Val    → {len(val_dataset.data):,} tokens | {len(val_dataset):,} samples\n"
        f"Vocab  → {tokenizer.get_vocab_size():,}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return tokenizer, train_loader, val_loader
