from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

from .config import Config


class IMDBDataset(Dataset):
    """
    Wraps IMDB text + labels and applies tokenization.
    """

    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def load_imdb_splits(cfg: Config):
    """
    Load IMDB from Hugging Face datasets and split the original train set
    into train/validation using a fixed random seed.
    """
    dataset = load_dataset("imdb")

    train_full = dataset["train"]
    test_dataset = dataset["test"]

    num_train = len(train_full)
    val_size = int(num_train * 0.2)  # 20% of 25k = 5k validation

    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(num_train)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_split = train_full.select(train_indices.tolist())
    val_split = train_full.select(val_indices.tolist())

    return train_split, val_split, test_dataset


def create_dataloaders(cfg: Config):
    """
    Creates PyTorch DataLoaders for train/val/test using IMDBDataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_split, val_split, test_split = load_imdb_splits(cfg)

    train_dataset = IMDBDataset(
        texts=train_split["text"],
        labels=train_split["label"],
        tokenizer=tokenizer,
        max_len=cfg.max_len,
    )

    val_dataset = IMDBDataset(
        texts=val_split["text"],
        labels=val_split["label"],
        tokenizer=tokenizer,
        max_len=cfg.max_len,
    )

    test_dataset = IMDBDataset(
        texts=test_split["text"],
        labels=test_split["label"],
        tokenizer=tokenizer,
        max_len=cfg.max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
