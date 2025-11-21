"""Dataset loading and non-IID partitioning utilities."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from config import DataCfg, ExperimentCfg


def load_and_tokenize(data_cfg: DataCfg, base_model: str, seed: int) -> Tuple[DatasetDict, AutoTokenizer, int]:
    """Load GLUE split and tokenize with a RoBERTa tokenizer."""

    raw = load_dataset(data_cfg.dataset_name, data_cfg.glue_task)
    # create a held-out portion from train for final evaluation
    raw_train_split = raw["train"].train_test_split(test_size=0.1, seed=seed)
    raw = DatasetDict(
        {
            "train": raw_train_split["train"],
            "train_holdout": raw_train_split["test"],
            "validation": raw["validation"],
            "test": raw.get("test", raw["validation"]),  # GLUE test lacks labels; fallback to validation
        }
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    def _tokenize(batch):
        return tokenizer(
            batch["sentence"],
            padding="max_length",
            truncation=True,
            max_length=data_cfg.max_length,
        )

    tokenized = raw.map(_tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    num_labels = int(max(raw["train"]["label"])) + 1
    return tokenized, tokenizer, num_labels


def dirichlet_partition(dataset: Dataset, num_clients: int, alpha: float, num_labels: int) -> Dict[int, List[int]]:
    """Return mapping client_id -> list of example indices per Dirichlet sampling."""

    per_label_indices = {k: np.where(np.array(dataset["labels"]) == k)[0].tolist() for k in range(num_labels)}
    client_indices: Dict[int, List[int]] = {cid: [] for cid in range(num_clients)}

    for label, idxs in per_label_indices.items():
        if not idxs:
            continue
        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        split_sizes = (proportions * len(idxs)).astype(int)

        # adjust to preserve total count
        while split_sizes.sum() < len(idxs):
            split_sizes[np.argmin(split_sizes)] += 1
        while split_sizes.sum() > len(idxs):
            split_sizes[np.argmax(split_sizes)] -= 1

        # shuffle and split
        np.random.shuffle(idxs)
        offset = 0
        for cid, size in enumerate(split_sizes):
            if size == 0:
                continue
            client_indices[cid].extend(idxs[offset : offset + size])
            offset += size

    for cid in client_indices:
        np.random.shuffle(client_indices[cid])
    return client_indices


def build_client_loaders(
    tokenized_train: Dataset,
    client_map: Dict[int, List[int]],
    batch_size: int,
) -> Dict[int, DataLoader]:
    loaders: Dict[int, DataLoader] = {}
    for cid, indices in client_map.items():
        subset = Subset(tokenized_train, indices)
        loaders[cid] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loaders


def build_eval_loader(dataset: Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def prepare_dataloaders(
    cfg: ExperimentCfg,
) -> Tuple[Dict[int, DataLoader], DataLoader, DataLoader, DataLoader, AutoTokenizer, int]:
    """Load data, sample a Dirichlet partition, and build loaders."""

    tokenized, tokenizer, num_labels = load_and_tokenize(cfg.data, cfg.lora.base_model, seed=cfg.seed)
    client_map = dirichlet_partition(
        tokenized["train"], num_clients=cfg.train.num_clients, alpha=cfg.data.dirichlet_alpha, num_labels=num_labels
    )
    client_loaders = build_client_loaders(tokenized["train"], client_map, cfg.train.batch_size)
    val_loader = build_eval_loader(tokenized["validation"], cfg.train.batch_size)
    holdout_loader = build_eval_loader(tokenized["train_holdout"], cfg.train.batch_size)
    test_loader = build_eval_loader(tokenized["test"], cfg.train.batch_size)
    return client_loaders, val_loader, holdout_loader, test_loader, tokenizer, num_labels
