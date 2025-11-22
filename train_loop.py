"""Local training and evaluation utilities."""
from __future__ import annotations

from typing import Dict, Tuple, List

import torch
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup


def clone_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_one_round(
    model,
    dataloader: DataLoader,
    device,
    lr: float,
    epochs: int,
    verbose: bool = False,
) -> Tuple[float, int]:
    """Train LoRA parameters locally."""

    model.to(device)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[Train] trainable params: {len(trainable_params)} -> {trainable_names}")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    total = len(dataloader.dataset)
    global_step = 0
    total_loss = 0.0

    for _ in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            global_step += 1

    avg_loss = total_loss / max(global_step, 1)
    return avg_loss, total


def evaluate(model, dataloader: DataLoader, device) -> Dict[str, float]:
    """Compute accuracy and average loss on a dataloader."""

    model.to(device)
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss_sum += outputs.loss.item()
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].numel()

    avg_loss = loss_sum / max(len(dataloader), 1)
    acc = correct / max(total, 1)
    return {"loss": avg_loss, "accuracy": acc}
