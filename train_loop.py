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
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    orthogonal_weight: float = 0.0,
    orthogonal_warmup_steps: int = 0,
    grad_clip_norm: float = 0.0,
) -> Tuple[float, int]:
    """Train LoRA parameters locally with optional orthogonality regularization on LoRA A."""

    def _as_float(value, name: str) -> float:
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be numeric, got {value!r}") from exc

    model.to(device)
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"[Train] trainable params: {len(trainable_params)}")

    lr = _as_float(lr, "lr")
    momentum = _as_float(momentum, "momentum")
    weight_decay = _as_float(weight_decay, "weight_decay")

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=max(1, total_steps),
    )
    total = len(dataloader.dataset)
    global_step = 0
    total_loss = 0.0

    for _ in range(epochs):
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            if orthogonal_weight > 0:
                warmup_steps = max(1, orthogonal_warmup_steps) if orthogonal_warmup_steps > 0 else 1
                ortho_scale = min(1.0, (global_step + 1) / warmup_steps) if orthogonal_warmup_steps > 0 else 1.0
                ortho_terms: List[torch.Tensor] = []
                for name, param in model.named_parameters():
                    if not param.requires_grad or "lora_A" not in name or param.dim() < 2:
                        continue
                    gram = param @ param.transpose(-1, -2)
                    ident = torch.eye(gram.size(0), device=device, dtype=param.dtype)
                    ortho_terms.append(torch.sum((gram - ident) ** 2))
                if ortho_terms:
                    ortho_loss = torch.stack(ortho_terms).mean()
                    loss = loss + orthogonal_weight * ortho_scale * ortho_loss

            loss.backward()
            if grad_clip_norm and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
            optimizer.step()
            scheduler.step()
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
