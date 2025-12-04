"""LoRA helpers for separating and managing adapter parameters."""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
from peft import LoraConfig, get_peft_model

from config import LoraCfg

LoraAB = Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]


def add_lora_to_model(model, cfg: LoraCfg):
    """Attach LoRA adapters to a base transformers model."""

    lora_config = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=cfg.target_modules,
    )
    return get_peft_model(model, lora_config)


def split_lora_params(state_dict: Dict[str, torch.Tensor]) -> LoraAB:
    """Split state dict into LoRA A and B tensors."""

    a_params, b_params = {}, {}
    for name, tensor in state_dict.items():
        if "lora_A" in name:
            a_params[name] = tensor
        elif "lora_B" in name:
            b_params[name] = tensor
    return a_params, b_params


def merge_lora_params(
    base_state: Dict[str, torch.Tensor],
    a_params: Dict[str, torch.Tensor],
    b_params: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Merge LoRA A/B into a base model state dict."""

    merged = base_state.copy()
    merged.update(a_params)
    merged.update(b_params)
    return merged


def mark_only_lora_trainable(model):
    """Freeze the base model; keep LoRA adapters and classifier head trainable."""

    for name, param in model.named_parameters():
        # allow any parameter belonging to classifier head or LoRA adapters
        if "lora_" in name or "classifier" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def lora_a_to_b_name(name: str) -> str:
    """Heuristic to switch lora_A name to paired lora_B."""

    return name.replace("lora_A", "lora_B")


def ordered_param_names(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """Stable ordering to map between ndarrays and a state dict."""

    return list(state_dict.keys())


def tensors_to_device(tensors: Iterable[torch.Tensor], device) -> List[torch.Tensor]:
    return [t.to(device) for t in tensors]


def get_local_param_names(model, param_names: List[str]) -> List[str]:
    """
    Auto-detect parameters that should remain local (not aggregated).

    Uses hybrid approach:
    1. Always includes LoRA B matrices
    2. Auto-detects trainable non-LoRA parameters (task heads)
    3. Works across any model architecture (RoBERTa, GPT-2, T5, ViT, etc.)

    Args:
        model: PyTorch model with PEFT adapters
        param_names: List of all parameter names in order

    Returns:
        List of parameter names that should be kept local
    """
    local_names = []

    # 1. LoRA B always kept local
    local_names.extend([n for n in param_names if "lora_B" in n])

    # 2. Task-specific heads: auto-detect trainable non-LoRA parameters
    # This includes: classifier, score, lm_head, head, etc. (model-agnostic)
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" not in name:
            if name in param_names:
                local_names.append(name)

    return local_names
