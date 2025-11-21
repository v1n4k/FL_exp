"""Model construction utilities."""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification

from config import ExperimentCfg
from lora_utils import add_lora_to_model, mark_only_lora_trainable, ordered_param_names


def build_model(cfg: ExperimentCfg, num_labels: int) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor], list[str]]:
    """Create a RoBERTa classifier with attached LoRA adapters."""

    base = AutoModelForSequenceClassification.from_pretrained(cfg.lora.base_model, num_labels=num_labels)
    model = add_lora_to_model(base, cfg.lora)
    mark_only_lora_trainable(model)
    state = model.state_dict()
    names = ordered_param_names(state)
    return model, state, names
