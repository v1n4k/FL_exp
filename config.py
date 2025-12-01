"""Configuration structures for the FedSA-Fold simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class LoraCfg:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "value"]
    )
    base_model: str = "roberta-large"


@dataclass
class RPCACfg:
    max_iter: int = 50
    tol: float = 1e-6
    lam: float | None = None


@dataclass
class DataCfg:
    dataset_name: str = "glue"
    glue_task: str = "sst2"
    max_length: int = 128
    dirichlet_alpha: float = 0.5


@dataclass
class TrainCfg:
    num_clients: int = 5
    num_rounds: int = 3
    local_epochs: int = 1
    batch_size: int = 8
    lr: float = 3e-4
    method: str = "fedsa_fold"  # extensible switch
    device: str | None = None
    init_noise_std: float = 0.0
    gpus_per_client: float = 0.0  # set >0 to allocate GPU via Flower/Ray
    optimizer: str = "sgd"  # "sgd" or "adamw"
    momentum: float = 0.9
    weight_decay: float = 0.0
    early_stop_patience: int = 3  # consecutive non-improve rounds on personalized metric
    orthogonal_reg_weight: float = 0.0  # weight for orthonormality penalty on LoRA A


@dataclass
class ExperimentCfg:
    seed: int = 42
    lora: LoraCfg = field(default_factory=LoraCfg)
    rpca: RPCACfg = field(default_factory=RPCACfg)
    data: DataCfg = field(default_factory=DataCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    extra: Dict[str, str] = field(default_factory=dict)
