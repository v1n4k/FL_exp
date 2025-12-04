"""Entry point wiring dataset, strategy, and simulation."""
from __future__ import annotations

import argparse
import random
from typing import Dict
import json
from pathlib import Path

import flwr as fl
import numpy as np
import torch
import os

from config import ExperimentCfg
from data import prepare_dataloaders
from fl_client import FedSAFoldClient
from fl_strategy import create_strategy
from model import build_model
from wandb_logger import WandbLogger


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> ExperimentCfg:
    parser = argparse.ArgumentParser(description="FedSA simulation with method selection")

    # Config file support
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")
    parser.add_argument("--method", type=str, default=None,
                        help="Method: fedsa_fold or fedsa_lora (overrides config)")

    # Make all existing arguments optional (default=None for CLI override)
    parser.add_argument("--num-clients", type=int, default=None)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--local-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Dirichlet concentration")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--init-noise-std", type=float, default=None)
    parser.add_argument("--gpus-per-client", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--orthogonal-reg-weight", type=float, default=None)
    parser.add_argument("--orthogonal-warmup-steps", type=int, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--client-cache-dir", type=str, default=None)

    args = parser.parse_args()

    # Load from config file if provided, otherwise use defaults
    if args.config:
        cfg = ExperimentCfg.from_yaml(args.config)
        print(f"[Config] Loaded from {args.config}")
    else:
        cfg = ExperimentCfg()
        print("[Config] Using defaults")

    # Override with CLI arguments (CLI takes precedence)
    if args.method is not None:
        cfg.train.method = args.method
    if args.seed is not None:
        cfg.seed = args.seed
    if args.num_clients is not None:
        cfg.train.num_clients = args.num_clients
    if args.rounds is not None:
        cfg.train.num_rounds = args.rounds
    if args.local_epochs is not None:
        cfg.train.local_epochs = args.local_epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.init_noise_std is not None:
        cfg.train.init_noise_std = args.init_noise_std
    if args.gpus_per_client is not None:
        cfg.train.gpus_per_client = args.gpus_per_client
    if args.optimizer is not None:
        cfg.train.optimizer = args.optimizer
    if args.momentum is not None:
        cfg.train.momentum = args.momentum
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.early_stop_patience is not None:
        cfg.train.early_stop_patience = args.early_stop_patience
    if args.orthogonal_reg_weight is not None:
        cfg.train.orthogonal_reg_weight = args.orthogonal_reg_weight
    if args.orthogonal_warmup_steps is not None:
        cfg.train.orthogonal_reg_warmup_steps = args.orthogonal_warmup_steps
    if args.grad_clip_norm is not None:
        cfg.train.grad_clip_norm = args.grad_clip_norm
    if args.client_cache_dir is not None:
        cfg.train.client_cache_dir = args.client_cache_dir
    if args.alpha is not None:
        cfg.data.dirichlet_alpha = args.alpha
    if args.max_length is not None:
        cfg.data.max_length = args.max_length

    # Special handling for wandb args
    if args.use_wandb:
        cfg.extra["use_wandb"] = "True"
    if args.wandb_project is not None:
        cfg.extra["wandb_project"] = args.wandb_project

    print(f"[Config] Method: {cfg.train.method}")
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    client_train_loaders, client_val_loaders, val_loader, holdout_loader, test_loader, tokenizer, num_labels = prepare_dataloaders(cfg)
    gpu_count = torch.cuda.device_count()
    # Diagnostics: print client dataset sizes
    for cid, loader in client_train_loaders.items():
        print(f"[Data] client {cid} train size: {len(loader.dataset)}; val size: {len(client_val_loaders[cid].dataset)}")

    # Build a template model for parameter ordering/state
    _, template_state, param_names = build_model(cfg, num_labels)

    def model_builder():
        model, _, _ = build_model(cfg, num_labels)
        return model, None, None

    use_wandb = cfg.extra.get("use_wandb", "False") == "True"
    wandb_run = None
    wandb_logger = None

    if use_wandb:
        try:
            import wandb

            # Log full config (not just subset)
            wandb_run = wandb.init(
                project=cfg.extra.get("wandb_project", "fedsa-exp"),
                group=cfg.extra.get("wandb_group", None),
                job_type=cfg.extra.get("wandb_job_type", cfg.train.method),
                config=cfg.to_dict(),  # Log ENTIRE config
                tags=[cfg.train.method, f"clients_{cfg.train.num_clients}"],
            )
            wandb_logger = WandbLogger(wandb_run, method=cfg.train.method)
            print("[Wandb] Initialized with full config logging")
        except Exception as exc:
            print(f"[Wandb] Init failed, continuing without logging: {exc}")
            wandb_run = None
            wandb_logger = None

    metrics_path = Path("fedsa_fold_metrics.jsonl")

    def log_fn(record: Dict):
        print(record)
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        # Use enhanced wandb logger if available
        if wandb_logger:
            wandb_logger.log(record)
        elif wandb_run:  # fallback to direct logging
            try:
                wandb_run.log(record)
            except Exception as exc:
                print(f"[Wandb] Log failed: {exc}")

    strategy = create_strategy(
        method=cfg.train.method,
        cfg=cfg,
        param_names=param_names,
        template_state=template_state,
        num_clients=cfg.train.num_clients,
        eval_loader=val_loader,
        model_builder=model_builder,
        log_fn=log_fn,
        wandb_logger=wandb_logger,
    )
    print(f"[Strategy] Using {cfg.train.method} strategy")

    def client_fn(cid: str):
        cid_int = int(cid)
        loader = client_train_loaders[cid_int]
        eval_loader = client_val_loaders[cid_int]
        # Let Ray/Flower assign GPUs via client_resources; avoid manual overrides.
        return FedSAFoldClient(cid, loader, eval_loader, model_builder, cfg, param_names, device_override=None)

    server_config = fl.server.ServerConfig(num_rounds=cfg.train.num_rounds)
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if torch.cuda.is_available() and cfg.train.gpus_per_client > 0:
        client_resources["num_gpus"] = cfg.train.gpus_per_client
    ray_init_args = {"include_dashboard": False, "log_to_driver": True, "logging_level": "DEBUG"}

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.train.num_clients,
        config=server_config,
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args=ray_init_args,
    )

    # Final evaluation on held-out split using the server's latest global state
    final_metrics = strategy.evaluate_global(holdout_loader, model_builder)
    print("Final holdout metrics:", final_metrics)
    log_fn({"stage": "holdout_final", **final_metrics})

    # Early stopping check based on personalized metric
    if strategy.no_improve_personalized >= cfg.train.early_stop_patience:
        print(
            f"Early stopping triggered after {strategy.no_improve_personalized} stale rounds "
            f"(best personalized loss {strategy.best_personalized_loss})"
        )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
