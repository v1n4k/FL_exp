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
from fl_strategy import FedSAFoldStrategy
from model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> ExperimentCfg:
    parser = argparse.ArgumentParser(description="FedSA-Fold simulation")
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="fedsa-fold")
    parser.add_argument("--init-noise-std", type=float, default=0.0)
    args = parser.parse_args()

    cfg = ExperimentCfg()
    cfg.seed = args.seed
    cfg.train.num_clients = args.num_clients
    cfg.train.num_rounds = args.rounds
    cfg.train.local_epochs = args.local_epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.init_noise_std = args.init_noise_std
    cfg.data.dirichlet_alpha = args.alpha
    cfg.data.max_length = args.max_length
    cfg.extra["use_wandb"] = str(args.use_wandb)
    cfg.extra["wandb_project"] = args.wandb_project
    return cfg


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    client_loaders, val_loader, holdout_loader, test_loader, tokenizer, num_labels = prepare_dataloaders(cfg)
    gpu_count = torch.cuda.device_count()

    # Build a template model for parameter ordering/state
    _, template_state, param_names = build_model(cfg, num_labels)

    def model_builder():
        model, _, _ = build_model(cfg, num_labels)
        return model, None, None

    use_wandb = cfg.extra.get("use_wandb", "False") == "True"
    wandb_run = None
    if use_wandb:
        try:
            import wandb

            wandb_run = wandb.init(
                project=cfg.extra.get("wandb_project", "fedsa-fold"),
                config={
                    "num_clients": cfg.train.num_clients,
                    "rounds": cfg.train.num_rounds,
                    "local_epochs": cfg.train.local_epochs,
                    "batch_size": cfg.train.batch_size,
                    "lr": cfg.train.lr,
                    "alpha": cfg.data.dirichlet_alpha,
                },
            )
        except Exception as exc:
            print("wandb init failed, continuing without logging:", exc)
            wandb_run = None

    metrics_path = Path("fedsa_fold_metrics.jsonl")

    def log_fn(record: Dict):
        print(record)
        with metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if wandb_run:
            try:
                wandb_run.log(record)
            except Exception as exc:  # keep training going
                print("wandb log failed:", exc)

    strategy = FedSAFoldStrategy(
        cfg,
        param_names,
        template_state,
        num_clients=cfg.train.num_clients,
        eval_loader=val_loader,
        model_builder=model_builder,
        log_fn=log_fn,
    )

    def client_fn(cid: str):
        cid_int = int(cid)
        loader = client_loaders[cid_int]
        device_override = None
        if gpu_count > 0:
            device_override = f"cuda:{cid_int % gpu_count}"
        return FedSAFoldClient(cid, loader, model_builder, cfg, param_names, device_override=device_override)

    server_config = fl.server.ServerConfig(num_rounds=cfg.train.num_rounds)
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.train.num_clients,
        config=server_config,
        strategy=strategy,
    )

    # Final evaluation on held-out split using the server's latest global state
    final_metrics = strategy.evaluate_global(holdout_loader, model_builder)
    print("Final holdout metrics:", final_metrics)
    log_fn({"stage": "holdout_final", **final_metrics})

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
