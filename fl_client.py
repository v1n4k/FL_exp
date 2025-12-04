"""Flower client implementing FedSA-Fold."""
from __future__ import annotations

from typing import Any, Dict, List
import base64
import json
from pathlib import Path

import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader

from config import ExperimentCfg
from lora_utils import lora_a_to_b_name, split_lora_params, get_local_param_names
from train_loop import clone_state_dict, train_one_round, evaluate


class FedSAFoldClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        dataloader: DataLoader,
        eval_loader: DataLoader | None,
        model_builder,
        cfg: ExperimentCfg,
        param_names: List[str],
        device_override: str | None = None,
    ):
        self.cid = cid
        self.dataloader = dataloader
        self.eval_loader = eval_loader or dataloader
        self.cfg = cfg
        chosen_device = device_override or cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(chosen_device)
        print(f"[Client {cid}] using device {self.device}")

        model, _, _ = model_builder()
        self.model = model.to(self.device)

        # ordering
        self.param_names = param_names
        self.lora_a_names = [n for n in param_names if "lora_A" in n]

        # Auto-detect local parameters (LoRA B + task head) - works across all models
        local_param_names = get_local_param_names(self.model, param_names)
        self.lora_b_names = [n for n in local_param_names if "lora_B" in n]
        self.task_head_names = [n for n in local_param_names if "lora_B" not in n]

        print(f"[Client {cid}] Detected {len(self.task_head_names)} task head params: {self.task_head_names}")

        # initialize B cache
        _, b_state = split_lora_params(self.model.state_dict())
        self.B_state = {k: v.clone().to(self.device) for k, v in b_state.items()}

        # Initialize task head cache (task-agnostic: classifier, lm_head, score, head, etc.)
        task_head_state = {
            n: self.model.state_dict()[n].clone().to(self.device)
            for n in self.task_head_names
        }
        self.task_head_state = task_head_state

        # Optional B persistence across rounds/actors
        cache_dir = Path(self.cfg.train.client_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f"client_{self.cid}_B.pt"
        if self.cache_path.exists():
            try:
                cached = torch.load(self.cache_path, map_location=self.device)
                if isinstance(cached, dict):
                    for name, tensor in cached.items():
                        if name in self.B_state and tensor.shape == self.B_state[name].shape:
                            self.B_state[name] = tensor.to(self.device)
                print(f"[Client {cid}] restored B from cache")
            except Exception as exc:
                print(f"[Client {cid}] failed to load B cache: {exc}")

        # Restore task head from disk cache (model-agnostic: works for any task head)
        self.task_head_cache_path = cache_dir / f"client_{self.cid}_task_head.pt"
        if self.task_head_cache_path.exists():
            try:
                cached_head = torch.load(self.task_head_cache_path, map_location=self.device)
                if isinstance(cached_head, dict):
                    for name, tensor in cached_head.items():
                        if name in self.task_head_state and tensor.shape == self.task_head_state[name].shape:
                            self.task_head_state[name] = tensor.to(self.device)
                print(f"[Client {cid}] restored task head ({len(cached_head)} params) from cache")
            except Exception as exc:
                print(f"[Client {cid}] failed to load task head cache: {exc}")

        # Optional small randomization of LoRA params to encourage diversity
        if self.cfg.train.init_noise_std > 0:
            torch.manual_seed(int(cid) + self.cfg.seed)
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if "lora_" in name:
                        param.add_(torch.randn_like(param) * self.cfg.train.init_noise_std)

    # Flower interface -----------------------------------------------------
    def get_parameters(self, config: Dict[str, Any] | None = None) -> List[np.ndarray]:
        state = self.model.state_dict()
        return [state[name].detach().cpu().numpy() for name in self.param_names]

    def set_parameters(self, parameters: List[np.ndarray]):
        state = self.model.state_dict()
        # keep local B
        for name, tensor in self.B_state.items():
            if name in state:
                state[name] = tensor.to(self.device)

        # keep local task head (restore from cache before applying server params)
        for name, tensor in self.task_head_state.items():
            if name in state:
                state[name] = tensor.to(self.device)

        for name, array in zip(self.param_names, parameters):
            if "lora_B" in name or name in self.task_head_names:  # Skip local params
                continue
            if name in state:
                state[name] = torch.tensor(array, device=self.device)
        self.model.load_state_dict(state, strict=False)

    def _apply_T_to_B(self, config: Dict[str, Any]):
        blob = config.get("T_blob")
        if not blob:
            return
        try:
            payload = json.loads(blob)
        except Exception:
            return
        state = self.model.state_dict()
        r = self.cfg.lora.r
        for name_B, b64 in payload.items():
            if name_B not in state:
                continue
            try:
                raw = base64.b64decode(b64)
                T_np = np.frombuffer(raw, dtype=np.float32).reshape(r, r)
            except Exception:
                continue
            B = state[name_B].to(self.device)
            T = torch.tensor(T_np, device=self.device, dtype=B.dtype)
            I = torch.eye(T.shape[0], device=self.device, dtype=B.dtype)
            state[name_B] = B @ (I + T)
        self.model.load_state_dict(state, strict=False)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> tuple[list[np.ndarray], int, dict]:
        try:
            # apply global A/base, keep local B
            self.set_parameters(parameters)

            # fold residuals into B if provided
            self._apply_T_to_B(config)

            # snapshot of A before training
            state_before = clone_state_dict(self.model)
            a_before, _ = split_lora_params(state_before)

            loss, num_examples = train_one_round(
                self.model,
                self.dataloader,
                self.device,
                lr=self.cfg.train.lr,
                epochs=self.cfg.train.local_epochs,
                verbose=True,
                optimizer_name=self.cfg.train.optimizer,
                momentum=self.cfg.train.momentum,
                weight_decay=self.cfg.train.weight_decay,
                orthogonal_weight=self.cfg.train.orthogonal_reg_weight,
                orthogonal_warmup_steps=self.cfg.train.orthogonal_reg_warmup_steps,
                grad_clip_norm=self.cfg.train.grad_clip_norm,
            )

            state_after = self.model.state_dict()
            a_after, b_after = split_lora_params(state_after)
            self.B_state = {k: v.detach().clone() for k, v in b_after.items()}
            try:
                torch.save(self.B_state, self.cache_path)
            except Exception as exc:
                print(f"[Client {self.cid}] failed to save B cache: {exc}")

            # Save updated task head to disk (persistent across client recreations)
            self.task_head_state = {
                n: state_after[n].detach().clone() for n in self.task_head_names
            }
            try:
                torch.save(self.task_head_state, self.task_head_cache_path)
            except Exception as exc:
                print(f"[Client {self.cid}] failed to save task head cache: {exc}")

            # compute deltas for LoRA A only (task head kept local)
            delta_a_payload = [
                (a_after[name] - a_before[name]).detach().cpu().numpy() for name in self.lora_a_names
            ]

            # diagnostics
            delta_a_norm = sum(
                float(torch.sum((a_after[n] - a_before[n]) ** 2).sqrt().cpu()) for n in self.lora_a_names if n in a_after
            )
            # Compute task head delta norm (for monitoring only - not sent to server)
            delta_head_norm = sum(
                float(torch.sum((state_after[n] - state_before[n]) ** 2).sqrt().cpu())
                for n in self.task_head_names if n in state_after
            )
            print(
                f"[Client {self.cid}] samples={num_examples}, loss={loss:.4f}, "
                f"deltaA_norm={delta_a_norm:.4e}, deltaHEAD_norm={delta_head_norm:.4e} [LOCAL]"
            )

            metrics = {"loss": float(loss)}
            return delta_a_payload, num_examples, metrics  # ONLY LoRA A deltas
        except Exception as exc:  # surface client-side errors to logs
            import traceback

            print(f"[Client {self.cid}] fit failed: {exc}")
            traceback.print_exc()
            raise

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        # apply global params and T, then evaluate personalized model
        self.set_parameters(parameters)
        self._apply_T_to_B(config)
        metrics = evaluate(self.model, self.eval_loader, self.device)
        return metrics["loss"], len(self.eval_loader.dataset), {"accuracy": metrics["accuracy"]}
