"""Flower client implementing FedSA-Fold."""
from __future__ import annotations

from typing import Any, Dict, List
import base64
import json

import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader

from config import ExperimentCfg
from lora_utils import lora_a_to_b_name, split_lora_params
from train_loop import clone_state_dict, train_one_round


class FedSAFoldClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        dataloader: DataLoader,
        model_builder,
        cfg: ExperimentCfg,
        param_names: List[str],
        device_override: str | None = None,
    ):
        self.cid = cid
        self.dataloader = dataloader
        self.cfg = cfg
        chosen_device = device_override or cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(chosen_device)
        print(f"[Client {cid}] using device {self.device}")

        model, _, _ = model_builder()
        self.model = model.to(self.device)

        # ordering
        self.param_names = param_names
        self.lora_a_names = [n for n in param_names if "lora_A" in n]
        self.classifier_names = [n for n in param_names if n.startswith("classifier")]

        # initialize B cache
        _, b_state = split_lora_params(self.model.state_dict())
        self.B_state = {k: v.clone().to(self.device) for k, v in b_state.items()}

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

        for name, array in zip(self.param_names, parameters):
            if "lora_B" in name:
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
                self.model, self.dataloader, self.device, lr=self.cfg.train.lr, epochs=self.cfg.train.local_epochs
            )

            state_after = self.model.state_dict()
            a_after, b_after = split_lora_params(state_after)
            self.B_state = {k: v.detach().clone() for k, v in b_after.items()}

            # compute deltas for LoRA A and classifier head
            delta_a_payload = [
                (a_after[name] - a_before[name]).detach().cpu().numpy() for name in self.lora_a_names
            ]
            delta_classifier = [
                (state_after[name] - state_before[name]).detach().cpu().numpy() for name in self.classifier_names
            ]
            metrics = {"loss": float(loss)}
            return delta_a_payload + delta_classifier, num_examples, metrics
        except Exception as exc:  # surface client-side errors to logs
            import traceback

            print(f"[Client {self.cid}] fit failed: {exc}")
            traceback.print_exc()
            raise

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Any]):
        # For brevity, skip client-side evaluation in this prototype
        return 0.0, len(self.dataloader.dataset), {"accuracy": 0.0}
