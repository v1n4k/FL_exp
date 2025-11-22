"""Server-side strategy implementing FedSA-Fold with RPCA."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable, Optional
import base64
import json

import numpy as np
import torch
import flwr as fl
from flwr.common import FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from config import ExperimentCfg
from lora_utils import lora_a_to_b_name, split_lora_params
from rpca_utils import robust_pca
from train_loop import evaluate as eval_fn


def state_dict_to_ndarrays(state_dict: Dict[str, torch.Tensor], param_names: List[str]) -> List[np.ndarray]:
    return [state_dict[name].cpu().numpy() for name in param_names]


def ndarrays_to_state_dict(
    param_names: List[str], arrays: List[np.ndarray], template_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    state = {k: v.clone() for k, v in template_state.items()}
    for name, arr in zip(param_names, arrays):
        state[name] = torch.tensor(arr)
    return state


class FedSAFoldStrategy(fl.server.strategy.Strategy):
    """Minimal Flower Strategy to aggregate A-only updates with RPCA."""

    def __init__(
        self,
        cfg: ExperimentCfg,
        param_names: List[str],
        template_state: Dict[str, torch.Tensor],
        num_clients: int,
        eval_loader,
        model_builder,
        log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.cfg = cfg
        self.param_names = param_names
        self.global_state = {k: v.clone() for k, v in template_state.items()}
        self.num_clients = num_clients
        self.eval_loader = eval_loader
        self.model_builder = model_builder
        self.client_T: Dict[str, Dict[str, np.ndarray]] = {}
        self.lora_a_names = [n for n in param_names if "lora_A" in n]
        self.classifier_names = [n for n in param_names if "classifier" in n]
        self.parameters = ndarrays_to_parameters(state_dict_to_ndarrays(self.global_state, self.param_names))
        self.logged_metrics: List[Dict[str, Any]] = []
        self.log_fn = log_fn
        self.best_val_loss: float | None = None
        self.no_improve_rounds: int = 0

    # Flower hooks ---------------------------------------------------------
    def initialize_parameters(self, client_manager) -> Parameters:
        return self.parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        fit_config = {}
        sample_size = self.num_clients
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
        instructions = []
        for client in clients:
            cid = getattr(client, "cid", "unknown")
            # Flower config must be scalar types; pack T into a single base64-encoded JSON string
            T_dict = self.client_T.get(cid, {})
            client_config = {}
            if T_dict:
                payload = {
                    name: base64.b64encode(v.astype(np.float32).tobytes()).decode("ascii")
                    for name, v in T_dict.items()
                }
                client_config["T_blob"] = json.dumps(payload)
            instructions.append(fl.common.FitIns(parameters=parameters, config=client_config))
        return list(zip(clients, instructions))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Any],
    ) -> Tuple[Parameters, Dict[str, Any]]:
        if failures:
            print(f"[Server] aggregate_fit saw {len(failures)} failures")
            for f in failures:
                print(f"[Server] failure detail: {f}")

        if not results:
            return self.parameters, {}

        # Collect delta A and classifier deltas per client
        client_delta_a: Dict[str, Dict[str, np.ndarray]] = {}
        client_delta_cls: Dict[str, Dict[str, np.ndarray]] = {}
        for client, fit_res in results:
            cid = getattr(client, "cid", "unknown")
            arrays = parameters_to_ndarrays(fit_res.parameters)
            split = len(self.lora_a_names)
            arr_a = arrays[:split]
            arr_cls = arrays[split:]
            client_delta_a[cid] = {name: arr for name, arr in zip(self.lora_a_names, arr_a)}
            client_delta_cls[cid] = {name: arr for name, arr in zip(self.classifier_names, arr_cls)}

        # Aggregate each LoRA A with RPCA
        for name in self.lora_a_names:
            A_global = self.global_state[name].cpu().numpy()
            client_list = list(client_delta_a.keys())
            stacked = np.stack(
                [client_delta_a[cid][name].reshape(-1) for cid in client_list],
                axis=1,
            )

            L, S = robust_pca(
                stacked,
                lam=self.cfg.rpca.lam,
                max_iter=self.cfg.rpca.max_iter,
                tol=self.cfg.rpca.tol,
            )

            # Update global A with mean of low-rank parts
            L_mean = L.mean(axis=1).reshape(self.global_state[name].shape)
            self.global_state[name] = torch.tensor(A_global + L_mean, dtype=self.global_state[name].dtype)

            # Build T matrices per client
            A_new_np = self.global_state[name].cpu().numpy()
            A_pinv = np.linalg.pinv(A_new_np)
            for col, cid in enumerate(client_list):
                S_i = S[:, col].reshape(self.global_state[name].shape)
                T_i = S_i @ A_pinv
                b_name = lora_a_to_b_name(name)
                self.client_T.setdefault(cid, {})[b_name] = T_i

        # Aggregate classifier head with simple mean of deltas
        if self.classifier_names:
            for name in self.classifier_names:
                deltas = [client_delta_cls[cid][name] for cid in client_delta_cls if name in client_delta_cls[cid]]
                if not deltas:
                    continue
                mean_delta = np.mean(np.stack(deltas, axis=0), axis=0)
                self.global_state[name] = torch.tensor(
                    self.global_state[name].cpu().numpy() + mean_delta, dtype=self.global_state[name].dtype
                )

        # Update stored parameters for the next round
        ndarrays = state_dict_to_ndarrays(self.global_state, self.param_names)
        self.parameters = ndarrays_to_parameters(ndarrays)
        metrics = {"server_round": server_round}
        return self.parameters, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvalIns]]:
        """
        Configure the next round of evaluation.
        Since we're doing server-side evaluation only, we return an empty list.
        """
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Aggregate evaluation results.
        Since we're doing server-side evaluation only, we return dummy values.
        """
        return 0.0, {}

    def evaluate(self, server_round: int, parameters: Parameters):
        metrics = self.evaluate_global(self.eval_loader, self.model_builder, parameters=parameters)
        metrics_with_round = {"round": server_round, **metrics}
        self.logged_metrics.append(metrics_with_round)
        if self.log_fn:
            self.log_fn({"stage": "val", **metrics_with_round})

        # early stopping on validation loss
        val_loss = metrics.get("loss")
        if val_loss is not None:
            if self.best_val_loss is None or val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.no_improve_rounds = 0
            else:
                self.no_improve_rounds += 1
        return 0.0, metrics
        return 0.0, metrics

    # Helper for external calls (main.py)
    def evaluate_global(self, dataloader, model_builder=None, parameters: Parameters | None = None) -> Dict[str, float]:
        arrays = parameters_to_ndarrays(parameters) if parameters is not None else state_dict_to_ndarrays(self.global_state, self.param_names)
        state = ndarrays_to_state_dict(self.param_names, arrays, self.global_state)
        builder = model_builder or self.model_builder
        model, _, _ = builder()
        model.load_state_dict(state, strict=False)
        device = torch.device(self.cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        metrics = eval_fn(model, dataloader, device)
        return metrics
