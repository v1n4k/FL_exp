"""Server-side strategy implementing FedSA-Fold with RPCA."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Callable, Optional
import base64
import json

import numpy as np
import torch
import flwr as fl
from flwr.common import EvaluateRes, EvaluateIns, FitRes, Parameters, ndarrays_to_parameters, parameters_to_ndarrays

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
        wandb_logger: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.param_names = param_names
        self.global_state = {k: v.clone() for k, v in template_state.items()}
        self.num_clients = num_clients
        self.eval_loader = eval_loader
        self.model_builder = model_builder
        self.client_T: Dict[str, Dict[str, np.ndarray]] = {}
        self.lora_a_names = [n for n in param_names if "lora_A" in n]
        # NOTE: No task head tracking on server - task heads remain personalized on clients
        self.parameters = ndarrays_to_parameters(state_dict_to_ndarrays(self.global_state, self.param_names))
        self.logged_metrics: List[Dict[str, Any]] = []
        self.log_fn = log_fn
        self.wandb_logger = wandb_logger
        self.best_val_loss: float | None = None
        self.no_improve_rounds: int = 0
        self.best_personalized_loss: float | None = None
        self.no_improve_personalized: int = 0
        self.last_global_metrics: Dict[str, Any] | None = None

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

        # Collect delta A only (task heads remain local on clients)
        client_delta_a: Dict[str, Dict[str, np.ndarray]] = {}
        client_weights: Dict[str, int] = {}
        for client, fit_res in results:
            cid = getattr(client, "cid", "unknown")
            arrays = parameters_to_ndarrays(fit_res.parameters)
            # IMPORTANT: Now only receiving LoRA A deltas (no task head)
            client_delta_a[cid] = {name: arr for name, arr in zip(self.lora_a_names, arrays)}
            client_weights[cid] = fit_res.num_examples

        # Aggregate each LoRA A with RPCA
        for name in self.lora_a_names:
            A_global = self.global_state[name].cpu().numpy()
            client_list = list(client_delta_a.keys())
            if not client_list:
                continue
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

            weights = np.array([client_weights[cid] for cid in client_list], dtype=np.float32)
            weight_sum = np.sum(weights) if np.sum(weights) > 0 else len(client_list)
            L_cols = [L[:, idx] for idx in range(L.shape[1])]
            L_mean = sum(w * col for w, col in zip(weights, L_cols)) / weight_sum
            L_mean = L_mean.reshape(self.global_state[name].shape)
            self.global_state[name] = torch.tensor(A_global + L_mean, dtype=self.global_state[name].dtype)

            if self.log_fn:
                l_norm = float(np.linalg.norm(L))
                s_norm = float(np.linalg.norm(S))
                lmean_norm = float(np.linalg.norm(L_mean))
                self.log_fn(
                    {
                        "stage": "rpca",
                        "round": server_round,
                        "name": name,
                        "L_norm": l_norm,
                        "S_norm": s_norm,
                        "L_mean_norm": lmean_norm,
                    }
                )

            # Enhanced wandb logging for RPCA
            if self.wandb_logger:
                self.wandb_logger.log_rpca_metrics(
                    round=server_round, name=name, L=L, S=S, L_mean=L_mean
                )

            # Orthogonality diagnostics on updated A
            if self.log_fn:
                A_new_np_diag = self.global_state[name].cpu().numpy()
                gram = A_new_np_diag @ A_new_np_diag.T
                ident = np.eye(gram.shape[0], dtype=gram.dtype)
                ortho_dev = float(np.linalg.norm(gram - ident, ord="fro"))
                try:
                    sv = np.linalg.svd(A_new_np_diag, compute_uv=False)
                    sv_min = float(np.min(sv)) if sv.size else 0.0
                    sv_max = float(np.max(sv)) if sv.size else 0.0
                except Exception:
                    sv_min = sv_max = 0.0
                self.log_fn(
                    {
                        "stage": "orthogonality",
                        "round": server_round,
                        "name": name,
                        "gram_dev_fro": ortho_dev,
                        "sv_min": sv_min,
                        "sv_max": sv_max,
                    }
                )

            # Enhanced wandb logging for orthogonality
            if self.wandb_logger:
                A_new_np_diag = self.global_state[name].cpu().numpy()
                self.wandb_logger.log_orthogonality_metrics(
                    round=server_round, name=name, A=A_new_np_diag
                )

            # Build T matrices per client
            A_new_np = self.global_state[name].cpu().numpy()
            A_T = A_new_np.T
            t_norms = []
            t_maxes = []
            t_ratios = []
            for col, cid in enumerate(client_list):
                S_i = S[:, col].reshape(self.global_state[name].shape)
                T_i = S_i @ A_T
                t_norm = float(np.linalg.norm(T_i))
                t_max = float(np.max(np.abs(T_i)))
                s_norm_i = float(np.linalg.norm(S_i)) + 1e-8
                t_norms.append(t_norm)
                t_maxes.append(t_max)
                t_ratios.append(t_norm / s_norm_i)
                b_name = lora_a_to_b_name(name)
                self.client_T.setdefault(cid, {})[b_name] = T_i

            if self.log_fn and t_norms:
                t_norms_np = np.array(t_norms, dtype=np.float32)
                t_maxes_np = np.array(t_maxes, dtype=np.float32)
                t_ratios_np = np.array(t_ratios, dtype=np.float32)
                self.log_fn(
                    {
                        "stage": "T_stats",
                        "round": server_round,
                        "name": name,
                        "T_norm_mean": float(np.mean(t_norms_np)),
                        "T_norm_p95": float(np.percentile(t_norms_np, 95)),
                        "T_norm_max": float(np.max(t_norms_np)),
                        "T_absmax_mean": float(np.mean(t_maxes_np)),
                        "T_absmax_p95": float(np.percentile(t_maxes_np, 95)),
                        "T_absmax_max": float(np.max(t_maxes_np)),
                        "T_over_S_mean": float(np.mean(t_ratios_np)),
                        "T_over_S_p95": float(np.percentile(t_ratios_np, 95)),
                        "T_over_S_max": float(np.max(t_ratios_np)),
                    }
                )

            # Enhanced wandb logging for T-matrix stats
            if self.wandb_logger and t_norms:
                self.wandb_logger.log_tmatrix_stats(
                    round=server_round,
                    name=name,
                    t_norms=t_norms,
                    t_maxes=t_maxes,
                    t_ratios=t_ratios
                )

        # Task head aggregation removed - task heads remain local on each client
        # Global state only tracks LoRA A parameters (aggregated with RPCA above)

        # Update stored parameters for the next round
        ndarrays = state_dict_to_ndarrays(self.global_state, self.param_names)
        self.parameters = ndarrays_to_parameters(ndarrays)
        metrics = {"server_round": server_round}
        return self.parameters, metrics

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        clients = client_manager.sample(num_clients=self.num_clients, min_num_clients=self.num_clients)
        instructions = []
        for client in clients:
            cid = getattr(client, "cid", "unknown")
            T_dict = self.client_T.get(cid, {})
            client_config = {}
            if T_dict:
                payload = {
                    name: base64.b64encode(v.astype(np.float32).tobytes()).decode("ascii")
                    for name, v in T_dict.items()
                }
                client_config["T_blob"] = json.dumps(payload)
            instructions.append(fl.common.EvaluateIns(parameters=parameters, config=client_config))
        return list(zip(clients, instructions))

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Any],
    ) -> Tuple[float, Dict[str, Any]]:
        if not results:
            return 0.0, {}
        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = 0.0
        weighted_acc = 0.0
        acc_list = []
        loss_list = []
        for _, res in results:
            w = res.num_examples / total_examples if total_examples > 0 else 1.0 / len(results)
            weighted_loss += w * res.loss
            acc_val = res.metrics.get("accuracy", 0.0)
            weighted_acc += w * acc_val
            acc_list.append(acc_val)
            loss_list.append(res.loss)
        acc_arr = np.array(acc_list, dtype=np.float32)
        loss_arr = np.array(loss_list, dtype=np.float32)
        metrics = {
            "personalized_loss": weighted_loss,
            "personalized_accuracy": weighted_acc,
            "personalized_acc_mean": float(np.mean(acc_arr)),
            "personalized_acc_p95": float(np.percentile(acc_arr, 95)),
            "personalized_loss_mean": float(np.mean(loss_arr)),
            "personalized_loss_p95": float(np.percentile(loss_arr, 95)),
            "round": server_round,
        }
        if self.last_global_metrics and self.last_global_metrics.get("round") == server_round:
            ga = self.last_global_metrics.get("global_accuracy")
            gl = self.last_global_metrics.get("global_loss")
            if ga is not None:
                metrics["personalization_gain_acc"] = weighted_acc - ga
            if gl is not None:
                metrics["personalization_gain_loss"] = weighted_loss - gl
        if self.log_fn:
            self.log_fn({"stage": "personalized_eval", **metrics})

        # Enhanced wandb logging for personalized eval
        if self.wandb_logger:
            self.wandb_logger.log_personalized_eval(
                round=server_round,
                weighted_loss=weighted_loss,
                weighted_acc=weighted_acc,
                acc_list=acc_list,
                loss_list=loss_list,
                personalization_gain_acc=metrics.get("personalization_gain_acc"),
                personalization_gain_loss=metrics.get("personalization_gain_loss")
            )

        if self.best_personalized_loss is None or weighted_loss < self.best_personalized_loss - 1e-4:
            self.best_personalized_loss = weighted_loss
            self.no_improve_personalized = 0
        else:
            self.no_improve_personalized += 1
        return weighted_loss, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        # Defer to client-side evaluation; keep global eval for reference if needed
        metrics = self.evaluate_global(self.eval_loader, self.model_builder, parameters=parameters)
        metrics_with_round = {"round": server_round, "global_loss": metrics.get("loss"), "global_accuracy": metrics.get("accuracy")}
        self.logged_metrics.append(metrics_with_round)
        if self.log_fn:
            self.log_fn({"stage": "global_val", **metrics_with_round})

        # Enhanced wandb logging for global eval
        if self.wandb_logger:
            self.wandb_logger.log_global_eval(
                round=server_round,
                loss=metrics.get("loss"),
                accuracy=metrics.get("accuracy")
            )

        self.last_global_metrics = {"round": server_round, **metrics}

        val_loss = metrics.get("loss")
        if val_loss is not None:
            if self.best_val_loss is None or val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.no_improve_rounds = 0
            else:
                self.no_improve_rounds += 1
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
        if model_builder is self.model_builder:  # assume this is the regular global eval path
            self.last_global_metrics = {"round": getattr(parameters, "round", None) or -1, **metrics}
        return metrics


class FedSALoRAStrategy(fl.server.strategy.Strategy):
    """
    Simple FedAvg-style LoRA aggregation baseline.

    Key differences from FedSAFoldStrategy:
    - NO RPCA decomposition
    - NO T-matrix calculation/distribution
    - Simple weighted average of LoRA A updates
    - B matrices and task heads remain local (same as FedSAFold)

    This serves as a baseline to measure the value added by RPCA+folding.
    """

    def __init__(
        self,
        cfg: ExperimentCfg,
        param_names: List[str],
        template_state: Dict[str, torch.Tensor],
        num_clients: int,
        eval_loader,
        model_builder,
        log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
        wandb_logger: Optional[Any] = None,
    ):
        self.cfg = cfg
        self.param_names = param_names
        self.global_state = {k: v.clone() for k, v in template_state.items()}
        self.num_clients = num_clients
        self.eval_loader = eval_loader
        self.model_builder = model_builder
        self.lora_a_names = [n for n in param_names if "lora_A" in n]
        self.parameters = ndarrays_to_parameters(
            state_dict_to_ndarrays(self.global_state, self.param_names)
        )
        self.logged_metrics: List[Dict[str, Any]] = []
        self.log_fn = log_fn
        self.wandb_logger = wandb_logger
        self.best_val_loss: float | None = None
        self.no_improve_rounds: int = 0
        self.best_personalized_loss: float | None = None
        self.no_improve_personalized: int = 0
        self.last_global_metrics: Dict[str, Any] | None = None

    def initialize_parameters(self, client_manager) -> Parameters:
        return self.parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]:
        # Simple FedAvg: no T-matrix, just send global params
        sample_size = self.num_clients
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=sample_size
        )
        instructions = [
            fl.common.FitIns(parameters=parameters, config={})
            for _ in clients
        ]
        return list(zip(clients, instructions))

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[Any],
    ) -> Tuple[Parameters, Dict[str, Any]]:
        if failures:
            print(f"[Server] aggregate_fit saw {len(failures)} failures")

        if not results:
            return self.parameters, {}

        # Collect delta A and weights
        client_delta_a: Dict[str, Dict[str, np.ndarray]] = {}
        client_weights: Dict[str, int] = {}

        for client, fit_res in results:
            cid = getattr(client, "cid", "unknown")
            arrays = parameters_to_ndarrays(fit_res.parameters)
            client_delta_a[cid] = {
                name: arr for name, arr in zip(self.lora_a_names, arrays)
            }
            client_weights[cid] = fit_res.num_examples

        # Simple weighted average for each LoRA A parameter
        client_list = list(client_delta_a.keys())
        weights = np.array([client_weights[cid] for cid in client_list], dtype=np.float32)
        weight_sum = np.sum(weights) if np.sum(weights) > 0 else len(client_list)

        for name in self.lora_a_names:
            A_global = self.global_state[name].cpu().numpy()

            # Weighted average of deltas
            delta_avg = sum(
                w * client_delta_a[cid][name]
                for w, cid in zip(weights, client_list)
            ) / weight_sum

            # Update global A
            self.global_state[name] = torch.tensor(
                A_global + delta_avg, dtype=self.global_state[name].dtype
            )

            # Log aggregation metrics (simpler than RPCA)
            if self.log_fn:
                delta_norm = float(np.linalg.norm(delta_avg))
                delta_variance = float(np.var([
                    np.linalg.norm(client_delta_a[cid][name])
                    for cid in client_list
                ]))
                self.log_fn({
                    "stage": "aggregation",
                    "round": server_round,
                    "name": name,
                    "delta_norm": delta_norm,
                    "delta_variance": delta_variance,
                })

            # Enhanced wandb logging
            if self.wandb_logger:
                delta_norm = float(np.linalg.norm(delta_avg))
                delta_variance = float(np.var([
                    np.linalg.norm(client_delta_a[cid][name])
                    for cid in client_list
                ]))
                self.wandb_logger.log_aggregation_metrics(
                    round=server_round,
                    name=name,
                    delta_norm=delta_norm,
                    delta_variance=delta_variance
                )

        # Update stored parameters
        ndarrays = state_dict_to_ndarrays(self.global_state, self.param_names)
        self.parameters = ndarrays_to_parameters(ndarrays)

        return self.parameters, {"server_round": server_round}

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateIns]]:
        # No T-matrix to send
        clients = client_manager.sample(
            num_clients=self.num_clients, min_num_clients=self.num_clients
        )
        instructions = [
            fl.common.EvaluateIns(parameters=parameters, config={})
            for _ in clients
        ]
        return list(zip(clients, instructions))

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
        failures: List[Any],
    ) -> Tuple[float, Dict[str, Any]]:
        # Same aggregation logic as FedSAFoldStrategy
        if not results:
            return 0.0, {}

        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = 0.0
        weighted_acc = 0.0
        acc_list = []
        loss_list = []

        for _, res in results:
            w = res.num_examples / total_examples if total_examples > 0 else 1.0 / len(results)
            weighted_loss += w * res.loss
            acc_val = res.metrics.get("accuracy", 0.0)
            weighted_acc += w * acc_val
            acc_list.append(acc_val)
            loss_list.append(res.loss)

        acc_arr = np.array(acc_list, dtype=np.float32)
        loss_arr = np.array(loss_list, dtype=np.float32)

        metrics = {
            "personalized_loss": weighted_loss,
            "personalized_accuracy": weighted_acc,
            "personalized_acc_mean": float(np.mean(acc_arr)),
            "personalized_acc_p95": float(np.percentile(acc_arr, 95)),
            "personalized_loss_mean": float(np.mean(loss_arr)),
            "personalized_loss_p95": float(np.percentile(loss_arr, 95)),
            "round": server_round,
        }

        # Compute personalization gain if global metrics available
        if self.last_global_metrics and self.last_global_metrics.get("round") == server_round:
            ga = self.last_global_metrics.get("global_accuracy")
            gl = self.last_global_metrics.get("global_loss")
            if ga is not None:
                metrics["personalization_gain_acc"] = weighted_acc - ga
            if gl is not None:
                metrics["personalization_gain_loss"] = weighted_loss - gl

        if self.log_fn:
            self.log_fn({"stage": "personalized_eval", **metrics})

        # Enhanced wandb logging
        if self.wandb_logger:
            self.wandb_logger.log_personalized_eval(
                round=server_round,
                weighted_loss=weighted_loss,
                weighted_acc=weighted_acc,
                acc_list=acc_list,
                loss_list=loss_list,
                personalization_gain_acc=metrics.get("personalization_gain_acc"),
                personalization_gain_loss=metrics.get("personalization_gain_loss")
            )

        # Early stopping check
        if self.best_personalized_loss is None or weighted_loss < self.best_personalized_loss - 1e-4:
            self.best_personalized_loss = weighted_loss
            self.no_improve_personalized = 0
        else:
            self.no_improve_personalized += 1

        return weighted_loss, metrics

    def evaluate(self, server_round: int, parameters: Parameters):
        # Global evaluation (same as FedSAFoldStrategy)
        metrics = self.evaluate_global(
            self.eval_loader, self.model_builder, parameters=parameters
        )
        metrics_with_round = {
            "round": server_round,
            "global_loss": metrics.get("loss"),
            "global_accuracy": metrics.get("accuracy")
        }
        self.logged_metrics.append(metrics_with_round)

        if self.log_fn:
            self.log_fn({"stage": "global_val", **metrics_with_round})

        # Enhanced wandb logging
        if self.wandb_logger:
            self.wandb_logger.log_global_eval(
                round=server_round,
                loss=metrics.get("loss"),
                accuracy=metrics.get("accuracy")
            )

        self.last_global_metrics = {"round": server_round, **metrics}

        # Track best val loss
        val_loss = metrics.get("loss")
        if val_loss is not None:
            if self.best_val_loss is None or val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.no_improve_rounds = 0
            else:
                self.no_improve_rounds += 1

        return 0.0, metrics

    def evaluate_global(
        self, dataloader, model_builder=None, parameters: Parameters | None = None
    ) -> Dict[str, float]:
        # Same implementation as FedSAFoldStrategy
        arrays = parameters_to_ndarrays(parameters) if parameters is not None \
                 else state_dict_to_ndarrays(self.global_state, self.param_names)
        state = ndarrays_to_state_dict(self.param_names, arrays, self.global_state)
        builder = model_builder or self.model_builder
        model, _, _ = builder()
        model.load_state_dict(state, strict=False)
        device = torch.device(
            self.cfg.train.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        metrics = eval_fn(model, dataloader, device)

        if model_builder is self.model_builder:
            self.last_global_metrics = {"round": getattr(parameters, "round", None) or -1, **metrics}

        return metrics


def create_strategy(
    method: str,
    cfg: ExperimentCfg,
    param_names: List[str],
    template_state: Dict[str, torch.Tensor],
    num_clients: int,
    eval_loader,
    model_builder,
    log_fn: Optional[Callable[[Dict[str, Any]], None]] = None,
    wandb_logger: Optional[Any] = None,
) -> fl.server.strategy.Strategy:
    """
    Factory function to create strategy based on method name.

    Supported methods:
    - "fedsa_fold": RPCA + T-matrix aggregation
    - "fedsa_lora": Simple weighted averaging baseline

    Raises ValueError if method not recognized.
    """
    method = method.lower().strip()

    if method == "fedsa_fold":
        return FedSAFoldStrategy(
            cfg, param_names, template_state,
            num_clients, eval_loader, model_builder,
            log_fn, wandb_logger
        )
    elif method == "fedsa_lora":
        return FedSALoRAStrategy(
            cfg, param_names, template_state,
            num_clients, eval_loader, model_builder,
            log_fn, wandb_logger
        )
    else:
        raise ValueError(
            f"Unknown method: {method}. Supported: ['fedsa_fold', 'fedsa_lora']"
        )
