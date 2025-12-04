"""Enhanced wandb logging utilities with hierarchical metrics and fairness tracking."""
from __future__ import annotations

from typing import Dict, Any, List
import numpy as np


class WandbLogger:
    """
    Wrapper for wandb logging with hierarchical metric naming and fairness tracking.

    Metric Hierarchy:
    - server/rpca/*: RPCA decomposition metrics
    - server/orthogonality/*: LoRA A orthogonality diagnostics
    - server/tmatrix/*: T-matrix statistics
    - server/aggregation/*: Simple aggregation metrics (FedSALoRA)
    - fairness/*: Cross-client fairness metrics
    - global/*: Global model evaluation
    - personalized/*: Personalized model evaluation
    """

    def __init__(self, wandb_run, method: str):
        self.wandb_run = wandb_run
        self.method = method

    def log(self, metrics: Dict[str, Any]):
        """Log metrics with error handling."""
        if not self.wandb_run:
            return

        try:
            self.wandb_run.log(metrics)
        except Exception as exc:
            print(f"[WandbLogger] log failed: {exc}")

    def log_rpca_metrics(self, round: int, name: str, L: np.ndarray,
                         S: np.ndarray, L_mean: np.ndarray):
        """Log RPCA decomposition metrics with hierarchical naming."""
        metrics = {
            "round": round,
            "server/rpca/L_norm": float(np.linalg.norm(L)),
            "server/rpca/S_norm": float(np.linalg.norm(S)),
            "server/rpca/L_mean_norm": float(np.linalg.norm(L_mean)),
            "server/rpca/L_fro": float(np.linalg.norm(L, ord='fro')),
            "server/rpca/S_fro": float(np.linalg.norm(S, ord='fro')),
            "server/rpca/L_to_S_ratio": float(np.linalg.norm(L) / (np.linalg.norm(S) + 1e-8)),
            "server/rpca/param_name": name,
        }
        self.log(metrics)

    def log_orthogonality_metrics(self, round: int, name: str, A: np.ndarray):
        """Log LoRA A orthogonality diagnostics."""
        gram = A @ A.T
        ident = np.eye(gram.shape[0], dtype=gram.dtype)
        ortho_dev = float(np.linalg.norm(gram - ident, ord="fro"))

        try:
            sv = np.linalg.svd(A, compute_uv=False)
            sv_min = float(np.min(sv)) if sv.size else 0.0
            sv_max = float(np.max(sv)) if sv.size else 0.0
            condition_number = sv_max / (sv_min + 1e-8)
        except Exception:
            sv_min = sv_max = condition_number = 0.0

        metrics = {
            "round": round,
            "server/orthogonality/gram_deviation_fro": ortho_dev,
            "server/orthogonality/sv_min": sv_min,
            "server/orthogonality/sv_max": sv_max,
            "server/orthogonality/condition_number": condition_number,
            "server/orthogonality/param_name": name,
        }
        self.log(metrics)

    def log_tmatrix_stats(self, round: int, name: str, t_norms: List[float],
                          t_maxes: List[float], t_ratios: List[float]):
        """Log T-matrix statistics across clients."""
        t_norms_np = np.array(t_norms, dtype=np.float32)
        t_maxes_np = np.array(t_maxes, dtype=np.float32)
        t_ratios_np = np.array(t_ratios, dtype=np.float32)

        metrics = {
            "round": round,
            "server/tmatrix/norm_mean": float(np.mean(t_norms_np)),
            "server/tmatrix/norm_std": float(np.std(t_norms_np)),
            "server/tmatrix/norm_p95": float(np.percentile(t_norms_np, 95)),
            "server/tmatrix/norm_max": float(np.max(t_norms_np)),
            "server/tmatrix/absmax_mean": float(np.mean(t_maxes_np)),
            "server/tmatrix/absmax_p95": float(np.percentile(t_maxes_np, 95)),
            "server/tmatrix/T_over_S_mean": float(np.mean(t_ratios_np)),
            "server/tmatrix/T_over_S_p95": float(np.percentile(t_ratios_np, 95)),
            "server/tmatrix/param_name": name,
        }
        self.log(metrics)

    def log_aggregation_metrics(self, round: int, name: str, delta_norm: float,
                                delta_variance: float):
        """Log simple aggregation metrics (for FedSALoRA)."""
        metrics = {
            "round": round,
            "server/aggregation/delta_norm": delta_norm,
            "server/aggregation/delta_variance": delta_variance,
            "server/aggregation/param_name": name,
        }
        self.log(metrics)

    def log_fairness_metrics(self, round: int, client_accuracies: List[float],
                            client_losses: List[float]):
        """
        Log fairness metrics across clients.

        Metrics: variance, worst-case, best-case, Gini coefficient
        """
        acc_arr = np.array(client_accuracies, dtype=np.float32)
        loss_arr = np.array(client_losses, dtype=np.float32)

        # Gini coefficient for accuracy
        def gini_coefficient(x):
            sorted_x = np.sort(x)
            n = len(x)
            cumsum = np.cumsum(sorted_x)
            return (2 * np.sum((n - np.arange(n)) * sorted_x)) / (n * cumsum[-1]) - (n + 1) / n

        metrics = {
            "round": round,
            "fairness/accuracy_variance": float(np.var(acc_arr)),
            "fairness/accuracy_std": float(np.std(acc_arr)),
            "fairness/accuracy_worst": float(np.min(acc_arr)),
            "fairness/accuracy_best": float(np.max(acc_arr)),
            "fairness/accuracy_gini": float(gini_coefficient(acc_arr)),
            "fairness/loss_variance": float(np.var(loss_arr)),
            "fairness/loss_std": float(np.std(loss_arr)),
            "fairness/loss_worst": float(np.max(loss_arr)),
            "fairness/loss_best": float(np.min(loss_arr)),
            "fairness/accuracy_range": float(np.max(acc_arr) - np.min(acc_arr)),
            "fairness/loss_range": float(np.max(loss_arr) - np.min(loss_arr)),
        }
        self.log(metrics)

    def log_global_eval(self, round: int, loss: float, accuracy: float):
        """Log global model evaluation metrics."""
        metrics = {
            "round": round,
            "global/loss": loss,
            "global/accuracy": accuracy,
        }
        self.log(metrics)

    def log_personalized_eval(self, round: int, weighted_loss: float, weighted_acc: float,
                             acc_list: List[float], loss_list: List[float],
                             personalization_gain_acc: float | None = None,
                             personalization_gain_loss: float | None = None):
        """Log personalized evaluation metrics."""
        acc_arr = np.array(acc_list, dtype=np.float32)
        loss_arr = np.array(loss_list, dtype=np.float32)

        metrics = {
            "round": round,
            "personalized/loss_weighted": weighted_loss,
            "personalized/accuracy_weighted": weighted_acc,
            "personalized/accuracy_mean": float(np.mean(acc_arr)),
            "personalized/accuracy_std": float(np.std(acc_arr)),
            "personalized/accuracy_p95": float(np.percentile(acc_arr, 95)),
            "personalized/loss_mean": float(np.mean(loss_arr)),
            "personalized/loss_std": float(np.std(loss_arr)),
            "personalized/loss_p95": float(np.percentile(loss_arr, 95)),
        }

        if personalization_gain_acc is not None:
            metrics["personalized/gain_accuracy"] = personalization_gain_acc
        if personalization_gain_loss is not None:
            metrics["personalized/gain_loss"] = personalization_gain_loss

        self.log(metrics)

        # Also log fairness metrics
        self.log_fairness_metrics(round, acc_list, loss_list)
