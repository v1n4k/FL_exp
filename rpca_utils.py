"""Robust PCA utilities with a safe fallback implementation."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def _soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def _fallback_rpca(M: np.ndarray, lam: float | None, max_iter: int, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Inexact ALM RPCA fallback to avoid dependency issues."""

    m, n = M.shape
    lam = lam or 1.0 / np.sqrt(max(m, n))
    mu = (m * n) / (4.0 * np.linalg.norm(M, 1) + 1e-8)
    L = np.zeros_like(M)
    S = np.zeros_like(M)
    Y = M / max(np.linalg.norm(M, 2), 1e-8)

    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(M - S + (1.0 / mu) * Y, full_matrices=False)
        s_shrink = _soft_threshold(s, 1.0 / mu)
        rank = (s_shrink > 0).sum()
        if rank == 0:
            L = np.zeros_like(M)
        else:
            L = (U[:, :rank] * s_shrink[:rank]) @ Vt[:rank, :]

        S = _soft_threshold(M - L + (1.0 / mu) * Y, lam / mu)
        Y = Y + mu * (M - L - S)

        err = np.linalg.norm(M - L - S, ord="fro") / (np.linalg.norm(M, ord="fro") + 1e-8)
        if err < tol:
            break
    return L, S


def robust_pca(M: np.ndarray, lam: float | None = None, max_iter: int = 100, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Run RPCA with `r_pca` when available, otherwise use a fallback solver."""

    try:
        from r_pca import R_pca  # type: ignore

        rpca = R_pca(M)
        L, S = rpca.fit(max_iter=max_iter, iter_stop=tol)
        return L, S
    except Exception:
        return _fallback_rpca(M, lam=lam, max_iter=max_iter, tol=tol)

