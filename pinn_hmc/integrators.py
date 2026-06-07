from __future__ import annotations

from typing import Callable, Tuple

import torch


def leapfrog_torch(
    q: torch.Tensor,
    p: torch.Tensor,
    grad_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    step_size: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Störmer-Verlet leapfrog integrator (torch, batched).

    grad_fn(q, p) -> force [batch, dim]  (used for p-update only; q-update uses p)
    Returns (q_new, -p_new, n_grad_evals).  n_grad_evals = n_steps + 1.
    """
    q = q.clone()
    p = p.clone()

    p = p - 0.5 * step_size * grad_fn(q, p)
    n_evals = 1

    for i in range(n_steps - 1):
        q = q + step_size * p
        p = p - step_size * grad_fn(q, p)
        n_evals += 1

    q = q + step_size * p
    p = p - 0.5 * step_size * grad_fn(q, p)
    n_evals += 1

    return q, -p, n_evals


def yoshida4_torch(
    q: torch.Tensor,
    p: torch.Tensor,
    grad_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    step_size: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Yoshida 4th-order symplectic integrator (torch, batched).

    Coefficients: w1 = 1/(2-2^(1/3)) ≈ 1.3512, w0 = -2^(1/3)/(2-2^(1/3)) ≈ -1.7024
    Each Yoshida step = 3 leapfrog sub-steps with weights [w1, w0, w1].

    grad_fn(q, p) -> force [batch, dim]
    Returns (q_new, -p_new, n_grad_evals).  n_grad_evals = 6 * n_steps.
    """
    cbrt2 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - cbrt2)
    w0 = -cbrt2 / (2.0 - cbrt2)  # ≈ -1.7024

    q = q.clone()
    p = p.clone()
    n_evals = 0

    for _ in range(n_steps):
        for wi in (w1, w0, w1):
            h = wi * step_size
            p = p - 0.5 * h * grad_fn(q, p)
            n_evals += 1
            q = q + h * p
            p = p - 0.5 * h * grad_fn(q, p)
            n_evals += 1

    return q, -p, n_evals
