from __future__ import annotations

from typing import Dict

import torch

from .model import TrajectoryPINN
from .target import BananaTarget


def compute_raw_losses(
    model: TrajectoryPINN,
    target: BananaTarget,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Returns raw losses only, without any weighting.

    Keys:
        - sup
        - phys
        - ic
        - q_sup
        - p_sup
    """
    dim = model.dim

    q0 = batch_x[:, :dim]
    p0 = batch_x[:, dim : 2 * dim]
    t = batch_x[:, 2 * dim : 2 * dim + 1].clone().detach().requires_grad_(True)

    q_true = batch_y[:, :dim]
    p_true = batch_y[:, dim:]

    q_hat, p_hat = model(q0, p0, t)

    q_sup = torch.mean((q_hat - q_true) ** 2)
    p_sup = torch.mean((p_hat - p_true) ** 2)
    sup = q_sup + p_sup

    dqdt_list = []
    dpdt_list = []
    for d in range(dim):
        dqdt_d = torch.autograd.grad(
            q_hat[:, d].sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0]
        dpdt_d = torch.autograd.grad(
            p_hat[:, d].sum(),
            t,
            create_graph=True,
            retain_graph=True,
        )[0]
        dqdt_list.append(dqdt_d)
        dpdt_list.append(dpdt_d)

    dqdt = torch.cat(dqdt_list, dim=-1)
    dpdt = torch.cat(dpdt_list, dim=-1)

    gradU = target.grad_U(q_hat)
    phys_q = dqdt - p_hat
    phys_p = dpdt + gradU
    phys = torch.mean(phys_q**2) + torch.mean(phys_p**2)

    t0 = torch.zeros_like(t)
    q_ic, p_ic = model(q0, p0, t0)
    ic = torch.mean((q_ic - q0) ** 2) + torch.mean((p_ic - p0) ** 2)

    return {
        "sup": sup,
        "phys": phys,
        "ic": ic,
        "q_sup": q_sup,
        "p_sup": p_sup,
    }

from .model import HamiltonianNN


def compute_hnn_raw_losses(
    model: HamiltonianNN,
    target: BananaTarget,
    batch_x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    HNN training data:
        batch_x = [q, p]
    True dynamics:
        dq/dt = p
        dp/dt = -grad_U(q)

    HNN predicts a scalar H_theta(q,p), and we train with:
        dH/dp ~ p
        -dH/dq ~ -grad_U(q)
    """
    dim = model.dim

    q = batch_x[:, :dim].clone().detach().requires_grad_(True)
    p = batch_x[:, dim:].clone().detach().requires_grad_(True)

    H_hat = model(q, p)  # [batch, 1]

    dH_dq = torch.autograd.grad(
        H_hat.sum(), q, create_graph=True, retain_graph=True
    )[0]
    dH_dp = torch.autograd.grad(
        H_hat.sum(), p, create_graph=True, retain_graph=True
    )[0]

    dqdt_true = p
    dpdt_true = -target.grad_U(q)

    q_loss = torch.mean((dH_dp - dqdt_true) ** 2)
    p_loss = torch.mean((-dH_dq - dpdt_true) ** 2)

    total = q_loss + p_loss

    return {
        "sup": total,
        "phys": torch.zeros((), device=total.device),  # placeholders for logging compatibility
        "ic": torch.zeros((), device=total.device),
        "q_sup": q_loss,
        "p_sup": p_loss,
    }