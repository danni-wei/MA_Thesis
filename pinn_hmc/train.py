from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from .balancers import LossBalancer
from .losses import compute_raw_losses
from .model import TrajectoryPINN
from .target import BananaTarget


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    verbose_every: int = 10


def evaluate_model(
    model: TrajectoryPINN,
    target: BananaTarget,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    balancer: LossBalancer,
) -> Dict[str, float]:
    model.eval()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total = {
        "sup": 0.0,
        "phys": 0.0,
        "ic": 0.0,
        "q_sup": 0.0,
        "p_sup": 0.0,
        "total_loss": 0.0,
    }
    n = 0

    for bx, by in loader:
        with torch.enable_grad():
            raw = compute_raw_losses(model, target, bx, by)
            loss, _ = balancer(raw, model)

        bs = bx.shape[0]
        n += bs
        total["sup"] += float(raw["sup"].detach().item()) * bs
        total["phys"] += float(raw["phys"].detach().item()) * bs
        total["ic"] += float(raw["ic"].detach().item()) * bs
        total["q_sup"] += float(raw["q_sup"].detach().item()) * bs
        total["p_sup"] += float(raw["p_sup"].detach().item()) * bs
        total["total_loss"] += float(loss.detach().item()) * bs

    return {k: v / max(n, 1) for k, v in total.items()}


def train_model(
    model: TrajectoryPINN,
    target: BananaTarget,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    train_cfg: TrainConfig,
    balancer: LossBalancer,
) -> Tuple[TrajectoryPINN, List[Dict[str, float]]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: List[Dict[str, float]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()

        running = {
            "sup": 0.0,
            "phys": 0.0,
            "ic": 0.0,
            "q_sup": 0.0,
            "p_sup": 0.0,
            "total_loss": 0.0,
            "lambda_sup": 0.0,
            "lambda_phys": 0.0,
            "lambda_ic": 0.0,
        }
        n_seen = 0

        for bx, by in train_loader:
            optimizer.zero_grad(set_to_none=True)

            raw = compute_raw_losses(model, target, bx, by)
            total_loss, balancer_stats = balancer(raw, model)

            total_loss.backward()
            if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            bs = bx.shape[0]
            n_seen += bs

            running["sup"] += float(raw["sup"].detach().item()) * bs
            running["phys"] += float(raw["phys"].detach().item()) * bs
            running["ic"] += float(raw["ic"].detach().item()) * bs
            running["q_sup"] += float(raw["q_sup"].detach().item()) * bs
            running["p_sup"] += float(raw["p_sup"].detach().item()) * bs
            running["total_loss"] += float(total_loss.detach().item()) * bs
            running["lambda_sup"] += float(balancer_stats["lambda_sup"].item()) * bs
            running["lambda_phys"] += float(balancer_stats["lambda_phys"].item()) * bs
            running["lambda_ic"] += float(balancer_stats["lambda_ic"].item()) * bs

        train_metrics = {k: v / max(n_seen, 1) for k, v in running.items()}
        val_metrics = evaluate_model(
            model=model,
            target=target,
            x=val_x,
            y=val_y,
            batch_size=train_cfg.batch_size,
            balancer=balancer,
        )

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if epoch % train_cfg.verbose_every == 0 or epoch == 1 or epoch == train_cfg.epochs:
            print(
                f"Epoch {epoch:03d} | "
                f"train total={train_metrics['total_loss']:.5e}, "
                f"sup={train_metrics['sup']:.5e}, phys={train_metrics['phys']:.5e}, ic={train_metrics['ic']:.5e} | "
                f"lambda=({train_metrics['lambda_sup']:.3e}, {train_metrics['lambda_phys']:.3e}, {train_metrics['lambda_ic']:.3e}) | "
                f"val total={val_metrics['total_loss']:.5e}"
            )

    return model, history

from .losses import compute_hnn_raw_losses
from .model import HamiltonianNN


def evaluate_hnn_model(
    model: HamiltonianNN,
    target: BananaTarget,
    x: torch.Tensor,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total = {
        "sup": 0.0,
        "q_sup": 0.0,
        "p_sup": 0.0,
    }
    n = 0

    for (bx,) in loader:
        with torch.enable_grad():
            raw = compute_hnn_raw_losses(model, target, bx)

        bs = bx.shape[0]
        n += bs
        total["sup"] += float(raw["sup"].detach().item()) * bs
        total["q_sup"] += float(raw["q_sup"].detach().item()) * bs
        total["p_sup"] += float(raw["p_sup"].detach().item()) * bs

    return {k: v / max(n, 1) for k, v in total.items()}


def train_hnn_model(
    model: HamiltonianNN,
    target: BananaTarget,
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    train_cfg: TrainConfig,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    train_dataset = TensorDataset(train_x)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()

        running = {"sup": 0.0, "q_sup": 0.0, "p_sup": 0.0}
        n_seen = 0

        for (bx,) in train_loader:
            optimizer.zero_grad(set_to_none=True)

            raw = compute_hnn_raw_losses(model, target, bx)
            total_loss = raw["sup"]

            total_loss.backward()
            if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            bs = bx.shape[0]
            n_seen += bs
            running["sup"] += float(raw["sup"].detach().item()) * bs
            running["q_sup"] += float(raw["q_sup"].detach().item()) * bs
            running["p_sup"] += float(raw["p_sup"].detach().item()) * bs

        train_metrics = {k: v / max(n_seen, 1) for k, v in running.items()}
        val_metrics = evaluate_hnn_model(
            model=model,
            target=target,
            x=val_x,
            batch_size=train_cfg.batch_size,
        )

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        if epoch % train_cfg.verbose_every == 0 or epoch == 1 or epoch == train_cfg.epochs:
            print(
                f"[HNN] Epoch {epoch:03d} | "
                f"train sup={train_metrics['sup']:.5e}, "
                f"q={train_metrics['q_sup']:.5e}, p={train_metrics['p_sup']:.5e} | "
                f"val sup={val_metrics['sup']:.5e}"
            )

    return model, history