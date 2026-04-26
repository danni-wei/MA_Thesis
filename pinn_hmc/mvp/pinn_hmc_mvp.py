import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# PINN + HMC MVP
# ------------------------------------------------------------
# This is a self-contained prototype for the idea:
# - baseline HMC with leapfrog on a 2D banana-shaped target
# - teacher trajectory generation (supervision)
# - PINN-style trajectory surrogate: (q0, p0, t) -> (q(t), p(t))
# - hybrid training loss: supervised + physics residual + IC loss
# - PINN-based proposal inside HMC with exact MH correction
#
# Default device: CUDA if available, otherwise CPU.
# Framework: PyTorch.
# ============================================================


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Target distribution: 2D banana potential
# ============================================================


class BananaTarget:
    """
    2D banana-shaped target.

    Potential:
        U(q1, q2) = 0.5 * (q1 / sigma1)^2
                    + 0.5 * ((q2 - b * (q1^2 - sigma1^2)) / sigma2)^2

    Momentum kinetic energy:
        K(p) = 0.5 * p^T p

    Hamiltonian:
        H(q, p) = U(q) + K(p)
    """

    def __init__(self, b: float = 0.1, sigma1: float = 1.0, sigma2: float = 1.0):
        self.b = float(b)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.dim = 2

    def U(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] != 2:
            raise ValueError(f"Expected q[..., 2], got shape {tuple(q.shape)}")
        q1 = q[..., 0]
        q2 = q[..., 1]
        z = q2 - self.b * (q1**2 - self.sigma1**2)
        return 0.5 * (q1 / self.sigma1) ** 2 + 0.5 * (z / self.sigma2) ** 2

    def grad_U(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] != 2:
            raise ValueError(f"Expected q[..., 2], got shape {tuple(q.shape)}")
        q1 = q[..., 0]
        q2 = q[..., 1]
        z = q2 - self.b * (q1**2 - self.sigma1**2)

        dU_dq1 = q1 / (self.sigma1**2) - (2.0 * self.b * q1 * z) / (self.sigma2**2)
        dU_dq2 = z / (self.sigma2**2)
        return torch.stack([dU_dq1, dU_dq2], dim=-1)

    def K(self, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(p**2, dim=-1)

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.U(q) + self.K(p)

    def sample_momentum(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.dim, device=device)


# ============================================================
# Baseline leapfrog integrator and HMC
# ============================================================


def leapfrog(
    target: BananaTarget,
    q0: torch.Tensor,
    p0: torch.Tensor,
    step_size: float,
    n_steps: int,
    return_trajectory: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Standard leapfrog integrator.

    Args:
        q0, p0: shape [batch, dim]
        step_size: epsilon
        n_steps: number of leapfrog steps
        return_trajectory: whether to store states for all steps

    Returns:
        q, p, traj
        traj contains:
            times: [n_steps+1]
            q: [batch, n_steps+1, dim]
            p: [batch, n_steps+1, dim]
    """
    if q0.shape != p0.shape:
        raise ValueError(f"q0 and p0 must have same shape, got {q0.shape} vs {p0.shape}")
    if q0.ndim != 2:
        raise ValueError(f"Expected q0/p0 shape [batch, dim], got {q0.shape}")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    q = q0.clone()
    p = p0.clone()

    traj_q: List[torch.Tensor] = []
    traj_p: List[torch.Tensor] = []
    if return_trajectory:
        traj_q.append(q.clone())
        traj_p.append(p.clone())

    p = p - 0.5 * step_size * target.grad_U(q)
    for i in range(n_steps):
        q = q + step_size * p
        if i != n_steps - 1:
            p = p - step_size * target.grad_U(q)
        if return_trajectory:
            traj_q.append(q.clone())
            # store full-step momentum approximation for convenience
            if i != n_steps - 1:
                p_store = p.clone()
            else:
                p_store = (p - 0.5 * step_size * target.grad_U(q)).clone()
            traj_p.append(p_store)

    p = p - 0.5 * step_size * target.grad_U(q)
    p = -p  # momentum flip for reversibility in proposal representation

    if return_trajectory:
        traj_p[-1] = p.clone()
        times = torch.linspace(
            0.0,
            n_steps * step_size,
            n_steps + 1,
            device=q0.device,
            dtype=q0.dtype,
        )
        traj = {
            "times": times,
            "q": torch.stack(traj_q, dim=1),
            "p": torch.stack(traj_p, dim=1),
        }
        return q, p, traj

    return q, p, None


@torch.no_grad()
def hmc_step_baseline(
    target: BananaTarget,
    q_current: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    One baseline HMC step for a batch of particles.
    q_current shape: [batch, dim]
    """
    batch_size = q_current.shape[0]
    p0 = target.sample_momentum(batch_size, q_current.device)
    q_prop, p_prop, _ = leapfrog(target, q_current, p0, step_size, n_steps, return_trajectory=False)

    H0 = target.H(q_current, p0)
    H1 = target.H(q_prop, p_prop)
    log_alpha = torch.clamp(H0 - H1, max=80.0)
    alpha = torch.exp(log_alpha).clamp(max=1.0)
    u = torch.rand_like(alpha)
    accept = u < alpha
    q_next = torch.where(accept[:, None], q_prop, q_current)

    info = {
        "accept_rate": float(accept.float().mean().item()),
        "mean_delta_H": float((H1 - H0).mean().item()),
        "mean_abs_delta_H": float((H1 - H0).abs().mean().item()),
    }
    return q_next, accept, alpha, info


@torch.no_grad()
def run_hmc_baseline(
    target: BananaTarget,
    n_samples: int,
    burn_in: int,
    step_size: float,
    n_steps: int,
    init_q: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = device or get_device()
    q = init_q.clone().to(device) if init_q is not None else torch.zeros(1, target.dim, device=device)
    if q.ndim == 1:
        q = q.unsqueeze(0)

    samples = []
    accept_rates = []
    delta_Hs = []

    total_steps = burn_in + n_samples
    for i in range(total_steps):
        q, _, _, info = hmc_step_baseline(target, q, step_size, n_steps)
        accept_rates.append(info["accept_rate"])
        delta_Hs.append(info["mean_abs_delta_H"])
        if i >= burn_in:
            samples.append(q.squeeze(0).detach().cpu())

    out = torch.stack(samples, dim=0)
    stats = {
        "mean_accept_rate": float(sum(accept_rates) / len(accept_rates)),
        "mean_abs_delta_H": float(sum(delta_Hs) / len(delta_Hs)),
    }
    return out, stats


# ============================================================
# Teacher trajectory generation
# ============================================================


@dataclass
class DatasetConfig:
    n_trajectories: int = 1500
    n_time_points: int = 21
    step_size: float = 0.08
    n_leapfrog_steps: int = 20
    q0_std: float = 1.5
    train_split: float = 0.9
    batch_size: int = 256


@torch.no_grad()
def generate_teacher_dataset(
    target: BananaTarget,
    config: DatasetConfig,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """
    Generates training tuples:
        input  = (q0, p0, t)
        target = (q(t), p(t))

    q0 is sampled from a simple Gaussian proposal for coverage.
    p0 is sampled from standard Gaussian as in HMC.
    Trajectories are generated using leapfrog.
    """
    device = device or get_device()

    q0 = config.q0_std * torch.randn(config.n_trajectories, target.dim, device=device)
    p0 = target.sample_momentum(config.n_trajectories, device)

    qT, pT, traj = leapfrog(
        target=target,
        q0=q0,
        p0=p0,
        step_size=config.step_size,
        n_steps=config.n_leapfrog_steps,
        return_trajectory=True,
    )
    assert traj is not None

    times = traj["times"]
    traj_q = traj["q"]
    traj_p = traj["p"]

    # Subsample time points if needed
    total_points = traj_q.shape[1]
    if config.n_time_points > total_points:
        raise ValueError(
            f"Requested n_time_points={config.n_time_points}, but trajectory has only {total_points} points"
        )

    indices = torch.linspace(0, total_points - 1, config.n_time_points, device=device).long()
    times = times[indices]
    traj_q = traj_q[:, indices, :]
    traj_p = traj_p[:, indices, :]

    n_traj, n_t, dim = traj_q.shape
    q0_rep = q0[:, None, :].expand(n_traj, n_t, dim)
    p0_rep = p0[:, None, :].expand(n_traj, n_t, dim)
    t_rep = times[None, :, None].expand(n_traj, n_t, 1)

    inputs = torch.cat([q0_rep, p0_rep, t_rep], dim=-1).reshape(-1, 2 * dim + 1)
    targets = torch.cat([traj_q, traj_p], dim=-1).reshape(-1, 2 * dim)

    # Train / validation split by flattened samples for simplicity
    n_total = inputs.shape[0]
    n_train = int(config.train_split * n_total)
    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    return {
        "train_x": inputs[train_idx],
        "train_y": targets[train_idx],
        "val_x": inputs[val_idx],
        "val_y": targets[val_idx],
        "trajectory_times": times,
        "q0_examples": q0[:16],
        "p0_examples": p0[:16],
    }


# ============================================================
# PINN model
# ============================================================


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")

        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrajectoryPINN(nn.Module):
    """
    Input:  [q0, p0, t] with shape [batch, 5] for dim=2
    Output: [q_hat, p_hat] with shape [batch, 4]
    """

    def __init__(self, dim: int, hidden_dim: int = 128, depth: int = 4):
        super().__init__()
        self.dim = dim
        self.model = MLP(in_dim=2 * dim + 1, out_dim=2 * dim, hidden_dim=hidden_dim, depth=depth)

    def forward(self, q0: torch.Tensor, p0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = torch.cat([q0, p0, t], dim=-1)
        out = self.model(x)
        q_hat = out[:, : self.dim]
        p_hat = out[:, self.dim :]
        return q_hat, p_hat

    def forward_flat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q0 = x[:, : self.dim]
        p0 = x[:, self.dim : 2 * self.dim]
        t = x[:, 2 * self.dim : 2 * self.dim + 1]
        return self.forward(q0, p0, t)


# ============================================================
# Losses
# ============================================================


def compute_losses(
    model: TrajectoryPINN,
    target: BananaTarget,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    lambda_sup: float,
    lambda_phys: float,
    lambda_ic: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Hybrid loss:
        L = lambda_sup * supervised + lambda_phys * physics + lambda_ic * initial_condition
    """
    dim = model.dim

    q0 = batch_x[:, :dim]
    p0 = batch_x[:, dim : 2 * dim]
    t = batch_x[:, 2 * dim : 2 * dim + 1].clone().detach().requires_grad_(True)

    q_true = batch_y[:, :dim]
    p_true = batch_y[:, dim:]

    q_hat, p_hat = model(q0, p0, t)

    # Supervised trajectory fitting
    loss_sup = torch.mean((q_hat - q_true) ** 2) + torch.mean((p_hat - p_true) ** 2)

    # Physics residuals using autodiff wrt time
    dqdt = []
    dpdt = []
    for d in range(dim):
        dqdt_d = torch.autograd.grad(
            q_hat[:, d].sum(), t, create_graph=True, retain_graph=True
        )[0]
        dpdt_d = torch.autograd.grad(
            p_hat[:, d].sum(), t, create_graph=True, retain_graph=True
        )[0]
        dqdt.append(dqdt_d)
        dpdt.append(dpdt_d)

    dqdt = torch.cat(dqdt, dim=-1)
    dpdt = torch.cat(dpdt, dim=-1)

    gradU = target.grad_U(q_hat)
    phys_q = dqdt - p_hat
    phys_p = dpdt + gradU
    loss_phys = torch.mean(phys_q**2) + torch.mean(phys_p**2)

    # Initial condition loss: evaluate at t=0 using same (q0, p0)
    t0 = torch.zeros_like(t)
    q_ic, p_ic = model(q0, p0, t0)
    loss_ic = torch.mean((q_ic - q0) ** 2) + torch.mean((p_ic - p0) ** 2)

    loss = lambda_sup * loss_sup + lambda_phys * loss_phys + lambda_ic * loss_ic
    metrics = {
        "loss": loss.detach(),
        "loss_sup": loss_sup.detach(),
        "loss_phys": loss_phys.detach(),
        "loss_ic": loss_ic.detach(),
    }
    return loss, metrics


# ============================================================
# Training
# ============================================================


@dataclass
class TrainConfig:
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-6
    hidden_dim: int = 128
    depth: int = 4
    lambda_sup: float = 1.0
    lambda_phys: float = 0.1
    lambda_ic: float = 1.0
    grad_clip: float = 1.0
    verbose_every: int = 10


@torch.no_grad()
def evaluate_model(
    model: TrajectoryPINN,
    target: BananaTarget,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    train_cfg: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total = {"loss": 0.0, "loss_sup": 0.0, "loss_phys": 0.0, "loss_ic": 0.0}
    n = 0
    for bx, by in loader:
        # Need gradients for physics residual, so disable outer no_grad locally
        with torch.enable_grad():
            loss, metrics = compute_losses(
                model,
                target,
                bx,
                by,
                train_cfg.lambda_sup,
                train_cfg.lambda_phys,
                train_cfg.lambda_ic,
            )
        bs = bx.shape[0]
        n += bs
        for k in total:
            total[k] += float(metrics[k].item()) * bs

    return {k: v / max(n, 1) for k, v in total.items()}


def train_pinn(
    target: BananaTarget,
    dataset_dict: Dict[str, torch.Tensor],
    data_cfg: DatasetConfig,
    train_cfg: TrainConfig,
    device: Optional[torch.device] = None,
) -> Tuple[TrajectoryPINN, List[Dict[str, float]]]:
    device = device or get_device()

    model = TrajectoryPINN(
        dim=target.dim,
        hidden_dim=train_cfg.hidden_dim,
        depth=train_cfg.depth,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )

    train_dataset = TensorDataset(dataset_dict["train_x"], dataset_dict["train_y"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    history: List[Dict[str, float]] = []

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        running = {"loss": 0.0, "loss_sup": 0.0, "loss_phys": 0.0, "loss_ic": 0.0}
        n_seen = 0

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss, metrics = compute_losses(
                model,
                target,
                bx,
                by,
                train_cfg.lambda_sup,
                train_cfg.lambda_phys,
                train_cfg.lambda_ic,
            )
            loss.backward()
            if train_cfg.grad_clip is not None and train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()

            bs = bx.shape[0]
            n_seen += bs
            for k in running:
                running[k] += float(metrics[k].item()) * bs

        train_metrics = {k: v / max(n_seen, 1) for k, v in running.items()}
        val_metrics = evaluate_model(
            model,
            target,
            dataset_dict["val_x"].to(device),
            dataset_dict["val_y"].to(device),
            data_cfg.batch_size,
            train_cfg,
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
                f"train loss={train_metrics['loss']:.5e}, sup={train_metrics['loss_sup']:.5e}, "
                f"phys={train_metrics['loss_phys']:.5e}, ic={train_metrics['loss_ic']:.5e} | "
                f"val loss={val_metrics['loss']:.5e}"
            )

    return model, history


# ============================================================
# PINN-based proposal HMC
# ============================================================


@torch.no_grad()
def pinn_proposal(
    model: TrajectoryPINN,
    q_current: torch.Tensor,
    p0: torch.Tensor,
    t_final: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.full((q_current.shape[0], 1), float(t_final), device=q_current.device, dtype=q_current.dtype)
    q_prop, p_prop = model(q_current, p0, t)
    p_prop = -p_prop  # keep proposal representation aligned with HMC convention
    return q_prop, p_prop


@torch.no_grad()
def hmc_step_pinn(
    model: TrajectoryPINN,
    target: BananaTarget,
    q_current: torch.Tensor,
    t_final: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float]]:
    batch_size = q_current.shape[0]
    p0 = target.sample_momentum(batch_size, q_current.device)
    q_prop, p_prop = pinn_proposal(model, q_current, p0, t_final)

    H0 = target.H(q_current, p0)
    H1 = target.H(q_prop, p_prop)
    log_alpha = torch.clamp(H0 - H1, max=80.0)
    alpha = torch.exp(log_alpha).clamp(max=1.0)
    u = torch.rand_like(alpha)
    accept = u < alpha
    q_next = torch.where(accept[:, None], q_prop, q_current)

    info = {
        "accept_rate": float(accept.float().mean().item()),
        "mean_delta_H": float((H1 - H0).mean().item()),
        "mean_abs_delta_H": float((H1 - H0).abs().mean().item()),
    }
    return q_next, accept, alpha, info


@torch.no_grad()
def run_hmc_pinn(
    model: TrajectoryPINN,
    target: BananaTarget,
    n_samples: int,
    burn_in: int,
    t_final: float,
    init_q: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = device or get_device()
    q = init_q.clone().to(device) if init_q is not None else torch.zeros(1, target.dim, device=device)
    if q.ndim == 1:
        q = q.unsqueeze(0)

    samples = []
    accept_rates = []
    delta_Hs = []

    total_steps = burn_in + n_samples
    for i in range(total_steps):
        q, _, _, info = hmc_step_pinn(model, target, q, t_final)
        accept_rates.append(info["accept_rate"])
        delta_Hs.append(info["mean_abs_delta_H"])
        if i >= burn_in:
            samples.append(q.squeeze(0).detach().cpu())

    out = torch.stack(samples, dim=0)
    stats = {
        "mean_accept_rate": float(sum(accept_rates) / len(accept_rates)),
        "mean_abs_delta_H": float(sum(delta_Hs) / len(delta_Hs)),
    }
    return out, stats


# ============================================================
# Quick diagnostics
# ============================================================


@torch.no_grad()
def compare_single_trajectory(
    model: TrajectoryPINN,
    target: BananaTarget,
    q0: torch.Tensor,
    p0: torch.Tensor,
    step_size: float,
    n_steps: int,
) -> Dict[str, float]:
    q_true, p_true, traj = leapfrog(
        target, q0.unsqueeze(0), p0.unsqueeze(0), step_size, n_steps, return_trajectory=True
    )
    assert traj is not None

    times = traj["times"]
    q_ref = traj["q"].squeeze(0)
    p_ref = traj["p"].squeeze(0)

    q0_rep = q0.unsqueeze(0).expand(times.shape[0], -1)
    p0_rep = p0.unsqueeze(0).expand(times.shape[0], -1)
    q_hat, p_hat = model(q0_rep, p0_rep, times.unsqueeze(-1))

    mse_q = torch.mean((q_hat - q_ref) ** 2).item()
    mse_p = torch.mean((p_hat - p_ref) ** 2).item()

    H_ref = target.H(q_ref, p_ref)
    H_hat = target.H(q_hat, p_hat)
    drift_ref = (H_ref - H_ref[0]).abs().mean().item()
    drift_hat = (H_hat - H_hat[0]).abs().mean().item()

    return {
        "mse_q": float(mse_q),
        "mse_p": float(mse_p),
        "mean_H_drift_ref": float(drift_ref),
        "mean_H_drift_hat": float(drift_hat),
    }


# ============================================================
# Main experiment
# ============================================================


def main() -> None:
    device = get_device()
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    torch.manual_seed(42)

    print(f"Using device: {device}")

    target = BananaTarget(b=0.15, sigma1=1.0, sigma2=1.0)

    data_cfg = DatasetConfig(
        n_trajectories=1800,
        n_time_points=21,
        step_size=0.08,
        n_leapfrog_steps=20,
        q0_std=1.8,
        train_split=0.9,
        batch_size=256,
    )

    train_cfg = TrainConfig(
        epochs=120,
        lr=1e-3,
        weight_decay=1e-6,
        hidden_dim=128,
        depth=4,
        lambda_sup=1.0,
        lambda_phys=0.1,
        lambda_ic=1.0,
        grad_clip=1.0,
        verbose_every=10,
    )

    t0 = time.time()
    dataset = generate_teacher_dataset(target, data_cfg, device=device)
    print(
        "Generated dataset:",
        dataset["train_x"].shape,
        dataset["train_y"].shape,
        dataset["val_x"].shape,
        dataset["val_y"].shape,
    )

    model, history = train_pinn(target, dataset, data_cfg, train_cfg, device=device)
    print(f"Training finished in {time.time() - t0:.2f} seconds")

    # Single-trajectory diagnostic
    q0 = torch.tensor([0.8, -0.5], device=device)
    p0 = torch.tensor([0.3, 1.0], device=device)
    diag = compare_single_trajectory(
        model, target, q0, p0, data_cfg.step_size, data_cfg.n_leapfrog_steps
    )
    print("Single-trajectory diagnostic:")
    for k, v in diag.items():
        print(f"  {k}: {v:.6e}")

    # Baseline HMC
    baseline_samples, baseline_stats = run_hmc_baseline(
        target=target,
        n_samples=2000,
        burn_in=500,
        step_size=data_cfg.step_size,
        n_steps=data_cfg.n_leapfrog_steps,
        device=device,
    )
    print("Baseline HMC stats:")
    for k, v in baseline_stats.items():
        print(f"  {k}: {v:.6f}")

    # PINN-HMC
    t_final = data_cfg.step_size * data_cfg.n_leapfrog_steps
    pinn_samples, pinn_stats = run_hmc_pinn(
        model=model,
        target=target,
        n_samples=2000,
        burn_in=500,
        t_final=t_final,
        device=device,
    )
    print("PINN-HMC stats:")
    for k, v in pinn_stats.items():
        print(f"  {k}: {v:.6f}")

    # Sample moments for a fast sanity check
    print("Baseline sample mean:", baseline_samples.mean(dim=0))
    print("Baseline sample std :", baseline_samples.std(dim=0))
    print("PINN sample mean    :", pinn_samples.mean(dim=0))
    print("PINN sample std     :", pinn_samples.std(dim=0))

    # Save checkpoint
    save_dict = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "data_cfg": data_cfg.__dict__,
        "train_cfg": train_cfg.__dict__,
        "target_cfg": {
            "b": target.b,
            "sigma1": target.sigma1,
            "sigma2": target.sigma2,
        },
    }
    torch.save(save_dict, "pinn_hmc_mvp_checkpoint.pt")
    print("Saved checkpoint to pinn_hmc_mvp_checkpoint.pt")


if __name__ == "__main__":
    main()
