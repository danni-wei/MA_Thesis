from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from .balancers import (
    EMANormalizer,
    FixedBalancer,
    GradNormBalancer,
    ProgressBalancer,
)
from .model import ModelConfig, TrajectoryPINN
from .target import BananaTarget, BananaTargetConfig, get_device
from .train import TrainConfig, train_model


@dataclass
class DatasetConfig:
    n_trajectories: int = 1800
    n_time_points: int = 21
    step_size: float = 0.08
    n_leapfrog_steps: int = 20
    q0_std: float = 1.8
    train_split: float = 0.9


@dataclass
class SamplerEvalConfig:
    n_samples: int = 2000
    burn_in: int = 500


def leapfrog(
    target: BananaTarget,
    q0: torch.Tensor,
    p0: torch.Tensor,
    step_size: float,
    n_steps: int,
    return_trajectory: bool = False,
):
    if q0.shape != p0.shape:
        raise ValueError(f"q0 and p0 must have same shape, got {q0.shape} vs {p0.shape}")

    q = q0.clone()
    p = p0.clone()

    traj_q = []
    traj_p = []

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
            if i != n_steps - 1:
                p_store = p.clone()
            else:
                p_store = (p - 0.5 * step_size * target.grad_U(q)).clone()
            traj_p.append(p_store)

    p = p - 0.5 * step_size * target.grad_U(q)
    p = -p

    if return_trajectory:
        times = torch.linspace(
            0.0,
            n_steps * step_size,
            n_steps + 1,
            device=q0.device,
            dtype=q0.dtype,
        )
        return q, p, {
            "times": times,
            "q": torch.stack(traj_q, dim=1),
            "p": torch.stack(traj_p, dim=1),
        }

    return q, p, None


@torch.no_grad()
def generate_teacher_dataset(
    target: BananaTarget,
    config: DatasetConfig,
    device: torch.device,
):
    q0 = config.q0_std * torch.randn(config.n_trajectories, target.dim, device=device)
    p0 = target.sample_momentum(config.n_trajectories, device)

    _, _, traj = leapfrog(
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

    total_points = traj_q.shape[1]
    if config.n_time_points > total_points:
        raise ValueError("n_time_points cannot exceed stored trajectory points")

    idx = torch.linspace(0, total_points - 1, config.n_time_points, device=device).long()

    times = times[idx]
    traj_q = traj_q[:, idx, :]
    traj_p = traj_p[:, idx, :]

    n_traj, n_t, dim = traj_q.shape
    q0_rep = q0[:, None, :].expand(n_traj, n_t, dim)
    p0_rep = p0[:, None, :].expand(n_traj, n_t, dim)
    t_rep = times[None, :, None].expand(n_traj, n_t, 1)

    inputs = torch.cat([q0_rep, p0_rep, t_rep], dim=-1).reshape(-1, 2 * dim + 1)
    outputs = torch.cat([traj_q, traj_p], dim=-1).reshape(-1, 2 * dim)

    n_total = inputs.shape[0]
    n_train = int(config.train_split * n_total)

    perm = torch.randperm(n_total, device=device)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    return {
        "train_x": inputs[train_idx],
        "train_y": outputs[train_idx],
        "val_x": inputs[val_idx],
        "val_y": outputs[val_idx],
    }


def build_balancer(method: str):
    method = method.lower()

    if method == "fixed":
        return FixedBalancer(lambda_sup=1.0, lambda_phys=0.1, lambda_ic=1.0)
    if method == "ema":
        return EMANormalizer(beta=0.98)
    if method == "progress":
        return ProgressBalancer(alpha=1.0)
    if method == "gradnorm":
        return GradNormBalancer(eta=0.1)

    raise ValueError(f"Unknown method: {method}")


@torch.no_grad()
def hmc_step_baseline(
    target: BananaTarget,
    q_current: torch.Tensor,
    step_size: float,
    n_steps: int,
):
    batch_size = q_current.shape[0]
    p0 = target.sample_momentum(batch_size, q_current.device)
    q_prop, p_prop, _ = leapfrog(
        target=target,
        q0=q_current,
        p0=p0,
        step_size=step_size,
        n_steps=n_steps,
        return_trajectory=False,
    )

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
):
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


@torch.no_grad()
def pinn_proposal(
    model: TrajectoryPINN,
    q_current: torch.Tensor,
    p0: torch.Tensor,
    t_final: float,
):
    t = torch.full(
        (q_current.shape[0], 1),
        float(t_final),
        device=q_current.device,
        dtype=q_current.dtype,
    )
    q_prop, p_prop = model(q_current, p0, t)
    p_prop = -p_prop
    return q_prop, p_prop


@torch.no_grad()
def hmc_step_pinn(
    model: TrajectoryPINN,
    target: BananaTarget,
    q_current: torch.Tensor,
    t_final: float,
):
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
):
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


@torch.no_grad()
def compare_single_trajectory(
    model: TrajectoryPINN,
    target: BananaTarget,
    q0: torch.Tensor,
    p0: torch.Tensor,
    step_size: float,
    n_steps: int,
):
    _, _, traj = leapfrog(
        target=target,
        q0=q0.unsqueeze(0),
        p0=p0.unsqueeze(0),
        step_size=step_size,
        n_steps=n_steps,
        return_trajectory=True,
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


def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    # -----------------------------
    # configs
    # -----------------------------
    target = BananaTarget(BananaTargetConfig(b=0.15, sigma1=1.0, sigma2=1.0))
    data_cfg = DatasetConfig()
    model_cfg = ModelConfig(dim=2, hidden_dim=128, depth=4)
    train_cfg = TrainConfig(
        epochs=120,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-6,
        grad_clip=1.0,
        verbose_every=10,
    )
    sampler_cfg = SamplerEvalConfig(n_samples=2000, burn_in=500)

    # -----------------------------
    # dataset
    # -----------------------------
    dataset = generate_teacher_dataset(target, data_cfg, device=device)
    print(
        "Generated dataset:",
        dataset["train_x"].shape,
        dataset["train_y"].shape,
        dataset["val_x"].shape,
        dataset["val_y"].shape,
    )

    # choose from: fixed / ema / progress / gradnorm
    method = "gradnorm"
    balancer = build_balancer(method)

    # -----------------------------
    # train
    # -----------------------------
    model = TrajectoryPINN(model_cfg).to(device)

    model, history = train_model(
        model=model,
        target=target,
        train_x=dataset["train_x"],
        train_y=dataset["train_y"],
        val_x=dataset["val_x"],
        val_y=dataset["val_y"],
        train_cfg=train_cfg,
        balancer=balancer,
    )

    # -----------------------------
    # single trajectory diagnostic
    # -----------------------------
    q0 = torch.tensor([0.8, -0.5], device=device)
    p0 = torch.tensor([0.3, 1.0], device=device)

    diag = compare_single_trajectory(
        model=model,
        target=target,
        q0=q0,
        p0=p0,
        step_size=data_cfg.step_size,
        n_steps=data_cfg.n_leapfrog_steps,
    )
    print("Single-trajectory diagnostic:")
    for k, v in diag.items():
        print(f"  {k}: {v:.6e}")

    # -----------------------------
    # baseline HMC evaluation
    # -----------------------------
    baseline_samples, baseline_stats = run_hmc_baseline(
        target=target,
        n_samples=sampler_cfg.n_samples,
        burn_in=sampler_cfg.burn_in,
        step_size=data_cfg.step_size,
        n_steps=data_cfg.n_leapfrog_steps,
        device=device,
    )
    print("Baseline HMC stats:")
    for k, v in baseline_stats.items():
        print(f"  {k}: {v:.6f}")

    # -----------------------------
    # PINN-HMC evaluation
    # -----------------------------
    t_final = data_cfg.step_size * data_cfg.n_leapfrog_steps
    pinn_samples, pinn_stats = run_hmc_pinn(
        model=model,
        target=target,
        n_samples=sampler_cfg.n_samples,
        burn_in=sampler_cfg.burn_in,
        t_final=t_final,
        device=device,
    )
    print("PINN-HMC stats:")
    for k, v in pinn_stats.items():
        print(f"  {k}: {v:.6f}")

    # -----------------------------
    # sample moments
    # -----------------------------
    baseline_mean = baseline_samples.mean(dim=0)
    baseline_std = baseline_samples.std(dim=0)
    pinn_mean = pinn_samples.mean(dim=0)
    pinn_std = pinn_samples.std(dim=0)

    print("Baseline sample mean:", baseline_mean)
    print("Baseline sample std :", baseline_std)
    print("PINN sample mean    :", pinn_mean)
    print("PINN sample std     :", pinn_std)

    # -----------------------------
    # save checkpoint
    # -----------------------------
    save_path = f"checkpoint_{method}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "method": method,
            "data_cfg": data_cfg.__dict__,
            "model_cfg": model_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
            "sampler_cfg": sampler_cfg.__dict__,
            "trajectory_diag": diag,
            "baseline_stats": baseline_stats,
            "pinn_stats": pinn_stats,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "pinn_mean": pinn_mean,
            "pinn_std": pinn_std,
            "baseline_samples": baseline_samples,
            "pinn_samples": pinn_samples,
        },
        save_path,
    )
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()