from __future__ import annotations

from dataclasses import dataclass

import torch

from .balancers import ProgressBalancer
from .losses import compute_hnn_raw_losses
from .model import HNNConfig, HamiltonianNN, ModelConfig, TrajectoryPINN
from .run_experiment import (
    DatasetConfig,
    SamplerEvalConfig,
    compare_single_trajectory,
    generate_teacher_dataset,
    get_device,
    run_hmc_baseline,
    run_hmc_pinn,
)
from .target import BananaTarget, BananaTargetConfig
from .train import TrainConfig, train_hnn_model, train_model


@torch.no_grad()
def generate_hnn_dataset(
    target: BananaTarget,
    config: DatasetConfig,
    device: torch.device,
):
    """
    Build HNN training data from trajectory states:
        x = [q, p]
    """
    from .run_experiment import leapfrog

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

    traj_q = traj["q"]   # [n_traj, T, dim]
    traj_p = traj["p"]   # [n_traj, T, dim]

    states = torch.cat([traj_q, traj_p], dim=-1).reshape(-1, 2 * target.dim)

    n_total = states.shape[0]
    n_train = int(config.train_split * n_total)
    perm = torch.randperm(n_total, device=device)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    return {
        "train_x": states[train_idx],
        "val_x": states[val_idx],
    }


def hnn_vector_field(
    model: HamiltonianNN,
    q: torch.Tensor,
    p: torch.Tensor,
):
    """
    Compute learned Hamiltonian vector field:
        dq/dt = dH/dp
        dp/dt = -dH/dq

    IMPORTANT:
    This function must enable gradients, because we differentiate H_theta
    w.r.t. its inputs (q, p).
    """
    with torch.enable_grad():
        q_req = q.detach().clone().requires_grad_(True)
        p_req = p.detach().clone().requires_grad_(True)

        H_hat = model(q_req, p_req)

        dH_dq = torch.autograd.grad(
            H_hat.sum(),
            q_req,
            create_graph=False,
            retain_graph=True,
        )[0]

        dH_dp = torch.autograd.grad(
            H_hat.sum(),
            p_req,
            create_graph=False,
            retain_graph=False,
        )[0]

    dqdt = dH_dp.detach()
    dpdt = (-dH_dq).detach()
    return dqdt, dpdt


def leapfrog_hnn(
    model: HamiltonianNN,
    q0: torch.Tensor,
    p0: torch.Tensor,
    step_size: float,
    n_steps: int,
):
    """
    Leapfrog-style integration using learned Hamiltonian gradients.
    """
    q = q0.clone()
    p = p0.clone()

    _, dpdt0 = hnn_vector_field(model, q, p)
    p = p + 0.5 * step_size * dpdt0

    for i in range(n_steps):
        dqdt, _ = hnn_vector_field(model, q, p)
        q = q + step_size * dqdt

        if i != n_steps - 1:
            _, dpdt = hnn_vector_field(model, q, p)
            p = p + step_size * dpdt

    _, dpdt_last = hnn_vector_field(model, q, p)
    p = p + 0.5 * step_size * dpdt_last
    p = -p

    return q, p


def hmc_step_hnn(
    model: HamiltonianNN,
    target: BananaTarget,
    q_current: torch.Tensor,
    step_size: float,
    n_steps: int,
):
    batch_size = q_current.shape[0]
    p0 = target.sample_momentum(batch_size, q_current.device)

    q_prop, p_prop = leapfrog_hnn(
        model=model,
        q0=q_current,
        p0=p0,
        step_size=step_size,
        n_steps=n_steps,
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


def run_hmc_hnn(
    model: HamiltonianNN,
    target: BananaTarget,
    n_samples: int,
    burn_in: int,
    step_size: float,
    n_steps: int,
    device: torch.device,
):
    q = torch.zeros(1, target.dim, device=device)

    samples = []
    accept_rates = []
    delta_Hs = []

    total_steps = burn_in + n_samples
    for i in range(total_steps):
        q, _, _, info = hmc_step_hnn(
            model=model,
            target=target,
            q_current=q,
            step_size=step_size,
            n_steps=n_steps,
        )
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


def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Using device: {device}")

    target = BananaTarget(BananaTargetConfig(b=0.15, sigma1=1.0, sigma2=1.0))
    data_cfg = DatasetConfig()
    sampler_cfg = SamplerEvalConfig(n_samples=2000, burn_in=500)

    # --------------------
    # PINN settings
    # --------------------
    pinn_model_cfg = ModelConfig(dim=2, hidden_dim=128, depth=4)
    pinn_train_cfg = TrainConfig(
        epochs=120,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-6,
        grad_clip=1.0,
        verbose_every=10,
    )
    pinn_balancer = ProgressBalancer(alpha=1.0)

    # --------------------
    # HNN settings
    # --------------------
    hnn_model_cfg = HNNConfig(dim=2, hidden_dim=128, depth=4)
    hnn_train_cfg = TrainConfig(
        epochs=120,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-6,
        grad_clip=1.0,
        verbose_every=10,
    )

    # --------------------
    # datasets
    # --------------------
    pinn_dataset = generate_teacher_dataset(target, data_cfg, device=device)
    hnn_dataset = generate_hnn_dataset(target, data_cfg, device=device)

    print("PINN dataset:")
    print(
        pinn_dataset["train_x"].shape,
        pinn_dataset["train_y"].shape,
        pinn_dataset["val_x"].shape,
        pinn_dataset["val_y"].shape,
    )
    print("HNN dataset:")
    print(
        hnn_dataset["train_x"].shape,
        hnn_dataset["val_x"].shape,
    )

    # --------------------
    # train PINN
    # --------------------
    pinn_model = TrajectoryPINN(pinn_model_cfg).to(device)
    pinn_model, pinn_history = train_model(
        model=pinn_model,
        target=target,
        train_x=pinn_dataset["train_x"],
        train_y=pinn_dataset["train_y"],
        val_x=pinn_dataset["val_x"],
        val_y=pinn_dataset["val_y"],
        train_cfg=pinn_train_cfg,
        balancer=pinn_balancer,
    )

    # --------------------
    # train HNN
    # --------------------
    hnn_model = HamiltonianNN(hnn_model_cfg).to(device)
    hnn_model, hnn_history = train_hnn_model(
        model=hnn_model,
        target=target,
        train_x=hnn_dataset["train_x"],
        val_x=hnn_dataset["val_x"],
        train_cfg=hnn_train_cfg,
    )

    # --------------------
    # diagnostics
    # --------------------
    q0 = torch.tensor([0.8, -0.5], device=device)
    p0 = torch.tensor([0.3, 1.0], device=device)

    pinn_diag = compare_single_trajectory(
        model=pinn_model,
        target=target,
        q0=q0,
        p0=p0,
        step_size=data_cfg.step_size,
        n_steps=data_cfg.n_leapfrog_steps,
    )

    hnn_state = torch.cat([q0.unsqueeze(0), p0.unsqueeze(0)], dim=-1)
    with torch.enable_grad():
        raw_hnn = compute_hnn_raw_losses(hnn_model, target, hnn_state)

    hnn_diag = {
        "hnn_q_loss": float(raw_hnn["q_sup"].item()),
        "hnn_p_loss": float(raw_hnn["p_sup"].item()),
        "hnn_total_loss": float(raw_hnn["sup"].item()),
    }

    print("PINN trajectory diagnostic:", pinn_diag)
    print("HNN state diagnostic:", hnn_diag)

    # --------------------
    # samplers
    # --------------------
    baseline_samples, baseline_stats = run_hmc_baseline(
        target=target,
        n_samples=sampler_cfg.n_samples,
        burn_in=sampler_cfg.burn_in,
        step_size=data_cfg.step_size,
        n_steps=data_cfg.n_leapfrog_steps,
        device=device,
    )

    t_final = data_cfg.step_size * data_cfg.n_leapfrog_steps
    pinn_samples, pinn_stats = run_hmc_pinn(
        model=pinn_model,
        target=target,
        n_samples=sampler_cfg.n_samples,
        burn_in=sampler_cfg.burn_in,
        t_final=t_final,
        device=device,
    )

    hnn_samples, hnn_stats = run_hmc_hnn(
        model=hnn_model,
        target=target,
        n_samples=sampler_cfg.n_samples,
        burn_in=sampler_cfg.burn_in,
        step_size=data_cfg.step_size,
        n_steps=data_cfg.n_leapfrog_steps,
        device=device,
    )

    print("Baseline HMC stats:", baseline_stats)
    print("PINN-HMC stats:", pinn_stats)
    print("HNNMC-style stats:", hnn_stats)

    # --------------------
    # sample stats
    # --------------------
    baseline_mean = baseline_samples.mean(dim=0)
    baseline_std = baseline_samples.std(dim=0)

    pinn_mean = pinn_samples.mean(dim=0)
    pinn_std = pinn_samples.std(dim=0)

    hnn_mean = hnn_samples.mean(dim=0)
    hnn_std = hnn_samples.std(dim=0)

    print("Baseline mean/std:", baseline_mean, baseline_std)
    print("PINN mean/std:", pinn_mean, pinn_std)
    print("HNN mean/std:", hnn_mean, hnn_std)

    save_path = "checkpoint_compare_pinn_hnn.pt"
    torch.save(
        {
            "baseline_samples": baseline_samples,
            "pinn_samples": pinn_samples,
            "hnn_samples": hnn_samples,
            "baseline_stats": baseline_stats,
            "pinn_stats": pinn_stats,
            "hnn_stats": hnn_stats,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "pinn_mean": pinn_mean,
            "pinn_std": pinn_std,
            "hnn_mean": hnn_mean,
            "hnn_std": hnn_std,
            "pinn_diag": pinn_diag,
            "hnn_diag": hnn_diag,
            "pinn_history": pinn_history,
            "hnn_history": hnn_history,
        },
        save_path,
    )
    print(f"Saved comparison checkpoint to {save_path}")


if __name__ == "__main__":
    main()