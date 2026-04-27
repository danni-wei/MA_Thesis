"""
experiment_sus.py
─────────────────
End-to-end Subset Simulation experiment comparing
    Baseline HMC | PINN-HMC | HNN-HMC
on the WangBananaTarget with EllipticalLimitState.

Run from the project root:
    python -m pinn_hmc.experiment_sus
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .balancers import ProgressBalancer
from .compare_pinn_hnn import generate_hnn_dataset, leapfrog_hnn
from .model import HNNConfig, HamiltonianNN, ModelConfig, TrajectoryPINN
from .run_experiment import DatasetConfig, generate_teacher_dataset
from .sus import run_subset_simulation
from .target import EllipticalLimitState, WangBananaTarget, get_device
from .train import TrainConfig, train_hnn_model, train_model

# ── Experiment-wide constants ─────────────────────────────────────────────────
STEP_SIZE = 0.08
N_STEPS = 20
T_FINAL = STEP_SIZE * N_STEPS      # 1.6 s

SUS_N = 500
SUS_P0 = 0.1
SUS_MAX_LEVELS = 20
SUS_BURN_IN = 200

R_VALUES: List[float] = [10.0, 12.0]

OUT_DIR = Path("results/sus_experiment")

METHODS = ["baseline", "pinn", "hnn"]
LABELS = {"baseline": "Baseline HMC", "pinn": "PINN-HMC", "hnn": "HNN-HMC"}
COLORS = {"baseline": "steelblue", "pinn": "darkorange", "hnn": "forestgreen"}


# ── Proposal function factories ───────────────────────────────────────────────

def make_baseline_proposal(
    target: WangBananaTarget,
    step_size: float = STEP_SIZE,
    n_steps: int = N_STEPS,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Exact leapfrog + Metropolis accept/reject on the true target."""

    def proposal_fn(q: torch.Tensor) -> torch.Tensor:
        q = q.detach()
        N = q.shape[0]
        p = target.sample_momentum(N, q.device)

        q_p = q.clone()
        p_p = (p - 0.5 * step_size * target.grad_U(q_p)).detach()
        for i in range(n_steps):
            q_p = (q_p + step_size * p_p).detach()
            if i < n_steps - 1:
                p_p = (p_p - step_size * target.grad_U(q_p)).detach()
        p_p = (p_p - 0.5 * step_size * target.grad_U(q_p)).detach()

        with torch.no_grad():
            log_alpha = torch.clamp(target.H(q, p) - target.H(q_p, p_p), max=80.0)
            accept = (
                torch.rand(N, device=q.device) < torch.exp(log_alpha).clamp(max=1.0)
            ).unsqueeze(-1)
        return torch.where(accept, q_p, q)

    return proposal_fn


def make_pinn_proposal(
    model: TrajectoryPINN,
    target: WangBananaTarget,
    t_final: float = T_FINAL,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Single PINN forward pass + Metropolis accept/reject."""
    model.eval()

    def proposal_fn(q: torch.Tensor) -> torch.Tensor:
        q = q.detach()
        N = q.shape[0]
        p0 = target.sample_momentum(N, q.device)
        t = torch.full((N, 1), float(t_final), device=q.device, dtype=q.dtype)

        with torch.no_grad():
            q_prop, p_prop = model(q, p0, t)
            p_prop = -p_prop
            log_alpha = torch.clamp(target.H(q, p0) - target.H(q_prop, p_prop), max=80.0)
            accept = (
                torch.rand(N, device=q.device) < torch.exp(log_alpha).clamp(max=1.0)
            ).unsqueeze(-1)
        return torch.where(accept, q_prop, q)

    return proposal_fn


def make_hnn_proposal(
    model: HamiltonianNN,
    target: WangBananaTarget,
    step_size: float = STEP_SIZE,
    n_steps: int = N_STEPS,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """HNN-driven leapfrog + Metropolis accept/reject on the true target."""
    model.eval()

    def proposal_fn(q: torch.Tensor) -> torch.Tensor:
        q = q.detach()
        N = q.shape[0]
        p0 = target.sample_momentum(N, q.device)

        # leapfrog_hnn uses enable_grad internally; returns detached tensors
        q_prop, p_prop = leapfrog_hnn(model, q, p0, step_size, n_steps)

        with torch.no_grad():
            log_alpha = torch.clamp(target.H(q, p0) - target.H(q_prop, p_prop), max=80.0)
            accept = (
                torch.rand(N, device=q.device) < torch.exp(log_alpha).clamp(max=1.0)
            ).unsqueeze(-1)
        return torch.where(accept, q_prop, q)

    return proposal_fn


# ── Plotting ──────────────────────────────────────────────────────────────────

def _ellipse_xy(ls: EllipticalLimitState, r: float, n: int = 400):
    """Parametric ellipse boundary where g(x,y;r) = 0."""
    angles = np.linspace(0, 2 * np.pi, n)
    z1 = r * ls.c1 * np.cos(angles)
    z2 = r * ls.c2 * np.sin(angles)
    # Inverse rotation: x = z1*cos - z2*sin, y = z1*sin + z2*cos … wait:
    # z1 = x*cos - y*sin, z2 = x*sin + y*cos  →  x = z1*cos + z2*sin, y = -z1*sin + z2*cos
    cos_t, sin_t = ls._cos_t, ls._sin_t
    return z1 * cos_t + z2 * sin_t, -z1 * sin_t + z2 * cos_t


def plot_scatter_samples(
    results_r: Dict,
    ls: EllipticalLimitState,
    r: float,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"SuS samples per level  (r = {r})", fontsize=13)

    cmap = plt.cm.plasma
    ex, ey = _ellipse_xy(ls, r)

    for ax, method in zip(axes, METHODS):
        all_samples = results_r[method]["all_samples"]
        n_lvl = len(all_samples)
        for lvl, s in enumerate(all_samples):
            c = cmap(lvl / max(n_lvl - 1, 1))
            ax.scatter(s[:, 0].numpy(), s[:, 1].numpy(),
                       s=4, alpha=0.5, color=c, label=f"L{lvl}")
        ax.plot(ex, ey, "r-", lw=1.5, label="Failure boundary")
        ax.set_title(LABELS[method])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(fontsize=6, loc="upper right", markerscale=2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_pf_comparison(results: Dict, r_values: List[float], out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))

    n_m = len(METHODS)
    n_r = len(r_values)
    bw = 0.6 / n_m

    for i, method in enumerate(METHODS):
        xs = [j + (i - n_m / 2 + 0.5) * bw for j in range(n_r)]
        pfs = [max(results[r][method]["P_F"], 1e-20) for r in r_values]
        ax.bar(xs, pfs, width=bw * 0.9, color=COLORS[method], label=LABELS[method])

    ax.set_xticks(range(n_r))
    ax.set_xticklabels([f"r = {r:.0f}" for r in r_values])
    ax.set_yscale("log")
    ax.set_ylabel("Estimated $P_F$")
    ax.set_title("Failure probability comparison")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_cost_comparison(
    results: Dict,
    r: float,
    t_train: Dict[str, float],
    out_path: Path,
):
    metrics = [
        ("training_time",       "Training (s)"),
        ("mean_proposal_time",  "Per-proposal (s)"),
        ("total_sampling_time", "Total sampling (s)"),
    ]
    n_met = len(metrics)
    n_m = len(METHODS)
    bw = 0.7 / n_met

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (key, label) in enumerate(metrics):
        xs = [i + (j - n_met / 2 + 0.5) * bw for i in range(n_m)]
        vals = []
        for method in METHODS:
            if key == "training_time":
                vals.append(t_train.get(method, 0.0))
            else:
                vals.append(results[r][method].get(key, 0.0))
        ax.bar(xs, vals, width=bw * 0.9, label=label)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"Computational cost  (r = {r:.0f})")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_training_loss(
    pinn_history: List[Dict],
    hnn_history: List[Dict],
    out_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PINN
    ep = [h["epoch"] for h in pinn_history]
    axes[0].plot(ep, [h["train_total_loss"] for h in pinn_history], label="Train")
    axes[0].plot(ep, [h["val_total_loss"]   for h in pinn_history], "--", label="Val")
    axes[0].set(title="PINN loss (ProgressBalancer)", xlabel="Epoch", ylabel="Total loss")
    axes[0].set_yscale("log")
    axes[0].legend()

    # HNN
    ep = [h["epoch"] for h in hnn_history]
    axes[1].plot(ep, [h["train_sup"] for h in hnn_history], label="Train")
    axes[1].plot(ep, [h["val_sup"]   for h in hnn_history], "--", label="Val")
    axes[1].set(title="HNN loss", xlabel="Epoch", ylabel="Supervised loss")
    axes[1].set_yscale("log")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(
    results: Dict,
    r_values: List[float],
    t_train: Dict[str, float],
):
    hdr = (f"{'Method':<16} {'r':>6} {'P_F':>12} {'Lvls':>5} "
           f"{'Train(s)':>10} {'Prop(ms)':>10} {'Samp(s)':>10}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)
    for r in r_values:
        for method in METHODS:
            res = results[r][method]
            print(
                f"{LABELS[method]:<16} {r:>6.0f} "
                f"{res['P_F']:>12.4e} {res['levels']:>5} "
                f"{t_train.get(method, 0.0):>10.2f} "
                f"{res['mean_proposal_time'] * 1e3:>10.3f} "
                f"{res['total_sampling_time']:>10.2f}"
            )
    print(sep)


# ── Report generation ─────────────────────────────────────────────────────────

def _md_table_rows(results: Dict, r_values: List[float], t_train: Dict) -> str:
    rows = []
    for r in r_values:
        for method in METHODS:
            res = results[r][method]
            rows.append(
                f"| {LABELS[method]} | {r:.0f} | {res['P_F']:.4e} "
                f"| {res['levels']} | {t_train.get(method, 0.0):.2f} "
                f"| {res['mean_proposal_time'] * 1e3:.3f} "
                f"| {res['total_sampling_time']:.2f} |"
            )
    return "\n".join(rows)


def generate_report(
    results: Dict,
    r_values: List[float],
    t_train: Dict[str, float],
    out_path: Path,
):
    table = _md_table_rows(results, r_values, t_train)

    md = f"""\
# Subset Simulation Experiment: Baseline HMC vs PINN-HMC vs HNN-HMC

## 1. Background

Structural reliability analysis requires estimating failure probabilities
$P_F = P(g(\\mathbf{{X}}) \\leq 0)$ that can be extremely small ($10^{{-5}}$–$10^{{-10}}$).
Direct Monte Carlo is unaffordable at these scales; **Subset Simulation** (SuS,
Au & Beck 2001; Wang et al. 2019) decomposes the rare event into a product of
more probable conditional events, each estimated by a Markov chain.

The quality of the Markov-chain transitions drives both the accuracy and the
computational cost of SuS. This experiment benchmarks three transition kernels:

* **Baseline HMC** — leapfrog with true gradients $\\nabla_q U$. Accurate but
  requires one gradient evaluation per leapfrog step ($L$ per proposal).
* **PINN-HMC** — a Physics-Informed Neural Network (PINN) trained to predict
  the trajectory endpoint $(q_T, p_T)$ given $(q_0, p_0, T)$, replacing the
  entire integrator with a single forward pass.  Upfront training cost is
  amortised over many SuS proposals.
* **HNN-HMC** — a Hamiltonian Neural Network (HNN) that learns the Hamiltonian
  $H_\\theta(q, p)$; leapfrog is run with the *learned* vector field, again
  trading training cost for cheaper per-step evaluation.

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Target | `WangBananaTarget` (Wang 2019 eq. 34): $a=1.15$, $b=0.5$, $\\rho=0.9$ |
| Limit-state | `EllipticalLimitState` (Wang 2019 eq. 40): $c_1=1$, $c_2=0.5$, $\\theta=\\pi/4$ |
| Radii $r$ | {', '.join(str(int(r)) for r in r_values)} |
| Model arch. | 4-layer MLP, 128 hidden, Tanh (both PINN and HNN) |
| PINN training | 120 epochs, Adam $10^{{-3}}$, ProgressBalancer $\\alpha=1$ |
| HNN training | 120 epochs, Adam $10^{{-3}}$ |
| Dataset | 1 800 trajectories, step size {STEP_SIZE}, {N_STEPS} leapfrog steps |
| SuS $N$ | {SUS_N} samples/level |
| SuS $p_0$ | {SUS_P0} |
| SuS max levels | {SUS_MAX_LEVELS} |
| SuS burn-in | {SUS_BURN_IN} |

## 3. Results

| Method | r | P_F | Levels | Train (s) | Prop (ms) | Samp (s) |
|--------|---|-----|--------|-----------|-----------|----------|
{table}

## 4. Analysis

[TO BE FILLED] — Discuss whether P_F estimates agree across methods.
Explain the trade-off between training time and per-proposal speed for PINN
and HNN.  Note which method delivers the best total-cost P_F estimate and
why.  Connect to the earlier banana-target HMC experiment: did PINN/HNN
generalise well to the rare-event tails explored by deeper SuS levels?

## 5. Limitations and Future Work

* **Distribution shift in rare-event tails**: PINN and HNN are trained on
  trajectory data from the prior.  Deeper SuS levels condition on ever more
  extreme samples; the learned model may degrade there.
* **Single training run**: models are trained once and reused across all SuS
  levels.  Level-adaptive retraining (RS-HMC-SS, Wang 2019) could improve
  accuracy at low $P_F$.
* **No $P_F$ uncertainty**: only one SuS run per method is reported.  Multiple
  independent runs would yield confidence intervals.
* **Small $N$**: $N = {SUS_N}$ gives a coefficient of variation of
  $\\approx \\sqrt{{(1-p_0)/(p_0 N)}} \\approx {np.sqrt((1-SUS_P0)/(SUS_P0*SUS_N)):.2f}$
  per level; larger $N$ would reduce variance.
* **Fixed hyper-parameters**: step size and number of leapfrog steps were not
  tuned for the WangBananaTarget; a tuning phase would likely improve all
  three methods.
"""

    out_path.write_text(md, encoding="utf-8")
    print(f"  Saved {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = get_device()
    print(f"Device: {device}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target = WangBananaTarget()
    ls = EllipticalLimitState()

    # ── Datasets ───────────────────────────────────────────────────────────────
    print("\n[1/5] Generating training datasets ...")
    data_cfg = DatasetConfig(
        n_trajectories=1800,
        step_size=STEP_SIZE,
        n_leapfrog_steps=N_STEPS,
    )
    pinn_dataset = generate_teacher_dataset(target, data_cfg, device)
    hnn_dataset  = generate_hnn_dataset(target, data_cfg, device)
    print(f"  PINN train: {tuple(pinn_dataset['train_x'].shape)}")
    print(f"  HNN  train: {tuple(hnn_dataset['train_x'].shape)}")

    # ── Train PINN ─────────────────────────────────────────────────────────────
    print("\n[2/5] Training PINN (ProgressBalancer, 120 epochs) ...")
    pinn_train_cfg = TrainConfig(
        epochs=120, batch_size=256, lr=1e-3,
        weight_decay=1e-6, grad_clip=1.0, verbose_every=20,
    )
    pinn_model = TrajectoryPINN(ModelConfig(dim=2, hidden_dim=128, depth=4)).to(device)
    t0 = time.perf_counter()
    pinn_model, pinn_history = train_model(
        pinn_model, target,
        pinn_dataset["train_x"], pinn_dataset["train_y"],
        pinn_dataset["val_x"],   pinn_dataset["val_y"],
        pinn_train_cfg, ProgressBalancer(alpha=1.0),
    )
    t_pinn = time.perf_counter() - t0
    print(f"  Training time: {t_pinn:.1f}s")

    # ── Train HNN ──────────────────────────────────────────────────────────────
    print("\n[3/5] Training HNN (120 epochs) ...")
    hnn_train_cfg = TrainConfig(
        epochs=120, batch_size=256, lr=1e-3,
        weight_decay=1e-6, grad_clip=1.0, verbose_every=20,
    )
    hnn_model = HamiltonianNN(HNNConfig(dim=2, hidden_dim=128, depth=4)).to(device)
    t0 = time.perf_counter()
    hnn_model, hnn_history = train_hnn_model(
        hnn_model, target,
        hnn_dataset["train_x"], hnn_dataset["val_x"],
        hnn_train_cfg,
    )
    t_hnn = time.perf_counter() - t0
    print(f"  Training time: {t_hnn:.1f}s")

    t_train = {"baseline": 0.0, "pinn": t_pinn, "hnn": t_hnn}

    # ── Build proposals ────────────────────────────────────────────────────────
    proposals = {
        "baseline": make_baseline_proposal(target),
        "pinn":     make_pinn_proposal(pinn_model, target),
        "hnn":      make_hnn_proposal(hnn_model, target),
    }

    # ── SuS experiments ────────────────────────────────────────────────────────
    print("\n[4/5] Running Subset Simulation ...")
    sus_results: Dict = {}
    for r in R_VALUES:
        sus_results[r] = {}
        print(f"\n  r = {r}")
        for method in METHODS:
            print(f"    [{method}] ", end="", flush=True)
            t0 = time.perf_counter()
            res = run_subset_simulation(
                proposal_fn=proposals[method],
                limit_state=ls,
                r=r,
                n_samples_per_level=SUS_N,
                p0=SUS_P0,
                max_levels=SUS_MAX_LEVELS,
                burn_in=SUS_BURN_IN,
                device=device,
            )
            res["total_sampling_time"] = time.perf_counter() - t0
            sus_results[r][method] = res
            print(
                f"P_F={res['P_F']:.3e}  levels={res['levels']}  "
                f"t={res['total_sampling_time']:.1f}s"
            )

    # ── Print table ────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print_results_table(sus_results, R_VALUES, t_train)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\n[5/5] Saving outputs ...")
    for r in R_VALUES:
        plot_scatter_samples(
            sus_results[r], ls, r,
            OUT_DIR / f"scatter_samples_r{int(r)}.png",
        )
    plot_pf_comparison(sus_results, R_VALUES, OUT_DIR / "pf_comparison.png")
    plot_cost_comparison(sus_results, R_VALUES[0], t_train, OUT_DIR / "cost_comparison.png")
    plot_training_loss(pinn_history, hnn_history, OUT_DIR / "training_loss.png")

    # ── Numerical results ──────────────────────────────────────────────────────
    torch.save(
        {"sus_results": sus_results, "t_train": t_train,
         "pinn_history": pinn_history, "hnn_history": hnn_history},
        OUT_DIR / "sus_results.pt",
    )
    print(f"  Saved sus_results.pt")

    # ── Report ─────────────────────────────────────────────────────────────────
    generate_report(sus_results, R_VALUES, t_train, OUT_DIR / "experiment_report.md")

    print(f"\nDone.  All outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
