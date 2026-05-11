"""
experiment_jacobian.py
──────────────────────
Jacobian-correction experiment: compares three MCMC kernels in Subset
Simulation on the WangBananaTarget.

    Baseline HMC          – exact leapfrog + standard M-H
    PINN-HMC              – PINN proposal, UNcorrected acceptance (original)
    PINN-HMC-corrected    – PINN proposal, Jacobian-corrected acceptance

Run from the project root:
    python -m pinn_hmc.experiment_jacobian
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
from .model import ModelConfig, TrajectoryPINN
from .run_experiment import (
    DatasetConfig,
    compute_pinn_log_jacobian,
    generate_teacher_dataset,
)
from .sus import run_subset_simulation
from .target import EllipticalLimitState, WangBananaTarget, get_device
from .train import TrainConfig, train_model

# ── Experiment-wide constants ─────────────────────────────────────────────────

STEP_SIZE   = 0.08
N_STEPS     = 20
T_FINAL     = STEP_SIZE * N_STEPS   # 1.6 s

# SuS parameters – N is kept smaller than experiment_sus (500) because the
# corrected proposal computes one 4×4 Jacobian per sample per step.
SUS_N          = 200
SUS_P0         = 0.1
SUS_MAX_LEVELS = 20
SUS_BURN_IN    = 200

R_VALUES: List[float] = [10.0, 12.0]

OUT_DIR = Path("results/jacobian_experiment")

METHODS = ["baseline", "pinn", "pinn_corrected"]
LABELS  = {
    "baseline":       "Baseline HMC",
    "pinn":           "PINN-HMC",
    "pinn_corrected": "PINN-HMC-corrected",
}
COLORS  = {
    "baseline":       "steelblue",
    "pinn":           "darkorange",
    "pinn_corrected": "forestgreen",
}

# ── Proposal factories ────────────────────────────────────────────────────────

def make_baseline_proposal(
    target: WangBananaTarget,
    step_size: float = STEP_SIZE,
    n_steps: int = N_STEPS,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Exact leapfrog + standard Metropolis accept/reject."""

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
    """PINN forward pass + standard (UNcorrected) Metropolis accept/reject."""
    model.eval()

    def proposal_fn(q: torch.Tensor) -> torch.Tensor:
        q = q.detach()
        N = q.shape[0]
        p0 = target.sample_momentum(N, q.device)
        t = torch.full((N, 1), float(t_final), device=q.device, dtype=q.dtype)

        with torch.no_grad():
            q_prop, p_prop = model(q, p0, t)
            p_prop = -p_prop
            log_alpha = torch.clamp(
                target.H(q, p0) - target.H(q_prop, p_prop), max=80.0
            )
            accept = (
                torch.rand(N, device=q.device) < torch.exp(log_alpha).clamp(max=1.0)
            ).unsqueeze(-1)
        return torch.where(accept, q_prop, q)

    return proposal_fn


def make_pinn_corrected_proposal(
    model: TrajectoryPINN,
    target: WangBananaTarget,
    t_final: float = T_FINAL,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    PINN forward pass + Jacobian-corrected Metropolis accept/reject.

        log α = H(q0,p0) − H(q_T,p_T) − log|det J_PINN(q0,p0)|

    The Jacobian is computed for each sample independently (per-sample loop).
    This is more expensive than the uncorrected version but maintains the
    correct stationary distribution of H.
    """
    model.eval()

    def proposal_fn(q: torch.Tensor) -> torch.Tensor:
        q = q.detach()
        N = q.shape[0]

        with torch.no_grad():
            p0 = target.sample_momentum(N, q.device)
            t = torch.full((N, 1), float(t_final), device=q.device, dtype=q.dtype)
            q_prop, p_prop = model(q, p0, t)
            p_prop = -p_prop
            H0 = target.H(q, p0)
            H1 = target.H(q_prop, p_prop)

        # Jacobian correction – one 4×4 Jacobian per chain
        log_det_J = torch.zeros(N, device=q.device, dtype=q.dtype)
        for i in range(N):
            log_det_J[i] = compute_pinn_log_jacobian(
                model, q[i : i + 1].detach(), p0[i : i + 1].detach(), t_final
            )

        with torch.no_grad():
            log_alpha = torch.clamp(H0 - H1 - log_det_J, max=80.0)
            accept = (
                torch.rand(N, device=q.device) < torch.exp(log_alpha).clamp(max=1.0)
            ).unsqueeze(-1)
        return torch.where(accept, q_prop, q)

    return proposal_fn


# ── log|det J| collection ─────────────────────────────────────────────────────

def collect_log_det_J(
    model: TrajectoryPINN,
    target: WangBananaTarget,
    t_final: float,
    n_samples: int,
    device: torch.device,
) -> List[float]:
    """
    Draw n_samples random (q, p) pairs and compute log|det J_PINN| for each.
    Used to characterise how far the PINN map deviates from volume-preservation.
    """
    model.eval()
    log_dets: List[float] = []
    q_rand = 1.8 * torch.randn(n_samples, 2, device=device)
    for i in range(n_samples):
        p = target.sample_momentum(1, device)
        ld = compute_pinn_log_jacobian(model, q_rand[i : i + 1], p, t_final)
        log_dets.append(float(ld.item()))
    return log_dets


# ── Plotting ──────────────────────────────────────────────────────────────────

def _ellipse_xy(ls: EllipticalLimitState, r: float, n: int = 400):
    angles = np.linspace(0, 2 * np.pi, n)
    z1 = r * ls.c1 * np.cos(angles)
    z2 = r * ls.c2 * np.sin(angles)
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
            ax.scatter(
                s[:, 0].numpy(), s[:, 1].numpy(),
                s=4, alpha=0.5, color=c, label=f"L{lvl}",
            )
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
    fig, ax = plt.subplots(figsize=(9, 5))

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
    ax.set_title("Failure probability comparison (Jacobian correction study)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_jacobian_stats(log_det_J_values: List[float], out_path: Path):
    """Histogram of log|det J_PINN| showing deviation from volume-preservation."""
    vals = np.array(log_det_J_values)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("PINN Jacobian statistics  (log|det J_PINN|)", fontsize=12)

    # Histogram
    ax = axes[0]
    ax.hist(vals, bins=30, color="forestgreen", edgecolor="white", alpha=0.8)
    ax.axvline(0.0, color="k", ls="--", lw=1.5, label="Volume-preserving (log|det J|=0)")
    ax.axvline(float(np.mean(vals)), color="crimson", ls="-", lw=1.5,
               label=f"Mean = {np.mean(vals):.3f}")
    ax.set_xlabel("log|det $J_{PINN}$|")
    ax.set_ylabel("Count")
    ax.set_title("Distribution over random (q, p)")
    ax.legend(fontsize=8)

    # |det J| on linear scale (capped for readability)
    ax2 = axes[1]
    det_vals = np.exp(np.clip(vals, -10, 10))
    ax2.hist(det_vals, bins=30, color="darkorange", edgecolor="white", alpha=0.8)
    ax2.axvline(1.0, color="k", ls="--", lw=1.5, label="|det J| = 1 (symplectic)")
    ax2.axvline(float(np.mean(det_vals)), color="crimson", ls="-", lw=1.5,
                label=f"Mean = {np.mean(det_vals):.3f}")
    ax2.set_xlabel("|det $J_{PINN}$|  (exp-capped at ±10)")
    ax2.set_ylabel("Count")
    ax2.set_title("|det J| on linear scale")
    ax2.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path.name}")


def plot_cost_comparison(
    results: Dict,
    r: float,
    t_train_pinn: float,
    out_path: Path,
):
    metrics = [
        ("training_time",       "Training (s)"),
        ("mean_proposal_time",  "Per-proposal (s)"),
        ("total_sampling_time", "Total sampling (s)"),
    ]
    n_met = len(metrics)
    n_m   = len(METHODS)
    bw    = 0.7 / n_met

    fig, ax = plt.subplots(figsize=(10, 5))
    t_train = {"baseline": 0.0, "pinn": t_train_pinn, "pinn_corrected": t_train_pinn}

    for j, (key, label) in enumerate(metrics):
        xs   = [i + (j - n_met / 2 + 0.5) * bw for i in range(n_m)]
        vals = []
        for method in METHODS:
            if key == "training_time":
                vals.append(t_train[method])
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


# ── Results table ─────────────────────────────────────────────────────────────

def print_results_table(results: Dict, r_values: List[float], t_train_pinn: float):
    t_train = {"baseline": 0.0, "pinn": t_train_pinn, "pinn_corrected": t_train_pinn}
    hdr = (f"{'Method':<22} {'r':>6} {'P_F':>12} {'Lvls':>5} "
           f"{'Train(s)':>10} {'Prop(ms)':>10} {'Samp(s)':>10}")
    sep = "-" * len(hdr)
    print(sep); print(hdr); print(sep)
    for r in r_values:
        for method in METHODS:
            res = results[r][method]
            print(
                f"{LABELS[method]:<22} {r:>6.0f} "
                f"{res['P_F']:>12.4e} {res['levels']:>5} "
                f"{t_train[method]:>10.2f} "
                f"{res['mean_proposal_time'] * 1e3:>10.3f} "
                f"{res['total_sampling_time']:>10.2f}"
            )
    print(sep)


# ── Report ────────────────────────────────────────────────────────────────────

def generate_report(
    results: Dict,
    r_values: List[float],
    t_train_pinn: float,
    log_det_stats: Dict,
    out_path: Path,
):
    t_train = {"baseline": 0.0, "pinn": t_train_pinn, "pinn_corrected": t_train_pinn}

    rows = []
    for r in r_values:
        for method in METHODS:
            res = results[r][method]
            rows.append(
                f"| {LABELS[method]} | {r:.0f} | {res['P_F']:.4e} "
                f"| {res['levels']} | {t_train[method]:.2f} "
                f"| {res['mean_proposal_time'] * 1e3:.3f} "
                f"| {res['total_sampling_time']:.2f} |"
            )
    table = "\n".join(rows)

    md = f"""\
# Jacobian Correction Experiment: PINN-HMC vs PINN-HMC-corrected

## 1. Background

Standard HMC uses a leapfrog (symplectic) integrator, which is volume-preserving:
`|det J_leapfrog| = 1`. The Metropolis-Hastings acceptance ratio simplifies to

    α = min(1, exp(−ΔH))

When the leapfrog is replaced by a PINN (Physics-Informed Neural Network), the
map `(q₀, p₀) → (q_T, p_T)` is no longer guaranteed to be volume-preserving.
The correct acceptance ratio is

    α = min(1, exp(−ΔH − log|det J_PINN(q₀, p₀)|))

Prior work (experiment_sus.py) used the *uncorrected* formula, which may bias
the stationary distribution and distort failure-probability estimates.

This experiment quantifies the discrepancy.

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Target | `WangBananaTarget` (Wang 2019): a={1.15}, b={0.5}, ρ={0.9} |
| Limit-state | `EllipticalLimitState`: c₁=1, c₂=0.5, θ=π/4 |
| Radii r | {', '.join(str(int(r)) for r in r_values)} |
| PINN architecture | 4-layer MLP, 128 hidden, Tanh |
| PINN training | 120 epochs, Adam 1e-3, ProgressBalancer α=1 |
| Dataset | 1 800 trajectories, step size {STEP_SIZE}, {N_STEPS} leapfrog steps |
| SuS N | {SUS_N} samples/level (reduced vs. experiment_sus due to Jacobian cost) |
| SuS p₀ | {SUS_P0} |
| SuS max levels | {SUS_MAX_LEVELS} |
| SuS burn-in | {SUS_BURN_IN} |
| Jacobian | torch.autograd.functional.jacobian (4×4), slogdet for stability |

## 3. Jacobian Statistics

| Statistic | Value |
|-----------|-------|
| Mean log|det J| | {log_det_stats['mean']:.4f} |
| Std  log|det J| | {log_det_stats['std']:.4f} |
| Min  log|det J| | {log_det_stats['min']:.4f} |
| Max  log|det J| | {log_det_stats['max']:.4f} |
| Mean |det J|    | {log_det_stats['mean_det']:.4f} |

A mean log|det J| of {log_det_stats['mean']:.4f} means the PINN map on average
{'expands' if log_det_stats['mean'] > 0 else 'shrinks'} phase-space volume by a factor
of ≈ {np.exp(log_det_stats['mean']):.3f}.  For a truly symplectic map this would be 0.

## 4. Results

| Method | r | P_F | Levels | Train (s) | Prop (ms) | Samp (s) |
|--------|---|-----|--------|-----------|-----------|----------|
{table}

## 5. Analysis

[TODO: Fill in after running the experiment]

Key questions to address:
- Does the Jacobian correction significantly change P_F estimates?
  If |log|det J|| is small (< 0.01), the correction is negligible.
  If it is large, the uncorrected PINN-HMC may produce a biased chain.
- Does the corrected acceptance rate differ substantially from the uncorrected?
  Higher |log|det J| ⇒ larger discrepancy in acceptance probabilities.
- Is the computational overhead of the Jacobian correction justified given the
  difference in P_F estimates?

## 6. Limitations

* **N = {SUS_N}**: Smaller than experiment_sus (N=500) because computing N
  Jacobians per SuS step is expensive.  This increases estimator variance
  (CoV ≈ {np.sqrt((1 - SUS_P0) / (SUS_P0 * SUS_N)):.2f} per level).
* **Single run**: No confidence intervals; results are indicative.
* **Fixed PINN**: The same trained model is reused across all SuS levels.
  Level-adaptive retraining (RS-HMC-SS) could reduce distribution-shift bias.
* **4D Jacobian only**: We consider the full (q, p) map; Jacobian in q-space
  alone (marginalising p) would require further analysis.
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

    # ── Dataset ────────────────────────────────────────────────────────────────
    print("\n[1/5] Generating PINN training dataset ...")
    data_cfg = DatasetConfig(
        n_trajectories=1800,
        step_size=STEP_SIZE,
        n_leapfrog_steps=N_STEPS,
    )
    dataset = generate_teacher_dataset(target, data_cfg, device)
    print(f"  train: {tuple(dataset['train_x'].shape)}  "
          f"val: {tuple(dataset['val_x'].shape)}")

    # ── Train PINN ─────────────────────────────────────────────────────────────
    print("\n[2/5] Training PINN (ProgressBalancer, 120 epochs) ...")
    train_cfg = TrainConfig(
        epochs=120, batch_size=256, lr=1e-3,
        weight_decay=1e-6, grad_clip=1.0, verbose_every=20,
    )
    pinn_model = TrajectoryPINN(ModelConfig(dim=2, hidden_dim=128, depth=4)).to(device)
    t0 = time.perf_counter()
    pinn_model, pinn_history = train_model(
        pinn_model, target,
        dataset["train_x"], dataset["train_y"],
        dataset["val_x"],   dataset["val_y"],
        train_cfg, ProgressBalancer(alpha=1.0),
    )
    t_train_pinn = time.perf_counter() - t0
    print(f"  Training time: {t_train_pinn:.1f}s")

    # ── Collect log|det J| statistics ─────────────────────────────────────────
    print("\n[3/5] Collecting Jacobian statistics (200 random samples) ...")
    t0 = time.perf_counter()
    log_det_J_values = collect_log_det_J(pinn_model, target, T_FINAL, 200, device)
    t_jac = time.perf_counter() - t0
    ld_arr = np.array(log_det_J_values)
    log_det_stats = {
        "mean":     float(np.mean(ld_arr)),
        "std":      float(np.std(ld_arr)),
        "min":      float(np.min(ld_arr)),
        "max":      float(np.max(ld_arr)),
        "mean_det": float(np.mean(np.exp(np.clip(ld_arr, -20, 20)))),
    }
    print(f"  Done in {t_jac:.1f}s")
    print(f"  log|det J|: mean={log_det_stats['mean']:.4f}  "
          f"std={log_det_stats['std']:.4f}  "
          f"min={log_det_stats['min']:.4f}  max={log_det_stats['max']:.4f}")
    print(f"  mean |det J| = {log_det_stats['mean_det']:.4f}  "
          f"(1.0 = volume-preserving)")

    # ── Build proposals ────────────────────────────────────────────────────────
    proposals = {
        "baseline":       make_baseline_proposal(target),
        "pinn":           make_pinn_proposal(pinn_model, target),
        "pinn_corrected": make_pinn_corrected_proposal(pinn_model, target),
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

    # ── Console table ──────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print_results_table(sus_results, R_VALUES, t_train_pinn)

    # ── Plots & outputs ────────────────────────────────────────────────────────
    print("\n[5/5] Saving outputs ...")
    for r in R_VALUES:
        plot_scatter_samples(
            sus_results[r], ls, r,
            OUT_DIR / f"scatter_samples_r{int(r)}.png",
        )

    plot_pf_comparison(sus_results, R_VALUES, OUT_DIR / "pf_comparison.png")
    plot_jacobian_stats(log_det_J_values, OUT_DIR / "jacobian_stats.png")
    plot_cost_comparison(sus_results, R_VALUES[0], t_train_pinn,
                         OUT_DIR / "cost_comparison.png")

    # Numerical checkpoint
    torch.save(
        {
            "sus_results":      sus_results,
            "t_train_pinn":     t_train_pinn,
            "pinn_history":     pinn_history,
            "log_det_J_values": log_det_J_values,
            "log_det_stats":    log_det_stats,
        },
        OUT_DIR / "jacobian_results.pt",
    )
    print("  Saved jacobian_results.pt")

    generate_report(
        sus_results, R_VALUES, t_train_pinn, log_det_stats,
        OUT_DIR / "experiment_report.md",
    )

    print(f"\nDone.  All outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
