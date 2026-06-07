"""
smoke_yoshida_hnn.py
────────────────────
Single-chain HMC smoke test — Stage 1.5 (full HNN vector field).

4 configs × 2 targets:
    leapfrog + true grad   (reference)
    yoshida4 + true grad   (reference)
    leapfrog + HNN full    (leapfrog_hnn: q-drift uses dH/dp)
    yoshida4 + HNN full    (yoshida4_hnn: Yoshida composition of leapfrog_hnn)

Run from project root:
    python -m pinn_hmc.smoke_yoshida_hnn
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .compare_pinn_hnn import (
        generate_hnn_dataset,
        leapfrog_hnn,
        yoshida4_hnn,
    )
    from .integrators import leapfrog_torch, yoshida4_torch
    from .model import HNNConfig, HamiltonianNN
    from .run_experiment import DatasetConfig
    from .target import BananaTarget, BananaTargetConfig, get_device
    from .train import TrainConfig, train_hnn_model
except ImportError:
    from pinn_hmc.compare_pinn_hnn import (
        generate_hnn_dataset,
        leapfrog_hnn,
        yoshida4_hnn,
    )
    from pinn_hmc.integrators import leapfrog_torch, yoshida4_torch
    from pinn_hmc.model import HNNConfig, HamiltonianNN
    from pinn_hmc.run_experiment import DatasetConfig
    from pinn_hmc.target import BananaTarget, BananaTargetConfig, get_device
    from pinn_hmc.train import TrainConfig, train_hnn_model

# ── Constants ─────────────────────────────────────────────────────────────────

OUT_DIR = Path("results/smoke_yoshida_hnn")
STEP_SIZE = 0.1
N_STEPS = 10
N_PROPOSALS = 2000
BURN_IN = 200
SEED = 42

# Exact hnn_vector_field call counts (for n_steps = 10):
#   leapfrog_hnn      : 2*10+1 = 21 calls/proposal
#   yoshida4_hnn      : 9*10   = 90 calls/proposal  (no half-kick merging)
# Each hnn_vector_field call = 1 forward + 2 backward passes (both dH/dq and dH/dp).
# Cost ratio yoshida4_hnn / leapfrog_hnn = 90/21 ≈ 4.29×
VF_LEAPFROG = 2 * N_STEPS + 1   # 21
VF_YOSHIDA  = 9 * N_STEPS       # 90


# ── Standard Gaussian target ──────────────────────────────────────────────────

class GaussianTarget:
    """2D standard Gaussian: U(q) = 0.5 ||q||^2, K(p) = 0.5 ||p||^2."""

    dim = 2

    def U(self, q: torch.Tensor) -> torch.Tensor:
        return 0.5 * (q ** 2).sum(dim=-1)

    def grad_U(self, q: torch.Tensor) -> torch.Tensor:
        return q.clone()

    def K(self, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (p ** 2).sum(dim=-1)

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.U(q) + self.K(p)

    def sample_momentum(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.dim, device=device)


# ── True grad_fn factory (for reference rows) ─────────────────────────────────

def make_grad_U_true(target) -> Callable:
    def grad_fn(q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return target.grad_U(q).detach()
    return grad_fn


# ── HNN training / loading ────────────────────────────────────────────────────

def _train_or_load_hnn(target, target_name: str, device: torch.device) -> HamiltonianNN:
    ck_path = OUT_DIR / f"hnn_{target_name}.pt"
    model = HamiltonianNN(HNNConfig(dim=2, hidden_dim=128, depth=4)).to(device)

    if ck_path.exists():
        sd = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(sd)
        print(f"  Loaded HNN({target_name}) from {ck_path.name}")
        return model

    print(f"  Training HNN({target_name}) ...")
    data_cfg = DatasetConfig(n_trajectories=1800, step_size=0.08, n_leapfrog_steps=20)
    dataset = generate_hnn_dataset(target, data_cfg, device)
    train_cfg = TrainConfig(
        epochs=120, batch_size=256, lr=1e-3,
        weight_decay=1e-6, grad_clip=1.0, verbose_every=20,
    )
    model, _ = train_hnn_model(model, target, dataset["train_x"], dataset["val_x"], train_cfg)
    torch.save(model.state_dict(), ck_path)
    print(f"  Saved HNN({target_name}) to {ck_path.name}")
    return model


# ── Proposal runner helpers ───────────────────────────────────────────────────

def _run_true_proposal(
    q: torch.Tensor,
    p0: torch.Tensor,
    integrator_fn: Callable,
    grad_fn: Callable,
    step_size: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Runs leapfrog_torch or yoshida4_torch with a true grad_fn."""
    return integrator_fn(q, p0, grad_fn, step_size, n_steps)


def _run_hnn_proposal(
    q: torch.Tensor,
    p0: torch.Tensor,
    integrator_fn: Callable,
    model: HamiltonianNN,
    step_size: float,
    n_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Runs leapfrog_hnn or yoshida4_hnn with the full HNN vector field.
    leapfrog_hnn returns (q, p); yoshida4_hnn returns (q, p, n_vf)."""
    result = integrator_fn(model, q, p0, step_size, n_steps)
    if len(result) == 2:
        q_prop, p_prop = result
        n_ev = 2 * n_steps + 1   # leapfrog_hnn: 2*L+1 calls
    else:
        q_prop, p_prop, n_ev = result
    return q_prop, p_prop, n_ev


# ── Single-chain HMC runner ───────────────────────────────────────────────────

def run_smoke_chain(
    target,
    is_hnn: bool,
    integrator_fn: Callable,
    grad_fn_or_model,           # grad_fn for true runs; HNN model for HNN runs
    n_proposals: int,
    step_size: float,
    n_steps: int,
    seed: int,
    device: torch.device,
) -> dict:
    """
    Single HMC chain with energy-conservation diagnostics.

    For HNN runs (is_hnn=True): integrator uses the full HNN vector field.
    For true runs (is_hnn=False): integrator uses grad_fn from the true target.

    Returns:
        dH_learned  – |ΔH_θ| per proposal (= |ΔH_true| for true runs)
        dH_true     – |ΔH_true| per proposal
        accept_rate – fraction accepted (MH on true H)
        n_grad_evals – total hnn_vf calls (HNN) or grad_fn calls (true)
    """
    torch.manual_seed(seed)
    q = torch.zeros(1, target.dim, device=device)

    # Unwrap for HNN runs
    hnn_model: Optional[HamiltonianNN] = grad_fn_or_model if is_hnn else None
    grad_fn: Optional[Callable] = grad_fn_or_model if not is_hnn else None

    dH_learned_list: list = []
    dH_true_list: list = []
    accept_list: list = []
    total_grads = 0

    for step_idx in range(n_proposals + BURN_IN):
        p0 = target.sample_momentum(1, device)

        with torch.no_grad():
            H0_true = target.H(q, p0)

        if is_hnn:
            q_prop, p_prop, n_ev = _run_hnn_proposal(
                q, p0, integrator_fn, hnn_model, step_size, n_steps,
            )
        else:
            q_prop, p_prop, n_ev = _run_true_proposal(
                q, p0, integrator_fn, grad_fn, step_size, n_steps,
            )
        total_grads += n_ev

        with torch.no_grad():
            H1_true = target.H(q_prop, p_prop)
            dH_true = (H1_true - H0_true).abs().item()

        if is_hnn and hnn_model is not None:
            with torch.no_grad():
                H0_hnn = hnn_model(q.detach(), p0.detach()).squeeze(-1)
                H1_hnn = hnn_model(q_prop.detach(), p_prop.detach()).squeeze(-1)
            dH_learned = (H1_hnn - H0_hnn).abs().item()
        else:
            dH_learned = dH_true

        with torch.no_grad():
            log_alpha = torch.clamp(H0_true - H1_true, max=80.0)
            accepted = (torch.rand(1, device=device) < torch.exp(log_alpha).clamp(max=1.0)).item()

        dH_learned_list.append(dH_learned)
        dH_true_list.append(dH_true)
        accept_list.append(float(accepted))

        if accepted:
            q = q_prop.detach()

    dH_learned_arr = np.array(dH_learned_list[BURN_IN:])
    dH_true_arr    = np.array(dH_true_list[BURN_IN:])
    accept_arr     = np.array(accept_list[BURN_IN:])

    return {
        "dH_learned":   dH_learned_arr,
        "dH_true":      dH_true_arr,
        "accept_rate":  float(accept_arr.mean()),
        "n_grad_evals": total_grads,
    }


# ── Table printing ────────────────────────────────────────────────────────────

_CONFIGS = [
    ("leapfrog", "true"),
    ("yoshida",  "true"),
    ("leapfrog", "hnn"),
    ("yoshida",  "hnn"),
]


def _print_table(results: dict, label: str = "") -> None:
    header = f"{'Config':<20} {'Target':<10} {'dH_learned':>13} {'dH_true':>13} {'accept':>8} {'n_evals':>10}"
    sep = "-" * len(header)
    if label:
        print(f"\n  [{label}]")
    print(f"\n{sep}")
    print(header)
    print(sep)
    for tgt_name in ("gaussian", "banana"):
        for integ, grad in _CONFIGS:
            key = f"{integ}+{grad}"
            r = results[tgt_name][key]
            print(
                f"{key:<20} {tgt_name:<10} "
                f"{r['dH_learned'].mean():>13.3e} "
                f"{r['dH_true'].mean():>13.3e} "
                f"{r['accept_rate']:>8.3f} "
                f"{r['n_grad_evals']:>10}"
            )
        print()
    print(sep)


def _print_comparison(results_v1: dict, results_v15: dict) -> None:
    print("\n" + "=" * 90)
    print("  Stage 1 (grad_fn, q-update = h*p) vs Stage 1.5 (full HNN vector field)")
    print("=" * 90)
    header = (f"{'Config':<20} {'Target':<10} "
              f"{'dH_lrn_v1':>12} {'dH_lrn_v15':>12} "
              f"{'dH_true_v1':>12} {'dH_true_v15':>12} "
              f"{'acpt_v15':>9}")
    sep = "-" * len(header)
    print(header)
    print(sep)
    for tgt_name in ("gaussian", "banana"):
        for integ, grad in _CONFIGS:
            key = f"{integ}+{grad}"
            r1  = results_v1[tgt_name][key]
            r15 = results_v15[tgt_name][key]
            print(
                f"{key:<20} {tgt_name:<10} "
                f"{r1['dH_learned'].mean():>12.3e} {r15['dH_learned'].mean():>12.3e} "
                f"{r1['dH_true'].mean():>12.3e} {r15['dH_true'].mean():>12.3e} "
                f"{r15['accept_rate']:>9.3f}"
            )
        print()
    print(sep)


# ── Plots ─────────────────────────────────────────────────────────────────────

_COLORS = {
    "leapfrog+true": "steelblue",
    "yoshida+true":  "darkorange",
    "leapfrog+hnn":  "mediumseagreen",
    "yoshida+hnn":   "firebrick",
}


def _save_dH_plots(results: dict, tag: str = "v15") -> None:
    """Standard dH_true trajectory plots (4 lines per target)."""
    for tgt_name in ("gaussian", "banana"):
        fig, ax = plt.subplots(figsize=(10, 4))
        for integ, grad in _CONFIGS:
            key = f"{integ}+{grad}"
            dh = results[tgt_name][key]["dH_true"]
            ax.plot(dh, alpha=0.7, linewidth=0.6, color=_COLORS[key], label=key)
        ax.set(
            xlabel="proposal index",
            ylabel="|dH_true|",
            title=f"|dH_true| trajectory — {tgt_name} ({tag})",
            yscale="log",
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out = OUT_DIR / f"dH_trajectory_{tgt_name}_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out.name}")


def _save_scissors_plots(results: dict) -> None:
    """
    Scissors-gap plot: for each target, show dH_learned (solid) and
    dH_true (dashed) for leapfrog+hnn and yoshida+hnn.
    The gap between yoshida+hnn's dH_learned and dH_true is the key result.
    """
    hnn_keys = ["leapfrog+hnn", "yoshida+hnn"]
    line_colors = {"leapfrog+hnn": "mediumseagreen", "yoshida+hnn": "firebrick"}

    for tgt_name in ("gaussian", "banana"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"Scissors gap: dH_learned vs dH_true — {tgt_name}", fontsize=11)

        ax_lrn, ax_true = axes
        for key in hnn_keys:
            r = results[tgt_name][key]
            ax_lrn.plot(r["dH_learned"], alpha=0.7, lw=0.6,
                        color=line_colors[key], label=key)
            ax_true.plot(r["dH_true"],   alpha=0.7, lw=0.6,
                         color=line_colors[key], label=key)

        # Also show yoshida+true as reference floor
        r_ref = results[tgt_name]["yoshida+true"]
        ax_lrn.axhline(r_ref["dH_learned"].mean(), color="darkorange",
                       ls="--", lw=1.2, label="yoshida+true (ref)")
        ax_true.axhline(r_ref["dH_true"].mean(), color="darkorange",
                        ls="--", lw=1.2, label="yoshida+true (ref)")

        for ax, ylabel, title in [
            (ax_lrn,  "|dH_learned|", "Learned Hamiltonian energy error"),
            (ax_true, "|dH_true|",    "True Hamiltonian energy error"),
        ]:
            ax.set(xlabel="proposal index", ylabel=ylabel, title=title, yscale="log")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out = OUT_DIR / f"scissors_{tgt_name}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Saved {out.name}")


# ── Record ────────────────────────────────────────────────────────────────────

def _append_record(results: dict) -> None:
    out_path = OUT_DIR / "record_yoshida_hnn.md"

    rows_v15 = []
    for tgt_name in ("gaussian", "banana"):
        for integ, grad in _CONFIGS:
            key = f"{integ}+{grad}"
            r = results[tgt_name][key]
            rows_v15.append(
                f"| {key} | {tgt_name} "
                f"| {r['dH_learned'].mean():.3e} "
                f"| {r['dH_true'].mean():.3e} "
                f"| {r['accept_rate']:.3f} "
                f"| {r['n_grad_evals']} |"
            )

    lf_hnn  = results["gaussian"]["leapfrog+hnn"]
    y4_hnn  = results["gaussian"]["yoshida+hnn"]
    ratio_learned = lf_hnn["dH_learned"].mean() / y4_hnn["dH_learned"].mean()
    ratio_true    = lf_hnn["dH_true"].mean()    / y4_hnn["dH_true"].mean()
    vf_ratio      = VF_YOSHIDA / VF_LEAPFROG

    section = f"""
---

## Stage 1.5 — Full HNN vector field (q-drift uses dH/dp)

### Setup

Same hyper-parameters as Stage 1 (step_size={STEP_SIZE}, n_steps={N_STEPS},
n_proposals={N_PROPOSALS}, same HNN checkpoints).

**Key change**: leapfrog+hnn and yoshida+hnn now use the complete HNN vector
field for BOTH q-drift (dH/dp) and p-kick (-dH/dq).

hnn_vector_field calls per proposal:
- leapfrog_hnn  : 2*{N_STEPS}+1 = {VF_LEAPFROG} calls  (each = 1 forward + 2 backward)
- yoshida4_hnn  : 9*{N_STEPS}   = {VF_YOSHIDA} calls  (3 sub-steps x 3 calls, no merging)
- Cost ratio    : {VF_YOSHIDA}/{VF_LEAPFROG} = {vf_ratio:.2f}x

### Results

| Config | Target | dH_learned | dH_true | accept | n_vf_calls |
|--------|--------|-----------|---------|--------|-----------|
{chr(10).join(rows_v15)}

### Observation (Gaussian)

- yoshida+hnn dH_learned / leapfrog+hnn dH_learned = {ratio_learned:.1f}x {'(order(s) of magnitude improvement -- scissors confirmed!)' if ratio_learned > 50 else f'(improvement factor; expected >>100x for 4th order)' if ratio_learned > 5 else '(no significant improvement -- check symmetry of base step)'}
- yoshida+hnn dH_true    / leapfrog+hnn dH_true    = {ratio_true:.2f}x  (HNN model error floor)
- Scissors gap {'confirmed' if ratio_learned / ratio_true > 5 else 'marginal or absent'}: dH_learned drops by {ratio_learned:.1f}x but dH_true only by {ratio_true:.2f}x

### Plots

- `dH_trajectory_gaussian_v15.png` / `dH_trajectory_banana_v15.png` — |dH_true| traces
- `scissors_gaussian.png` / `scissors_banana.png` — dH_learned vs dH_true side-by-side
"""

    existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    out_path.write_text(existing + section, encoding="utf-8")
    print(f"  Appended Stage 1.5 to {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"step_size={STEP_SIZE}, n_steps={N_STEPS}, n_proposals={N_PROPOSALS}, burn_in={BURN_IN}")
    print(f"HNN vf calls: leapfrog_hnn={VF_LEAPFROG}/proposal, yoshida4_hnn={VF_YOSHIDA}/proposal "
          f"(ratio {VF_YOSHIDA}/{VF_LEAPFROG}={VF_YOSHIDA/VF_LEAPFROG:.2f}x)\n")

    gaussian = GaussianTarget()
    banana   = BananaTarget(BananaTargetConfig(b=0.15, sigma1=1.0, sigma2=1.0))

    # ── Load HNN checkpoints ──────────────────────────────────────────────────
    print("[HNN setup]")
    hnn_gaussian = _train_or_load_hnn(gaussian, "gaussian", device)
    hnn_banana   = _train_or_load_hnn(banana,   "banana",   device)
    print()

    # ── Build per-config runners ──────────────────────────────────────────────
    # Config: (integ_name, grad_name, is_hnn, integrator_fn, grad_fn_or_model)
    def make_configs(target, hnn):
        gfn = make_grad_U_true(target)
        return [
            ("leapfrog", "true", False, leapfrog_torch,  gfn),
            ("yoshida",  "true", False, yoshida4_torch,  gfn),
            ("leapfrog", "hnn",  True,  leapfrog_hnn,    hnn),
            ("yoshida",  "hnn",  True,  yoshida4_hnn,    hnn),
        ]

    target_map = {"gaussian": (gaussian, hnn_gaussian), "banana": (banana, hnn_banana)}

    results: dict = {}
    for tgt_name, (target, hnn) in target_map.items():
        results[tgt_name] = {}
        for integ_name, grad_name, is_hnn, integ_fn, gfn_or_model in make_configs(target, hnn):
            key = f"{integ_name}+{grad_name}"
            print(f"  [{tgt_name}] {key} ...", end=" ", flush=True)
            res = run_smoke_chain(
                target=target,
                is_hnn=is_hnn,
                integrator_fn=integ_fn,
                grad_fn_or_model=gfn_or_model,
                n_proposals=N_PROPOSALS,
                step_size=STEP_SIZE,
                n_steps=N_STEPS,
                seed=SEED,
                device=device,
            )
            results[tgt_name][key] = res
            print(
                f"accept={res['accept_rate']:.3f}  "
                f"dH_lrn={res['dH_learned'].mean():.3e}  "
                f"dH_true={res['dH_true'].mean():.3e}  "
                f"n_ev={res['n_grad_evals']}"
            )

    # ── Print Stage 1.5 table ─────────────────────────────────────────────────
    _print_table(results, label="Stage 1.5: full HNN vector field")

    # ── Compare with Stage 1 if available ────────────────────────────────────
    v1_path = OUT_DIR / "smoke_results.pt"
    if v1_path.exists():
        print("\nLoading Stage 1 results for comparison ...")
        results_v1 = torch.load(v1_path, map_location="cpu", weights_only=False)
        _print_comparison(results_v1, results)

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[Saving outputs]")
    _save_dH_plots(results, tag="v15")
    _save_scissors_plots(results)
    _append_record(results)
    torch.save(results, OUT_DIR / "smoke_results_v15.pt")
    print(f"  Saved smoke_results_v15.pt")
    print(f"\nDone. All outputs in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
