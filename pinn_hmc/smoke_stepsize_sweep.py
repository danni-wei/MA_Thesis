"""
smoke_stepsize_sweep.py  (Stage 1.6)
──────────────────────────────────────────────────────────────────────────────
Step-size convergence study: mean_IC max_t |dH| vs h, log-log scale.

Fixed total integration time T = 1.0 (= STEP_SIZE × N_STEPS from Stage 1.5).
Step sizes swept: h ∈ {0.2, 0.1, 0.05, 0.02, 0.01, 0.005}.
n_steps = round(T / h) per h value.

For N_IC initial conditions, integrates a trajectory of total time T and
records  mean_IC max_t |H(q_t, p_t) - H(q_0, p_0)|.

Combinations (per target: gaussian, banana):
    true  + leapfrog  -> dH_true
    true  + yoshida   -> dH_true
    hnn   + leapfrog  -> dH_learned  AND  dH_true
    hnn   + yoshida   -> dH_learned  AND  dH_true

Outputs:
    results/smoke_yoshida_hnn/convergence_gaussian.png
    results/smoke_yoshida_hnn/convergence_banana.png
    results/smoke_yoshida_hnn/sweep_results.pt
    Appended slope table to record_yoshida_hnn.md
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .compare_pinn_hnn import leapfrog_hnn_one_step
    from .model import HNNConfig, HamiltonianNN
    from .smoke_yoshida_hnn import GaussianTarget
    from .target import BananaTarget, BananaTargetConfig, get_device
except ImportError:
    from pinn_hmc.compare_pinn_hnn import leapfrog_hnn_one_step
    from pinn_hmc.model import HNNConfig, HamiltonianNN
    from pinn_hmc.smoke_yoshida_hnn import GaussianTarget
    from pinn_hmc.target import BananaTarget, BananaTargetConfig, get_device

# ── Configuration ──────────────────────────────────────────────────────────────

OUT_DIR = Path("results/smoke_yoshida_hnn")
T_FIXED = 1.0            # Stage 1.5 step_size × n_steps = 0.1 × 10
H_VALUES = [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
N_IC = 200
SEED = 42

_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1 = 1.0 / (2.0 - _CBRT2)       #  ~1.3512
_W0 = -_CBRT2 / (2.0 - _CBRT2)   # ~-1.7024


# ── HNN checkpoint loader ──────────────────────────────────────────────────────

def _load_hnn(name: str, device: torch.device) -> HamiltonianNN:
    ck_path = OUT_DIR / f"hnn_{name}.pt"
    if not ck_path.exists():
        raise FileNotFoundError(
            f"Checkpoint {ck_path} not found. Run Stage 1.5 (smoke_yoshida_hnn) first."
        )
    model = HamiltonianNN(HNNConfig(dim=2, hidden_dim=128, depth=4)).to(device)
    model.load_state_dict(torch.load(ck_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  Loaded HNN({name}) from {ck_path.name}")
    return model


# ── Step-by-step trajectory integrators (no p-negation, record H at each step) ─

@torch.no_grad()
def _traj_leapfrog_true(
    target, q0: torch.Tensor, p0: torch.Tensor, h: float, n_steps: int
) -> torch.Tensor:
    """Leapfrog (true gradients), explicit KDK per step (no half-kick merging).
    H recorded at valid KDK endpoint after each step to avoid O(h) bias.
    Returns max_t |dH_true| per IC. Shape: [batch]."""
    q = q0.detach().clone()
    p = p0.detach().clone()
    H0 = target.H(q, p)
    max_dH = torch.zeros(q.shape[0], device=q.device)
    for _ in range(n_steps):
        p = p - 0.5 * h * target.grad_U(q)   # half-kick
        q = q + h * p                          # drift
        p = p - 0.5 * h * target.grad_U(q)   # half-kick → valid KDK state
        max_dH = torch.max(max_dH, (target.H(q, p) - H0).abs())
    return max_dH


@torch.no_grad()
def _traj_yoshida_true(
    target, q0: torch.Tensor, p0: torch.Tensor, h: float, n_steps: int
) -> torch.Tensor:
    """Yoshida (true gradients). H recorded after each complete outer step. Shape: [batch]."""
    q = q0.detach().clone()
    p = p0.detach().clone()
    H0 = target.H(q, p)
    max_dH = torch.zeros(q.shape[0], device=q.device)
    for _ in range(n_steps):
        for wi in (_W1, _W0, _W1):
            hi = wi * h
            p = p - 0.5 * hi * target.grad_U(q)
            q = q + hi * p
            p = p - 0.5 * hi * target.grad_U(q)
        max_dH = torch.max(max_dH, (target.H(q, p) - H0).abs())
    return max_dH


def _traj_leapfrog_hnn(
    model: HamiltonianNN,
    target,
    q0: torch.Tensor,
    p0: torch.Tensor,
    h: float,
    n_steps: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Leapfrog HNN (full vector field). Returns (max dH_learned, max dH_true). Shape: [batch] each."""
    q = q0.detach().clone()
    p = p0.detach().clone()
    with torch.no_grad():
        H0_true = target.H(q, p)
        H0_lrn = model(q, p).squeeze(-1)
    max_dH_lrn = torch.zeros(q.shape[0], device=q.device)
    max_dH_true = torch.zeros(q.shape[0], device=q.device)
    for _ in range(n_steps):
        q, p, _ = leapfrog_hnn_one_step(model, q, p, h)
        with torch.no_grad():
            dH_lrn = (model(q, p).squeeze(-1) - H0_lrn).abs()
            dH_true = (target.H(q, p) - H0_true).abs()
        max_dH_lrn = torch.max(max_dH_lrn, dH_lrn)
        max_dH_true = torch.max(max_dH_true, dH_true)
    return max_dH_lrn, max_dH_true


def _traj_yoshida_hnn(
    model: HamiltonianNN,
    target,
    q0: torch.Tensor,
    p0: torch.Tensor,
    h: float,
    n_steps: int,
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Yoshida HNN. H recorded after each complete outer step. Returns (max dH_learned, max dH_true)."""
    q = q0.detach().clone()
    p = p0.detach().clone()
    with torch.no_grad():
        H0_true = target.H(q, p)
        H0_lrn = model(q, p).squeeze(-1)
    max_dH_lrn = torch.zeros(q.shape[0], device=q.device)
    max_dH_true = torch.zeros(q.shape[0], device=q.device)
    for _ in range(n_steps):
        q, p, _ = leapfrog_hnn_one_step(model, q, p, _W1 * h)
        q, p, _ = leapfrog_hnn_one_step(model, q, p, _W0 * h)
        q, p, _ = leapfrog_hnn_one_step(model, q, p, _W1 * h)
        with torch.no_grad():
            dH_lrn = (model(q, p).squeeze(-1) - H0_lrn).abs()
            dH_true = (target.H(q, p) - H0_true).abs()
        max_dH_lrn = torch.max(max_dH_lrn, dH_lrn)
        max_dH_true = torch.max(max_dH_true, dH_true)
    return max_dH_lrn, max_dH_true


# ── Sweep ──────────────────────────────────────────────────────────────────────

def run_sweep(
    target_name: str,
    target,
    model: HamiltonianNN,
    device: torch.device,
) -> dict:
    """
    Returns dict with keys:
        h_values, n_steps,
        true_lf, true_y4,
        hnn_lf_lrn, hnn_lf_true,
        hnn_y4_lrn, hnn_y4_true
    Each *_lf / *_y4 / *_true is a list of mean max_dH values per h.
    """
    torch.manual_seed(SEED)
    q0 = torch.randn(N_IC, target.dim, device=device)
    p0 = target.sample_momentum(N_IC, device)

    keys = ["true_lf", "true_y4", "hnn_lf_lrn", "hnn_lf_true", "hnn_y4_lrn", "hnn_y4_true"]
    data: dict = {k: [] for k in keys}
    n_steps_list = []

    for h in H_VALUES:
        n_steps = max(1, round(T_FIXED / h))
        n_steps_list.append(n_steps)
        print(f"  [{target_name}] h={h:.3f}  n_steps={n_steps}", end="  ", flush=True)

        v_tlf  = _traj_leapfrog_true(target, q0, p0, h, n_steps).mean().item()
        v_ty4  = _traj_yoshida_true(target, q0, p0, h, n_steps).mean().item()

        lf_lrn, lf_true = _traj_leapfrog_hnn(model, target, q0, p0, h, n_steps)
        v_lf_lrn  = lf_lrn.mean().item()
        v_lf_true = lf_true.mean().item()

        y4_lrn, y4_true = _traj_yoshida_hnn(model, target, q0, p0, h, n_steps)
        v_y4_lrn  = y4_lrn.mean().item()
        v_y4_true = y4_true.mean().item()

        data["true_lf"].append(v_tlf)
        data["true_y4"].append(v_ty4)
        data["hnn_lf_lrn"].append(v_lf_lrn)
        data["hnn_lf_true"].append(v_lf_true)
        data["hnn_y4_lrn"].append(v_y4_lrn)
        data["hnn_y4_true"].append(v_y4_true)

        print(
            f"true_lf={v_tlf:.2e}  true_y4={v_ty4:.2e}  "
            f"hnn_lf_lrn={v_lf_lrn:.2e}  hnn_y4_lrn={v_y4_lrn:.2e}  "
            f"floor={v_lf_true:.2e}"
        )

    data["h_values"] = list(H_VALUES)
    data["n_steps"] = n_steps_list
    return data


# ── Slope fitting ──────────────────────────────────────────────────────────────

def _fit_slope(h_list: list, dH_list: list) -> float:
    h_arr  = np.array(h_list, dtype=float)
    dH_arr = np.array(dH_list, dtype=float)
    mask = (dH_arr > 0) & np.isfinite(dH_arr)
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(h_arr[mask]), np.log(dH_arr[mask]), 1)
    return float(slope)


# ── Convergence plot ───────────────────────────────────────────────────────────

_LINE_STYLES = [
    # (data_key,       label,                     color,            fmt,    lw  )
    ("true_lf",    "true+leapfrog",          "steelblue",      "o-",   1.8),
    ("true_y4",    "true+yoshida",           "darkorange",     "s-",   1.8),
    ("hnn_lf_lrn", "hnn+leapfrog (dH_lrn)", "mediumseagreen", "o--",  1.5),
    ("hnn_y4_lrn", "hnn+yoshida (dH_lrn)",  "firebrick",      "s--",  1.5),
    ("hnn_lf_true","hnn dH_true (floor)",   "gray",           ":^",   1.2),
]


def plot_convergence(data: dict, target_name: str) -> None:
    h_arr = np.array(data["h_values"])

    fig, ax = plt.subplots(figsize=(8, 6))

    for key, label, color, fmt, lw in _LINE_STYLES:
        vals = np.array(data[key])
        slope = _fit_slope(data["h_values"], data[key])
        s_str = f"{slope:.2f}" if np.isfinite(slope) else "N/A"
        ax.plot(h_arr, vals, fmt, color=color, linewidth=lw, markersize=5,
                label=f"{label}  (slope {s_str})")

    # Reference slope lines anchored at h=0.1
    try:
        idx01 = data["h_values"].index(0.1)
        v_lf01 = data["true_lf"][idx01]
        v_y401 = data["true_y4"][idx01]
        h_ref = h_arr
        ax.plot(h_ref, v_lf01 * (h_ref / 0.1) ** 2,
                "k:",  linewidth=0.7, alpha=0.45, label="ref slope 2")
        ax.plot(h_ref, v_y401 * (h_ref / 0.1) ** 4,
                "k-.", linewidth=0.7, alpha=0.45, label="ref slope 4")
    except (ValueError, IndexError):
        pass

    ax.set(
        xscale="log", yscale="log",
        xlabel="step size  h",
        ylabel=r"mean$_{\rm IC}$ $\max_t\,|\Delta H|$",
        title=f"Step-size convergence — {target_name}   (T = {T_FIXED})",
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out = OUT_DIR / f"convergence_{target_name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out.name}")

    # Print slope table to stdout
    print(f"\n  --- Slopes ({target_name}) ---")
    for key, label, *_ in _LINE_STYLES:
        slope = _fit_slope(data["h_values"], data[key])
        s_str = f"{slope:+.3f}" if np.isfinite(slope) else "  N/A"
        print(f"    {label:<30}  slope = {s_str}")
    print()


# ── Record ─────────────────────────────────────────────────────────────────────

def append_record(all_results: dict) -> None:
    out_path = OUT_DIR / "record_yoshida_hnn.md"

    rows = []
    for tgt_name, data in all_results.items():
        for key, label, *_ in _LINE_STYLES:
            slope = _fit_slope(data["h_values"], data[key])
            s_str = f"{slope:.3f}" if np.isfinite(slope) else "N/A"
            rows.append(f"| {tgt_name} | {label} | {s_str} |")
        rows.append("|   |   |   |")

    # h_values and n_steps from first target
    first = next(iter(all_results.values()))
    h_n = "  ".join(
        f"h={h:.3f}(n={n})"
        for h, n in zip(first["h_values"], first["n_steps"])
    )

    section = f"""
---

## Stage 1.6 - Step-size convergence sweep

T_fixed={T_FIXED}, N_IC={N_IC}, h values: {first["h_values"]}

{h_n}

| Target | Curve | log-log slope (full h range) |
|--------|-------|------------------------------|
{chr(10).join(rows)}

### Plots

- `convergence_gaussian.png`
- `convergence_banana.png`

### Reading guide

- true+leapfrog slope ~2 and true+yoshida slope ~4: integrators correct, control passed.
- hnn+leapfrog and hnn+yoshida slopes: if both plateau at small h, dH_true floor
  dominates and Yoshida order advantage does not transfer.
- hnn dH_true floor slope ~0: model error is h-independent, confirming it is the
  fundamental bottleneck for energy conservation in HNN-driven integration.
"""

    existing = out_path.read_text(encoding="utf-8") if out_path.exists() else ""
    out_path.write_text(existing + section, encoding="utf-8")
    print(f"  Appended Stage 1.6 to {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    torch.manual_seed(SEED)
    device = get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"T_fixed={T_FIXED}, N_IC={N_IC}")
    print(f"h values: {H_VALUES}")
    print(f"n_steps:  {[max(1, round(T_FIXED / h)) for h in H_VALUES]}\n")

    gaussian = GaussianTarget()
    banana   = BananaTarget(BananaTargetConfig(b=0.15, sigma1=1.0, sigma2=1.0))

    print("[Loading HNN checkpoints]")
    hnn_gaussian = _load_hnn("gaussian", device)
    hnn_banana   = _load_hnn("banana",   device)
    print()

    all_results: dict = {}
    for tgt_name, tgt, hnn in [
        ("gaussian", gaussian, hnn_gaussian),
        ("banana",   banana,   hnn_banana),
    ]:
        print(f"Sweeping {tgt_name} ...")
        res = run_sweep(tgt_name, tgt, hnn, device)
        all_results[tgt_name] = res
        plot_convergence(res, tgt_name)

    print("[Writing record]")
    append_record(all_results)
    torch.save(all_results, OUT_DIR / "sweep_results.pt")
    print(f"  Saved sweep_results.pt")
    print(f"\nDone. All outputs in {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
