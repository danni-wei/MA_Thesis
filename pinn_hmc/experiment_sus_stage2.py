"""
experiment_sus_stage2.py
─────────────────────────
Stage 2 — SuS integrator comparison: leapfrog+hnn vs yoshida+hnn.

Run from project root:
    python -u _run_v2stage2.py           # smoke (N_rep=2)
    python -u _run_v2stage2.py --full    # full  (N_rep=30)
"""
from __future__ import annotations

import io
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# Force UTF-8 stdout so non-ASCII glyphs don't crash on Windows cp1252 consoles.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .compare_pinn_hnn import leapfrog_hnn_one_step, yoshida4_hnn
    from .model import HNNConfig, HamiltonianNN
    from .smoke_yoshida_hnn import GaussianTarget
    from .target import BananaTarget, BananaTargetConfig, get_device
except ImportError:
    from pinn_hmc.compare_pinn_hnn import leapfrog_hnn_one_step, yoshida4_hnn
    from pinn_hmc.model import HNNConfig, HamiltonianNN
    from pinn_hmc.smoke_yoshida_hnn import GaussianTarget
    from pinn_hmc.target import BananaTarget, BananaTargetConfig, get_device

# ── Constants ──────────────────────────────────────────────────────────────────

OUT_DIR  = Path("results/smoke_yoshida_hnn")
REC_PATH = OUT_DIR / "record_yoshida_hnn.md"
LOG_PATH = OUT_DIR / "sus_stage2_run.log"

STEP_SIZE   = 0.1
N_STEPS     = 10
VF_LEAPFROG = 2 * N_STEPS + 1   # 21
VF_YOSHIDA  = 9 * N_STEPS       # 90

BETA   = 3.0
PF_REF = 1.3499e-3   # Φ(-3)

SUS_N          = 500
SUS_P0         = 0.1
SUS_BURN_IN    = 200
SUS_MAX_LEVELS = 8
SEED_BASE      = 0

# ── Logging (stdout + file, always flushed) ────────────────────────────────────

_log_fh = None

def _log(msg: str = "") -> None:
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")
        _log_fh.flush()

def _open_log() -> None:
    global _log_fh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

def _close_log() -> None:
    if _log_fh is not None:
        _log_fh.close()


# ── Linear limit-state ─────────────────────────────────────────────────────────

class LinearLimitState:
    """g(q) = β − q[0].  Failure: g(q) ≤ 0, i.e. q[0] ≥ β."""
    def __init__(self, beta: float = BETA):
        self.beta = beta

    def evaluate(self, q: torch.Tensor) -> torch.Tensor:
        return self.beta - q[:, 0]

    def is_failure(self, q: torch.Tensor) -> torch.Tensor:
        return self.evaluate(q) <= 0.0


# ── HNN loading ────────────────────────────────────────────────────────────────

def _load_hnn(name: str, device: torch.device) -> HamiltonianNN:
    ck_path = OUT_DIR / f"hnn_{name}.pt"
    if not ck_path.exists():
        raise FileNotFoundError(
            f"HNN checkpoint not found: {ck_path}\n"
            "Run Stage 1.5 first to train/save HNN models."
        )
    model = HamiltonianNN(HNNConfig(dim=2, hidden_dim=128, depth=4)).to(device)
    sd = torch.load(ck_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    _log(f"  Loaded HNN({name}) from {ck_path.name}")
    return model


# ── Integrator factories ───────────────────────────────────────────────────────

def make_integrate_leapfrog_hnn(
    model: HamiltonianNN, step_size: float, n_steps: int,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def integrate(q: torch.Tensor, p: torch.Tensor):
        for _ in range(n_steps):
            q, p, _ = leapfrog_hnn_one_step(model, q, p, step_size)
        return q, p
    return integrate


def make_integrate_yoshida_hnn(
    model: HamiltonianNN, step_size: float, n_steps: int,
) -> Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    def integrate(q: torch.Tensor, p: torch.Tensor):
        q_new, p_neg, _ = yoshida4_hnn(model, q, p, step_size, n_steps)
        return q_new, -p_neg   # yoshida4_hnn returns -p; un-negate
    return integrate


# ── Single SuS run ─────────────────────────────────────────────────────────────

def _run_one_sus(
    integrate_fn: Callable,
    target,
    limit_state: LinearLimitState,
    device: torch.device,
    seed: int,
    rep_label: str = "",
) -> Dict:
    torch.manual_seed(seed)

    N       = SUS_N
    p0_frac = SUS_P0
    n_seeds = max(1, int(round(p0_frac * N)))
    spc     = math.ceil(N / n_seeds)

    thresholds:            List[float] = []
    per_level_geom_rej:    List[float] = []
    per_level_energy_rej:  List[float] = []
    per_level_accept_rate: List[float] = []

    # ── burn-in ────────────────────────────────────────────────────────────────
    q = torch.zeros(N, 2, device=device)
    for _ in range(SUS_BURN_IN):
        p_cur  = target.sample_momentum(N, device)
        q_raw, p_raw = integrate_fn(q, p_cur)
        with torch.no_grad():
            H0 = target.H(q, p_cur)
            H1 = target.H(q_raw, p_raw)
            mh_ok = (torch.rand(N, device=device)
                     < torch.exp(torch.clamp(H0 - H1, max=80.0)).clamp(max=1.0))
            q = torch.where(mh_ok.unsqueeze(-1), q_raw, q)

    # ── conditional levels ─────────────────────────────────────────────────────
    for level in range(SUS_MAX_LEVELS):
        with torch.no_grad():
            g_vals = limit_state.evaluate(q)

        threshold = float(torch.quantile(g_vals, p0_frac).item())
        thresholds.append(threshold)
        _log(f"    {rep_label} L{level+1}: threshold={threshold:.4f}")

        if threshold <= 0.0:
            n_fail = int((g_vals <= 0.0).sum().item())
            P_F = (p0_frac ** level) * (n_fail / N)
            _log(f"    {rep_label} → terminated at L{level+1}, Pf={P_F:.4e}")
            return _build_result(P_F, level + 1, thresholds,
                                 per_level_geom_rej, per_level_energy_rej, per_level_accept_rate)

        # MCMC resampling
        seed_idx        = torch.argsort(g_vals)[:n_seeds]
        chains          = q[seed_idx].detach().clone()
        chain_snapshots = [chains.clone()]
        lv_geom = lv_ener = lv_acpt = lv_n = 0

        for _ in range(spc - 1):
            p_cur   = target.sample_momentum(n_seeds, device)
            q_raw, p_raw = integrate_fn(chains, p_cur)
            with torch.no_grad():
                geom_ok = limit_state.evaluate(q_raw) <= threshold
                H0      = target.H(chains, p_cur)
                H1      = target.H(q_raw, p_raw)
                mh_ok   = (torch.rand(n_seeds, device=device)
                           < torch.exp(torch.clamp(H0 - H1, max=80.0)).clamp(max=1.0))
                full_ok = geom_ok & mh_ok
                chains  = torch.where(full_ok.unsqueeze(-1), q_raw, chains)

            lv_geom += int((~geom_ok).sum())
            lv_ener += int((geom_ok & ~mh_ok).sum())
            lv_acpt += int(full_ok.sum())
            lv_n    += n_seeds
            chain_snapshots.append(chains.clone())

        per_level_geom_rej.append(lv_geom / max(lv_n, 1))
        per_level_energy_rej.append(lv_ener / max(lv_n, 1))
        per_level_accept_rate.append(lv_acpt / max(lv_n, 1))

        q = torch.stack(chain_snapshots, dim=1).reshape(-1, 2)[:N]

    with torch.no_grad():
        g_final = limit_state.evaluate(q)
    n_fail = int((g_final <= 0.0).sum())
    P_F = (p0_frac ** SUS_MAX_LEVELS) * (n_fail / N)
    _log(f"    {rep_label} → max_levels reached, Pf={P_F:.4e}")
    return _build_result(P_F, SUS_MAX_LEVELS, thresholds,
                         per_level_geom_rej, per_level_energy_rej, per_level_accept_rate)


def _build_result(P_F, levels, thresholds, geom, ener, acpt) -> Dict:
    return dict(P_F=P_F, levels=levels, thresholds=thresholds,
                per_level_geom_reject=geom,
                per_level_energy_reject=ener,
                per_level_accept_rate=acpt)


# ── Replications ───────────────────────────────────────────────────────────────

def run_replications(
    target_name: str, target, integrate_fn: Callable,
    integ_name: str, vf_per_proposal: int, device: torch.device,
    n_rep: int,
) -> Dict:
    _log(f"\n[{target_name}] {integ_name}  ({n_rep} reps)")
    results = []
    t0 = time.perf_counter()

    for rep in range(n_rep):
        rep_label = f"rep{rep+1:02d}/{n_rep}"
        _log(f"  {rep_label} seed={SEED_BASE+rep} ...")
        t1 = time.perf_counter()
        res = _run_one_sus(integrate_fn, target, LinearLimitState(BETA),
                           device, seed=SEED_BASE + rep, rep_label=rep_label)
        results.append(res)
        elapsed_rep = time.perf_counter() - t1
        elapsed_tot = time.perf_counter() - t0
        _log(f"  {rep_label} done: Pf={res['P_F']:.4e}  levels={res['levels']}"
             f"  ({elapsed_rep:.1f}s rep, {elapsed_tot:.1f}s total)")

    pf_vals    = np.array([r["P_F"]     for r in results])
    lv_vals    = np.array([r["levels"]  for r in results])
    mean_pf    = float(pf_vals.mean())
    std_pf     = float(pf_vals.std())
    cov_pf     = std_pf / mean_pf if mean_pf > 0 else float("nan")
    mean_levels = float(lv_vals.mean())

    n_seeds = max(1, int(round(SUS_P0 * SUS_N)))
    spc     = math.ceil(SUS_N / n_seeds)
    est_proposals = SUS_N * SUS_BURN_IN + mean_levels * n_seeds * (spc - 1)
    est_vf = vf_per_proposal * est_proposals

    max_lv = max(len(r["per_level_geom_reject"]) for r in results)
    lv_geom, lv_ener, lv_acpt = [], [], []
    for lv in range(max_lv):
        g = [r["per_level_geom_reject"][lv]   for r in results if len(r["per_level_geom_reject"]) > lv]
        e = [r["per_level_energy_reject"][lv]  for r in results if len(r["per_level_energy_reject"]) > lv]
        a = [r["per_level_accept_rate"][lv]    for r in results if len(r["per_level_accept_rate"]) > lv]
        lv_geom.append(float(np.mean(g)) if g else float("nan"))
        lv_ener.append(float(np.mean(e)) if e else float("nan"))
        lv_acpt.append(float(np.mean(a)) if a else float("nan"))

    return dict(target=target_name, integ=integ_name,
                vf_per_proposal=vf_per_proposal,
                pf_vals=pf_vals, mean_pf=mean_pf, std_pf=std_pf,
                cov_pf=cov_pf, mean_levels=mean_levels, est_vf_per_run=est_vf,
                level_geom_reject=lv_geom,
                level_energy_reject=lv_ener,
                level_accept_rate=lv_acpt)


# ── Tables ─────────────────────────────────────────────────────────────────────

def print_main_table(all_results: List[Dict]) -> None:
    hdr = f"{'Config':<22} {'Target':<10} {'mean_Pf':>10} {'Pf_ref':>10} {'COV':>6} {'est_vf/run':>12} {'lv':>4}"
    sep = "─" * len(hdr)
    _log("\n" + sep)
    _log(hdr)
    _log(sep)
    for r in all_results:
        _log(f"{r['integ']:<22} {r['target']:<10} "
             f"{r['mean_pf']:>10.4e} {PF_REF:>10.4e} {r['cov_pf']:>6.3f} "
             f"{r['est_vf_per_run']:>12.3e} {r['mean_levels']:>4.1f}")
    _log(sep)


def print_mechanism_table(all_results: List[Dict]) -> None:
    _log("\nMechanism table (mean frac per conditional level, averaged over reps):")
    for r in all_results:
        _log(f"\n  {r['integ']} / {r['target']}:")
        _log(f"  {'lv':>3}  {'geom_rej':>9}  {'energy_rej':>11}  {'accept':>7}")
        for lv, (g, e, a) in enumerate(zip(
            r["level_geom_reject"], r["level_energy_reject"], r["level_accept_rate"]
        )):
            _log(f"  L{lv+1:>1}  {g:>9.3f}  {e:>11.3f}  {a:>7.3f}")


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_pf_distribution(all_results: List[Dict], out_dir: Path, tag: str) -> None:
    targets = sorted(set(r["target"] for r in all_results))
    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 5))
    if len(targets) == 1:
        axes = [axes]
    for ax, tgt in zip(axes, targets):
        sub    = [r for r in all_results if r["target"] == tgt]
        data   = [r["pf_vals"] for r in sub]
        labels = [r["integ"]   for r in sub]
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        ax.axhline(PF_REF, color="red", ls="--", lw=1.2, label=f"Pf_ref={PF_REF:.2e}")
        ax.set_yscale("log"); ax.set_title(f"Pf distribution — {tgt}")
        ax.set_ylabel("Estimated P_F"); ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"sus_pf_distribution{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    _log(f"  Saved {path.name}")


def plot_mechanism_bars(all_results: List[Dict], out_dir: Path, tag: str) -> None:
    targets = sorted(set(r["target"] for r in all_results))
    integrs = sorted(set(r["integ"]  for r in all_results))
    fig, axes = plt.subplots(len(targets), len(integrs),
                             figsize=(5 * len(integrs), 4 * len(targets)), squeeze=False)
    for row, tgt in enumerate(targets):
        for col, integ in enumerate(integrs):
            ax  = axes[row][col]
            sub = next((r for r in all_results if r["target"] == tgt and r["integ"] == integ), None)
            if sub is None:
                ax.set_visible(False); continue
            x    = np.arange(len(sub["level_geom_reject"]))
            geom = sub["level_geom_reject"]
            ener = sub["level_energy_reject"]
            acpt = sub["level_accept_rate"]
            ax.bar(x, geom, label="geom reject", color="#DD8452", alpha=0.85)
            ax.bar(x, ener, bottom=geom, label="energy reject", color="#4C72B0", alpha=0.85)
            ax.bar(x, acpt, bottom=[g + e for g, e in zip(geom, ener)],
                   label="accepted", color="#55A868", alpha=0.85)
            ax.set_xticks(x); ax.set_xticklabels([f"L{i+1}" for i in x])
            ax.set_ylim(0, 1.05); ax.set_title(f"{tgt} / {integ}", fontsize=9)
            ax.set_ylabel("fraction")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"sus_mechanism_bars{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    _log(f"  Saved {path.name}")


# ── Record ─────────────────────────────────────────────────────────────────────

def append_record(all_results: List[Dict], n_rep: int) -> None:
    lines = [
        "", "---", "",
        f"## Stage 2 — SuS integrator comparison (leapfrog+hnn vs yoshida+hnn, N_rep={n_rep})",
        "",
        f"N={SUS_N}, p0={SUS_P0}, β={BETA}, Pf_ref={PF_REF:.4e}",
        f"step_size={STEP_SIZE}, n_steps={N_STEPS}, VF_lf={VF_LEAPFROG}, VF_y4={VF_YOSHIDA}",
        "",
        "### Main table",
        "",
        "| Config | Target | mean_Pf | COV | est_vf/run | mean_lv |",
        "|--------|--------|---------|-----|------------|---------|",
    ]
    for r in all_results:
        lines.append(f"| {r['integ']} | {r['target']} | {r['mean_pf']:.4e} "
                     f"| {r['cov_pf']:.3f} | {r['est_vf_per_run']:.3e} | {r['mean_levels']:.2f} |")
    lines += ["", "### Mechanism", ""]
    for r in all_results:
        lines += [f"**{r['integ']} / {r['target']}**", "",
                  "| lv | geom_rej | energy_rej | accept |",
                  "|----|----------|------------|--------|"]
        for lv, (g, e, a) in enumerate(zip(
            r["level_geom_reject"], r["level_energy_reject"], r["level_accept_rate"]
        )):
            lines.append(f"| L{lv+1} | {g:.3f} | {e:.3f} | {a:.3f} |")
        lines.append("")
    with open(REC_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _log(f"  Appended Stage 2 to {REC_PATH.name}")


# ── Main ────────────────────────────────────────────────────────────────────────

def main(n_rep: int = 2) -> None:
    _open_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if n_rep <= 5 else ""

    _log("=" * 60)
    _log(f"Stage 2 SuS — N_rep={n_rep}  tag={tag or 'full'}")
    _log(f"step_size={STEP_SIZE}, n_steps={N_STEPS}, SUS_N={SUS_N}, burn_in={SUS_BURN_IN}")
    _log("=" * 60)

    device = get_device()
    _log(f"Device: {device}")

    _log("\n[Loading HNN models]")
    hnn_gaussian = _load_hnn("gaussian", device)
    hnn_banana   = _load_hnn("banana",   device)
    _log("Models loaded.")

    gaussian_target = GaussianTarget()
    banana_target   = BananaTarget(BananaTargetConfig(b=0.15, sigma1=1.0, sigma2=1.0))

    configs = [
        ("gaussian", gaussian_target, hnn_gaussian, "leapfrog+hnn",
         make_integrate_leapfrog_hnn(hnn_gaussian, STEP_SIZE, N_STEPS), VF_LEAPFROG),
        ("gaussian", gaussian_target, hnn_gaussian, "yoshida+hnn",
         make_integrate_yoshida_hnn(hnn_gaussian, STEP_SIZE, N_STEPS), VF_YOSHIDA),
        ("banana",   banana_target,   hnn_banana,   "leapfrog+hnn",
         make_integrate_leapfrog_hnn(hnn_banana,   STEP_SIZE, N_STEPS), VF_LEAPFROG),
        ("banana",   banana_target,   hnn_banana,   "yoshida+hnn",
         make_integrate_yoshida_hnn(hnn_banana,   STEP_SIZE, N_STEPS), VF_YOSHIDA),
    ]

    _log(f"\n[Running {n_rep} reps × {len(configs)} configs]")
    t_start = time.perf_counter()
    all_results: List[Dict] = []
    for (tgt_name, target, _hnn, integ_name, integrate_fn, vf) in configs:
        res = run_replications(tgt_name, target, integrate_fn,
                               integ_name, vf, device, n_rep)
        all_results.append(res)

    _log(f"\nTotal wall time: {time.perf_counter()-t_start:.1f}s")

    print_main_table(all_results)
    print_mechanism_table(all_results)

    _log("\n[Saving outputs]")
    plot_pf_distribution(all_results, OUT_DIR, tag)
    plot_mechanism_bars(all_results, OUT_DIR, tag)
    torch.save(all_results, OUT_DIR / f"sus_stage2_results{tag}.pt")
    _log(f"  Saved sus_stage2_results{tag}.pt")

    if n_rep > 5:
        append_record(all_results, n_rep)

    _log(f"\nDone. All outputs in {OUT_DIR}")
    _close_log()


if __name__ == "__main__":
    full = "--full" in sys.argv
    main(n_rep=30 if full else 2)
