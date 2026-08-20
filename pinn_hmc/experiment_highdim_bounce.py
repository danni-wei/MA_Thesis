"""
experiment_highdim_bounce.py
─────────────────────────────────────────────────────────────────────────────
Extension of experiment_geometry_bounce.py to higher dimensions.

Scope (per plan, staged):
  Stage 1 — dimension threading sanity check (rho=0 only; bugs #1/#2 in
            experiment_geometry_bounce.py, both correlation-related, are
            untouched and don't trigger at rho=0).
  Stage 2 — Papaioannou 2015 Example 3 quadratic LSF.
  Stage 3 — Arm A vs Arm C smoke comparison across d.
  Stage 4 — full n_rep=30 run with raw-data + record outputs.

Pure bouncing only: Arm A (fixed leapfrog baseline) vs Arm C (geometry-aware
adaptive step + barrier bounce). Arm B and the HNN path are intentionally
excluded. Analytic gradients only.

Run from project root:
    python -u pinn_hmc/experiment_highdim_bounce.py --stage1
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

OUT_DIR = Path("results/highdim_bounce")

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm as Norm

from pinn_hmc.experiment_geometry_bounce import (
    ThalerLinearLSF,
    CorrelatedGaussianTarget,
    ControllerConfig,
    run_exp1_linear_lsf,
    run_replications,
)


class QuadraticLSF:
    """
    Papaioannou et al. 2015, Example 3: quadratic limit-state function.

        g3(u) = beta - (1/sqrt(d)) * sum(u_i) + (kappa/4) * (u_1 - u_2)^2

    Failure domain: {g3 <= 0}.

    Analytic Pf via a reduction to two independent standard normals:
    S = sum(u_i)/sqrt(d) ~ N(0,1) and Y = (u_1-u_2)/sqrt(2) ~ N(0,1) are
    uncorrelated linear combinations of the iid u_i, hence (jointly Gaussian)
    independent. Substituting (u_1-u_2)^2 = 2*Y^2:

        g3 = beta - S + (kappa/2) * Y^2 <= 0  <=>  S >= beta + (kappa/2) Y^2

        Pf = E_Y[ Phi(-(beta + (kappa/2) Y^2)) ]
           = integral_{-inf}^{inf} Phi(-(beta + (kappa/2) y^2)) * phi(y) dy

    evaluated here by 1-D numerical quadrature (equivalent to Papaioannou's
    Eq. 34 double integral, reduced analytically by the S/Y independence
    above before integrating).
    """
    def __init__(self, beta: float, kappa: float, d: int = 2):
        if d < 2:
            raise ValueError("QuadraticLSF requires d >= 2 (uses u_1, u_2)")
        self.beta  = float(beta)
        self.kappa = float(kappa)
        self.d     = d
        self._inv_sqrt_d = 1.0 / math.sqrt(d)

        self.pf_ref = self._compute_pf_ref()
        pf_clamped  = max(self.pf_ref, 1e-300)
        self.beta_ref = float(-Norm.ppf(pf_clamped))

    def _compute_pf_ref(self) -> float:
        beta, kappa = self.beta, self.kappa
        integrand = lambda y: float(Norm.cdf(-(beta + 0.5 * kappa * y * y)) * Norm.pdf(y))
        val, _err = quad(integrand, -12.0, 12.0, limit=400)
        return float(val)

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        s    = q.sum(axis=-1) * self._inv_sqrt_d
        diff = q[..., 0] - q[..., 1]
        return self.beta - s + 0.25 * self.kappa * diff * diff

    def grad_G(self, q: np.ndarray) -> np.ndarray:
        diff = q[..., 0] - q[..., 1]
        g = np.full_like(q, -self._inv_sqrt_d)
        g[..., 0] = g[..., 0] + 0.5 * self.kappa * diff
        g[..., 1] = g[..., 1] - 0.5 * self.kappa * diff
        return g

    def is_failure(self, q: np.ndarray) -> np.ndarray:
        return self.evaluate(q) <= 0.0


def solve_quadratic_beta(kappa: float, target_pf: float,
                          lo: float = 0.0, hi: float = 8.0) -> float:
    """
    Root-find beta such that QuadraticLSF(beta, kappa).pf_ref == target_pf.

    Note: pf_ref for QuadraticLSF does not depend on d (S = sum(u_i)/sqrt(d)
    is N(0,1) for any d), so this is solved once and reused across d.
    """
    def f(beta: float) -> float:
        return QuadraticLSF(beta=beta, kappa=kappa, d=2).pf_ref - target_pf
    return float(brentq(f, lo, hi, xtol=1e-10))


def make_ctrl_cfg() -> ControllerConfig:
    """Identical to the ControllerConfig used in main() for the Chapter 6
    beta=3.5 rho=0 three-arm study — kept fixed for comparability."""
    return ControllerConfig(
        c_cfl=0.5, eps_E=0.05, dt_min=1e-3, dt_max=0.5,
        dt_max_change=5.0, n_bounce_max=5, tol_hit=1e-6, bisect_max_iter=30,
        bounce_fixed_step=False,
    )


def stage1(beta: float = 3.5, dims=(2, 10, 50, 100)) -> None:
    print("=" * 70)
    print("STAGE 1 — dimension threading (rho=0)")
    print("=" * 70)

    print(f"\nbeta={beta}  Phi(-beta) [scipy, independent check] = "
          f"{float(Norm.cdf(-beta)):.6e}")
    print(f"\n{'d':>5}  {'sigma_max':>9}  {'G_offset':>10}  "
          f"{'G_offset/sqrt(d)':>16}  {'pf_ref':>12}  {'matches Phi(-beta)?':>20}")

    ref = float(Norm.cdf(-beta))
    for d in dims:
        lsf = ThalerLinearLSF(beta=beta, rho=0.0, d=d)
        offset_over_sqrtd = lsf.G_offset / np.sqrt(d)
        match = abs(lsf.pf_ref - ref) < 1e-12 and abs(offset_over_sqrtd - beta) < 1e-9
        print(f"{d:>5}  {lsf.sigma_max:>9.4f}  {lsf.G_offset:>10.4f}  "
              f"{offset_over_sqrtd:>16.6f}  {lsf.pf_ref:>12.6e}  {str(match):>20}")

    print("\nRunning smoke test: Arm A only, d=10, n_rep=1 ...")
    ctrl_cfg = make_ctrl_cfg()
    result = run_exp1_linear_lsf(
        beta=beta, rho=0.0, n_rep=1, ctrl_cfg=ctrl_cfg, arms=("A",), d=10,
    )
    r = result["arms"]["A"]
    print(f"\n  d=10  Arm A  n_rep=1")
    print(f"  beta_hat   = {r['mean_beta']:.4f}   (beta_ref = {beta})")
    print(f"  P_F        = {r['mean_pf']:.4e}     (pf_ref   = {ref:.4e})")
    print(f"  levels     = {r['mean_levels']:.2f}")
    print(f"  bias_beta  = {r['bias_beta']:+.4f}")
    print("\nStage 1 complete — no errors.")


def stage2(beta: float = 3.5, kappa: float = 5.0, d: int = 10) -> None:
    print("=" * 70)
    print("STAGE 2 — quadratic LSF (Papaioannou 2015, Example 3)")
    print("=" * 70)

    lsf = QuadraticLSF(beta=beta, kappa=kappa, d=d)
    print(f"\nbeta={beta}  kappa={kappa}  d={d}")
    print(f"  pf_ref (quadrature)          = {lsf.pf_ref:.6e}")
    print(f"  beta_ref = -Phi^-1(pf_ref)   = {lsf.beta_ref:.4f}")
    print(f"  [for reference, linear-only Phi(-beta) would be "
          f"{float(Norm.cdf(-beta)):.6e} — the kappa term visibly shifts Pf]")

    print(f"\nRunning smoke test: Arm A only, d={d}, kappa={kappa}, n_rep=1 ...")
    ctrl_cfg = make_ctrl_cfg()
    target   = CorrelatedGaussianTarget(rho=0.0, d=d)
    result   = run_replications(
        "A", target, lsf, ctrl_cfg, n_rep=1,
        label=f"quad_beta={beta}_kappa={kappa}_d={d}",
    )
    print(f"\n  d={d}  Arm A  n_rep=1  (quadratic LSF)")
    print(f"  beta_hat   = {result['mean_beta']:.4f}   (beta_ref = {lsf.beta_ref:.4f})")
    print(f"  P_F        = {result['mean_pf']:.4e}     (pf_ref   = {lsf.pf_ref:.4e})")
    print(f"  levels     = {result['mean_levels']:.2f}")
    print(f"  bias_beta  = {result['bias_beta']:+.4f}")
    print("\nStage 2 complete — no errors.")


def stage3(n_rep: int = 3) -> None:
    print("=" * 70)
    print("STAGE 3 — Arm A vs Arm C smoke comparison")
    print("=" * 70)

    ctrl_cfg = make_ctrl_cfg()
    beta_lin = 3.5
    pf_target = float(Norm.cdf(-beta_lin))
    kappa = 5.0
    beta_quad = solve_quadratic_beta(kappa=kappa, target_pf=pf_target)
    print(f"\nLinear LSF target: beta={beta_lin}  Pf={pf_target:.4e}")
    print(f"Quadratic LSF beta solved for matching Pf: beta={beta_quad:.4f}  "
          f"kappa={kappa}  (pf_ref recomputed below)")

    combos = []
    for d in (10, 50, 100):
        lsf    = ThalerLinearLSF(beta=beta_lin, rho=0.0, d=d)
        target = CorrelatedGaussianTarget(rho=0.0, d=d)
        combos.append((f"linear d={d}", d, target, lsf))

    d_quad = 100
    lsf_quad    = QuadraticLSF(beta=beta_quad, kappa=kappa, d=d_quad)
    target_quad = CorrelatedGaussianTarget(rho=0.0, d=d_quad)
    combos.append((f"quadratic d={d_quad} kappa={kappa}", d_quad, target_quad, lsf_quad))

    rows = []
    for label, d, target, lsf in combos:
        print(f"\n--- {label}  (pf_ref={lsf.pf_ref:.4e}, beta_ref={lsf.beta_ref:.4f}) ---")
        for arm in ("A", "C"):
            r = run_replications(arm, target, lsf, ctrl_cfg, n_rep,
                                  label=f"{label}_{arm}")
            geom_rej = r["agg"]["geom_rej"]
            rows.append(dict(
                lsf=label, arm=arm,
                beta_hat=r["mean_beta"], std_beta=r["std_beta"],
                beta_ref=lsf.beta_ref, cov_beta=r["cov_beta"],
                mean_levels=r["mean_levels"],
                geom_rej=geom_rej,
            ))

    print("\n" + "=" * 100)
    print("SMOKE TABLE")
    print("=" * 100)
    print(f"{'LSF':<26}{'arm':>4}{'beta_hat':>10}{'beta_ref':>10}"
          f"{'COV_beta':>10}{'mean_lv':>9}   per-level geom_rej")
    for row in rows:
        gr = ", ".join(f"{g:.2f}" for g in row["geom_rej"])
        print(f"{row['lsf']:<26}{row['arm']:>4}{row['beta_hat']:>10.4f}"
              f"{row['beta_ref']:>10.4f}{row['cov_beta']:>10.3f}"
              f"{row['mean_levels']:>9.2f}   [{gr}]")

    print("\nStage 3 complete — no errors.")


def _pad_stack(list_of_lists) -> np.ndarray:
    """Stack per-rep per-level lists of possibly differing length into a
    [n_rep, max_levels] array, padding short reps (early SuS termination)
    with NaN."""
    max_len = max(len(l) for l in list_of_lists)
    out = np.full((len(list_of_lists), max_len), np.nan, dtype=np.float64)
    for i, l in enumerate(list_of_lists):
        out[i, :len(l)] = l
    return out


def _make_combos(beta_lin: float, kappa: float, beta_quad: float):
    combos = []
    for d in (10, 50, 100):
        lsf    = ThalerLinearLSF(beta=beta_lin, rho=0.0, d=d)
        target = CorrelatedGaussianTarget(rho=0.0, d=d)
        combos.append((f"linear_d{d}", d, target, lsf))
    d_quad = 100
    lsf_quad    = QuadraticLSF(beta=beta_quad, kappa=kappa, d=d_quad)
    target_quad = CorrelatedGaussianTarget(rho=0.0, d=d_quad)
    combos.append((f"quadratic_d{d_quad}_kappa{kappa}", d_quad, target_quad, lsf_quad))
    return combos


def stage4(n_rep: int = 30) -> None:
    print("=" * 70)
    print(f"STAGE 4 — full run  n_rep={n_rep}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctrl_cfg = make_ctrl_cfg()

    beta_lin  = 3.5
    pf_target = float(Norm.cdf(-beta_lin))
    kappa     = 5.0
    beta_quad = solve_quadratic_beta(kappa=kappa, target_pf=pf_target)
    print(f"\nLinear LSF: beta={beta_lin}  Pf_target={pf_target:.4e}")
    print(f"Quadratic LSF: beta={beta_quad:.4f}  kappa={kappa}  (matched Pf)")

    combos = _make_combos(beta_lin, kappa, beta_quad)

    npz_payload = {}
    summary_rows = []   # for markdown record

    for label, d, target, lsf in combos:
        print(f"\n{'='*70}\n{label}  (pf_ref={lsf.pf_ref:.4e}, beta_ref={lsf.beta_ref:.4f})\n{'='*70}")
        arm_results = {}
        for arm in ("A", "C"):
            r = run_replications(arm, target, lsf, ctrl_cfg, n_rep,
                                  label=f"{label}_{arm}")
            arm_results[arm] = r

            per_rep = r["results"]
            geom_rej_raw     = _pad_stack([rr["per_level_geom_rej"]     for rr in per_rep])
            accept_rate_raw  = _pad_stack([rr["per_level_accept_rate"]  for rr in per_rep])
            bounces_raw      = _pad_stack([rr["per_level_n_bounces_mean"] for rr in per_rep])

            npz_payload[f"{label}_{arm}_beta"]        = r["beta_arr"]
            npz_payload[f"{label}_{arm}_pf"]          = r["pf_arr"]
            npz_payload[f"{label}_{arm}_levels"]      = np.array([rr["levels"] for rr in per_rep])
            npz_payload[f"{label}_{arm}_geom_rej"]    = geom_rej_raw
            npz_payload[f"{label}_{arm}_accept_rate"] = accept_rate_raw
            npz_payload[f"{label}_{arm}_bounces"]     = bounces_raw

        summary_rows.append(dict(label=label, d=d, lsf=lsf, arms=arm_results))

    npz_path = OUT_DIR / "highdim_bounce_raw.npz"
    np.savez(npz_path, **npz_payload)
    print(f"\nSaved raw data: {npz_path}")

    # ── Markdown record ─────────────────────────────────────────────────
    lines = [
        "# High-Dimensional Barrier Bouncing: Arm A vs Arm C",
        "",
        "Pure bouncing comparison (Arm A fixed leapfrog vs Arm C geometry-aware "
        "adaptive step + barrier bounce). Analytic gradients only; Arm B and the "
        "HNN path excluded. rho=0 (independent standard Gaussian, "
        "Papaioannou 2015 Example 1 setting).",
        "",
        "## Controller settings (identical to Chapter 6 beta=3.5 rho=0 study)",
        "",
        "| param | value |",
        "|-------|-------|",
        f"| c_cfl | {ctrl_cfg.c_cfl} |",
        f"| eps_E | {ctrl_cfg.eps_E} |",
        f"| tau_safety | {ctrl_cfg.tau_safety} |",
        f"| dt_min | {ctrl_cfg.dt_min} |",
        f"| dt_max | {ctrl_cfg.dt_max} |",
        f"| dt_max_change | {ctrl_cfg.dt_max_change} |",
        f"| n_bounce_max | {ctrl_cfg.n_bounce_max} |",
        f"| tol_hit | {ctrl_cfg.tol_hit} |",
        f"| bisect_max_iter | {ctrl_cfg.bisect_max_iter} |",
        f"| N (SUS_N) | 1000 |  p0 (SUS_P0) | 0.1 |",
        f"| n_rep | {n_rep} |",
        "",
        f"Linear LSF: beta={beta_lin}, Pf_ref=Phi(-beta)={pf_target:.4e} (all d).",
        f"Quadratic LSF (Papaioannou Ex. 3): beta={beta_quad:.4f}, kappa={kappa}, "
        f"matched to the same target Pf via quadrature root-find.",
        "",
        "## Summary",
        "",
        "| LSF | d | arm | beta_hat | std | bias | COV_beta | mean_lv |",
        "|-----|---|-----|----------|-----|------|----------|---------|",
    ]
    for row in summary_rows:
        for arm, r in row["arms"].items():
            lines.append(
                f"| {row['label']} | {row['d']} | {arm} | {r['mean_beta']:.4f} | "
                f"{r['std_beta']:.4f} | {r['bias_beta']:+.4f} | {r['cov_beta']:.4f} | "
                f"{r['mean_levels']:.2f} |"
            )

    lines += ["", "## COV reduction: Arm C vs Arm A", "",
              "| LSF | d | COV_A | COV_C | reduction (1 - COV_C/COV_A) |",
              "|-----|---|-------|-------|------------------------------|"]
    for row in summary_rows:
        cov_a = row["arms"]["A"]["cov_beta"]
        cov_c = row["arms"]["C"]["cov_beta"]
        reduction = 1.0 - (cov_c / cov_a) if cov_a not in (0.0, float("nan")) else float("nan")
        lines.append(f"| {row['label']} | {row['d']} | {cov_a:.4f} | {cov_c:.4f} | "
                      f"{reduction:+.3f} |")

    lines += ["", "## Per-level mechanism (mean across reps)", ""]
    for row in summary_rows:
        lines.append(f"### {row['label']}")
        lines.append("")
        for arm, r in row["arms"].items():
            agg = r["agg"]
            lines += [
                f"**Arm {arm}**",
                "",
                "| lv | geom_rej | accept_rate | n_bounces_mean |",
                "|----|----------|-------------|----------------|",
            ]
            for lv in range(len(agg["geom_rej"])):
                lines.append(
                    f"| L{lv+1} | {agg['geom_rej'][lv]:.4f} | "
                    f"{agg['accept_rate'][lv]:.4f} | {agg['n_bounces_mean'][lv]:.3f} |"
                )
            lines.append("")

    record_path = OUT_DIR / "record_highdim_bounce.md"
    with open(record_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved record: {record_path}")

    print("\nStage 4 complete — no errors.")


if __name__ == "__main__":
    if "--stage1" in sys.argv:
        stage1()
    if "--stage2" in sys.argv:
        stage2()
    if "--stage3" in sys.argv:
        stage3()
    if "--stage4" in sys.argv:
        stage4()
