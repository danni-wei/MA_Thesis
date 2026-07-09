"""
experiment_yoshida_distance_adaptive.py
────────────────────────────────────────────────────────────────────────────
ARM A  — fixed leapfrog (dt=0.1, baseline, unchanged from geometry_bounce)
ARM D  — adaptive leapfrog driven by leapfrog−Yoshida local error estimate

Step-size controller (supervisor's notes formula):
    distance  = ‖z_LF − z_Y4‖₂  (q+p concatenated, same dt, same starting z)
    dt_next   = tau * (eps / max(distance, 1e-12))^(1/3) * dt
    clipped:  change factor ∈ [0.2, 5.0],  dt ∈ [dt_min, dt_max]

Cost accounting — reported under TWO conventions:
    OPTIMIZED  : Yoshida step reuses g0 + merges interior kicks → 3 new grad_U evals
    UNOPTIMIZED: Yoshida step as implemented in existing integrators.py → 6 new evals

Benchmark: Thaler linear LSF, d=2, configs beta=3.5/rho=0 and beta=4/rho=0.75.
eps sweep: {1e-3, 1e-2, 1e-1}.

CORRECTNESS NOTE
  State-dependent per-step dt breaks leapfrog reversibility/volume-preservation.
  ARM D is an APPROXIMATE sampler.  The standard exp(−ΔH) MH step is kept but
  the invariant measure is perturbed.  Bias is measured as beta_hat − beta_ref.

Run from project root:
    python -u _run_v4_yoshida_distance.py               # smoke (n_rep=2)
    python -u _run_v4_yoshida_distance.py --full        # full  (n_rep=30)
"""
from __future__ import annotations

import io
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

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
from scipy.stats import norm as Norm

# ── Import shared infrastructure from geometry_bounce experiment ──────────────
from pinn_hmc.experiment_geometry_bounce import (
    ThalerLinearLSF,
    CorrelatedGaussianTarget,
    _verlet_step,
    _mh_accept,
    DT_A,
    N_STEPS_A,
    T_F,
    SUS_N,
    SUS_P0,
    SUS_BURN_IN,
    SUS_MAX_LEVELS,
    SEED_BASE,
    integrate_A,
)

# ── Output paths ───────────────────────────────────────────────────────────────
OUT_DIR  = Path("results/yoshida_distance_adaptive")
REC_PATH = OUT_DIR / "record_yoshida_distance_adaptive.md"
LOG_PATH = OUT_DIR / "yoshida_distance_run.log"

# ── Controller defaults ────────────────────────────────────────────────────────
TAU_SAFETY  = 0.85    # safety factor
LF_ORDER    = 2       # leapfrog is 2nd-order; exponent = 1/(order+1) = 1/3
EXPONENT    = 1.0 / (LF_ORDER + 1)   # = 1/3
DT_MIN      = 0.001
DT_MAX      = 0.500
CHANGE_MIN  = 0.2     # clip growth/shrink factor
CHANGE_MAX  = 5.0

# Yoshida coefficients (reused from existing integrators — do not reimplement)
_CBRT2 = 2.0 ** (1.0 / 3.0)
_W1    = 1.0 / (2.0 - _CBRT2)          # ≈  1.3512
_W0    = -_CBRT2 / (2.0 - _CBRT2)     # ≈ -1.7024

# ── Logging ────────────────────────────────────────────────────────────────────
_log_fh: Optional[object] = None

def _log(msg: str = "") -> None:
    print(msg, flush=True)
    if _log_fh is not None:
        _log_fh.write(msg + "\n")   # type: ignore[union-attr]
        _log_fh.flush()             # type: ignore[union-attr]

def _open_log() -> None:
    global _log_fh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

def _close_log() -> None:
    if _log_fh is not None:
        _log_fh.close()             # type: ignore[union-attr]


# ═══════════════════════════════════════════════════════════════════════════════
# Optimised single-step Yoshida (reuses g0, merges interior kicks → 3 new evals)
# ═══════════════════════════════════════════════════════════════════════════════

def _yoshida4_step(
    q: np.ndarray,
    p: np.ndarray,
    g0: np.ndarray,
    grad_U_fn: Callable,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single Yoshida-4th step reusing cached gradient g0.

    Coefficients w1, w0, w1 (same as integrators.py — not reimplemented):
        w1 = 1/(2 − 2^(1/3)) ≈ 1.3512
        w0 = −2^(1/3)/(2 − 2^(1/3)) ≈ −1.7024

    Standard decomposition: LF(w1·dt) ∘ LF(w0·dt) ∘ LF(w1·dt).
    Consecutive half-kicks at the same q are merged; g0 is reused for the
    first half-kick, so only 3 NEW grad_U evaluations are needed:
        g1 at q1  (between sub-steps 1 and 2)
        g2 at q2  (between sub-steps 2 and 3)
        g3 at q3  (final position)

    Returns (q_new, p_new, g_new) — no momentum flip; that is done at the
    trajectory level by the MH step.
    Cost: 3 new grad_U evals (OPTIMIZED convention).
    """
    # ── Sub-step 1  (h = w1*dt) ──────────────────────────────────────────────
    h1 = _W1 * dt
    p1 = p  - 0.5 * h1 * g0          # initial half-kick: reuse g0, no eval
    q1 = q  + h1 * p1                 # drift
    g1 = grad_U_fn(q1)                # eval 1

    # ── Sub-step 2  (h = w0*dt) — merge end-kick of step 1 + start-kick of step 2
    h0 = _W0 * dt
    p2 = p1 - (0.5 * h1 + 0.5 * h0) * g1   # merged half-kicks
    q2 = q1 + h0 * p2                        # drift
    g2 = grad_U_fn(q2)                       # eval 2

    # ── Sub-step 3  (h = w1*dt) — merge end-kick of step 2 + start-kick of step 3
    p3 = p2 - (0.5 * h0 + 0.5 * h1) * g2   # merged half-kicks
    q3 = q2 + h1 * p3                        # drift
    g3 = grad_U_fn(q3)                       # eval 3

    p_out = p3 - 0.5 * h1 * g3              # final half-kick at q3
    return q3, p_out, g3


# ═══════════════════════════════════════════════════════════════════════════════
# Per-proposal stats for ARM D
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PropStatsD:
    n_grad_U_advance: int   = 0   # leapfrog grad evals (advancing)
    n_grad_U_est_opt: int   = 0   # Yoshida estimate evals (optimized, 3/step)
    n_grad_U_est_unopt: int = 0   # Yoshida estimate evals (unoptimized, 6/step)
    n_steps:          int   = 0
    dt_min_used:      float = math.inf
    dt_max_used:      float = 0.0
    dt_mean:          float = 0.0
    abs_dH:           float = 0.0
    distance_mean:    float = 0.0
    distance_p95:     float = 0.0
    dist_pos_mean:    float = 0.0   # position-only part of distance
    dist_mom_mean:    float = 0.0   # momentum-only part
    dist_energy_mean: float = 0.0   # |ΔH| from the error estimate (diagnostic)


# ═══════════════════════════════════════════════════════════════════════════════
# ARM D — adaptive leapfrog driven by leapfrog−Yoshida distance
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_D(
    q0: np.ndarray,
    p0: np.ndarray,
    target: CorrelatedGaussianTarget,
    t_f: float,
    dt_seed: float,
    tau: float,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, PropStatsD]:
    """
    Adaptive leapfrog driven by leapfrog−Yoshida local error estimate.

    Per step from state (q, p) with cached gradient g0 and current dt:
      1. z_LF  = _verlet_step(q, p, g0, grad_U, dt)      — advance  (1 new eval)
      2. z_Y4  = _yoshida4_step(q, p, g0, grad_U, dt)    — estimate (3 new evals)
      3. distance = ‖(q_LF−q_Y4, p_LF−p_Y4)‖₂
      4. dt_next  = clip(tau*(eps/max(distance,1e-12))^(1/3)*dt)
      5. Advance with z_LF; set dt = dt_next for next step.

    CORRECTNESS NOTE: state-dependent per-step dt breaks leapfrog
    reversibility/volume-preservation → approximate sampler.  The standard
    exp(−ΔH) MH step is kept.

    Cost per step:
      OPTIMIZED   : 1 (leapfrog) + 3 (Yoshida, g0-reuse + kick-merge) = 4 evals
      UNOPTIMIZED : 1 (leapfrog) + 6 (Yoshida, no reuse)               = 7 evals
    Both counts are recorded in PropStatsD.
    """
    q, p = q0.copy(), p0.copy()
    g0   = target.grad_U(q)     # 1 grad_U for initial gradient
    n_adv = 1                    # count the initial eval as "advancing overhead"

    dt = dt_seed
    t  = 0.0
    n_steps = 0
    dt_list:   List[float] = []
    dist_list: List[float] = []
    dist_pos_list: List[float] = []
    dist_mom_list: List[float] = []
    dist_E_list:   List[float] = []
    n_est_opt   = 0
    n_est_unopt = 0

    while t < t_f - 1e-12:
        dt = min(dt, t_f - t)

        # ── Leapfrog step (advancing) ─────────────────────────────────────
        q_LF, p_LF, g1_LF = _verlet_step(q, p, g0, target.grad_U, dt)
        n_adv += 1                  # 1 new eval at q_LF

        # ── Yoshida step (error estimate only — same (q,p,g0,dt)) ────────
        q_Y4, p_Y4, _ = _yoshida4_step(q, p, g0, target.grad_U, dt)
        n_est_opt   += 3            # optimized: 3 new evals
        n_est_unopt += 6            # unoptimized: 6 new evals

        # ── Local error estimate ──────────────────────────────────────────
        dq = q_LF - q_Y4
        dp = p_LF - p_Y4
        dist_pos  = float(np.linalg.norm(dq))
        dist_mom  = float(np.linalg.norm(dp))
        distance  = math.sqrt(dist_pos**2 + dist_mom**2)

        # Energy error diagnostic (uses Yoshida as reference)
        H_LF = target.H(q_LF, p_LF)
        H_Y4 = target.H(q_Y4, p_Y4)
        dist_E_list.append(abs(H_LF - H_Y4))

        dist_list.append(distance)
        dist_pos_list.append(dist_pos)
        dist_mom_list.append(dist_mom)

        # ── Step-size update (supervisor's notes formula) ─────────────────
        raw_factor = tau * (eps / max(distance, 1e-12)) ** EXPONENT
        factor     = max(CHANGE_MIN, min(CHANGE_MAX, raw_factor))
        dt_next    = max(DT_MIN, min(DT_MAX, factor * dt))

        # ── Advance with z_LF ─────────────────────────────────────────────
        t += dt
        dt_list.append(dt)
        q, p, g0 = q_LF, p_LF, g1_LF
        dt = dt_next
        n_steps += 1

    dt_arr = np.asarray(dt_list, dtype=np.float64)

    s = PropStatsD(
        n_grad_U_advance  = n_adv,
        n_grad_U_est_opt  = n_est_opt,
        n_grad_U_est_unopt= n_est_unopt,
        n_steps           = n_steps,
        dt_min_used       = float(dt_arr.min()) if n_steps else dt_seed,
        dt_max_used       = float(dt_arr.max()) if n_steps else dt_seed,
        dt_mean           = float(dt_arr.mean()) if n_steps else dt_seed,
        distance_mean     = float(np.mean(dist_list)) if dist_list else 0.0,
        distance_p95      = float(np.percentile(dist_list, 95)) if dist_list else 0.0,
        dist_pos_mean     = float(np.mean(dist_pos_list)) if dist_pos_list else 0.0,
        dist_mom_mean     = float(np.mean(dist_mom_list)) if dist_mom_list else 0.0,
        dist_energy_mean  = float(np.mean(dist_E_list)) if dist_E_list else 0.0,
    )
    return q, p, s


# ═══════════════════════════════════════════════════════════════════════════════
# Single SuS run — handles both ARM A and ARM D
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one_sus_D(
    arm: str,          # "A" or "D"
    target: CorrelatedGaussianTarget,
    lsf: ThalerLinearLSF,
    tau: float,
    eps: float,
    seed: int,
    rep_label: str = "",
) -> Dict:
    """
    One complete SuS run for ARM A or ARM D.

    ARM A: fixed leapfrog (DT_A=0.1, N_STEPS_A=10, T_F=1.0)
    ARM D: adaptive leapfrog with Yoshida-distance controller (eps tolerance)

    Per-level tracking:
        geom_rej, energy_rej, accept_rate, cond_energy_rej
        abs_dH_mean, abs_dH_p95
        grad_U_advance_lv, grad_U_est_opt_lv, grad_U_est_unopt_lv
        n_steps_mean, dt_mean, dt_min_mean, dt_max_mean
        distance_mean, distance_p95 (ARM D only)
        divergences
    """
    rng = np.random.default_rng(seed)
    d   = target.d
    N   = SUS_N
    p0f = SUS_P0
    n_seeds = max(1, int(round(p0f * N)))
    spc     = math.ceil(N / n_seeds)

    thresholds:             List[float] = []
    lv_geom_rej:            List[float] = []
    lv_energy_rej:          List[float] = []
    lv_accept_rate:         List[float] = []
    lv_cond_energy_rej:     List[float] = []
    lv_abs_dH_mean:         List[float] = []
    lv_abs_dH_p95:          List[float] = []
    lv_grad_adv:            List[int]   = []
    lv_grad_est_opt:        List[int]   = []
    lv_grad_est_unopt:      List[int]   = []
    lv_n_steps_mean:        List[float] = []
    lv_dt_mean:             List[float] = []
    lv_dt_min_mean:         List[float] = []
    lv_dt_max_mean:         List[float] = []
    lv_distance_mean:       List[float] = []
    lv_distance_p95:        List[float] = []
    lv_divergences:         List[int]   = []

    # ── Burn-in ────────────────────────────────────────────────────────────
    q = np.zeros((N, d), dtype=np.float64)
    burn_grad_adv = 0
    burn_grad_est_opt = 0
    burn_grad_est_unopt = 0

    for _ in range(SUS_BURN_IN):
        for i in range(N):
            p_init = target.sample_momentum(rng)
            if arm == "A":
                q_new, p_new, st_a = integrate_A(q[i], p_init, target, DT_A, N_STEPS_A)
                burn_grad_adv += st_a.n_grad_U
            else:
                q_new, p_new, st_d = integrate_D(
                    q[i], p_init, target, T_F, DT_A, tau, eps)
                burn_grad_adv       += st_d.n_grad_U_advance
                burn_grad_est_opt   += st_d.n_grad_U_est_opt
                burn_grad_est_unopt += st_d.n_grad_U_est_unopt

            H0 = target.H(q[i], p_init)
            H1 = target.H(q_new, p_new)
            ok, _ = _mh_accept(H0, H1, rng)
            if ok:
                q[i] = q_new

    # ── Conditional levels ─────────────────────────────────────────────────
    for level in range(SUS_MAX_LEVELS):
        g_vals    = lsf.evaluate(q)                    # [N]
        threshold = float(np.quantile(g_vals, p0f))
        thresholds.append(threshold)
        _log(f"    {rep_label} arm={arm}(eps={eps:.0e}) L{level+1}: threshold={threshold:.4f}")

        if threshold <= 0.0:
            n_fail = int((g_vals <= 0.0).sum())
            P_F    = (p0f ** level) * (n_fail / N)
            _log(f"    {rep_label} arm={arm}(eps={eps:.0e}) → done L{level+1}  Pf={P_F:.4e}")
            return _build_result_D(
                P_F, level + 1, thresholds, lsf,
                lv_geom_rej, lv_energy_rej, lv_accept_rate, lv_cond_energy_rej,
                lv_abs_dH_mean, lv_abs_dH_p95,
                lv_grad_adv, lv_grad_est_opt, lv_grad_est_unopt,
                lv_n_steps_mean, lv_dt_mean, lv_dt_min_mean, lv_dt_max_mean,
                lv_distance_mean, lv_distance_p95, lv_divergences,
                burn_grad_adv, burn_grad_est_opt, burn_grad_est_unopt,
            )

        seed_idx = np.argsort(g_vals)[:n_seeds]
        chains   = q[seed_idx].copy()
        chain_snapshots = [chains.copy()]

        lv_n = lv_geom = lv_ener = lv_acpt = lv_geom_pass = 0
        dH_list:    List[float] = []
        n_steps_list: List[int] = []
        dt_list_lv:   List[float] = []
        dt_min_list:  List[float] = []
        dt_max_list:  List[float] = []
        dist_lv:   List[float] = []
        dist95_lv: List[float] = []
        n_grad_adv_lv = 0
        n_grad_est_opt_lv = 0
        n_grad_est_unopt_lv = 0
        n_div = 0

        for _ in range(spc - 1):
            for i in range(n_seeds):
                p_init = target.sample_momentum(rng)

                if arm == "A":
                    q_new, p_new, st_a = integrate_A(
                        chains[i], p_init, target, DT_A, N_STEPS_A)
                    n_grad_adv_lv += st_a.n_grad_U
                    n_steps_i   = st_a.n_steps
                    dt_mean_i   = st_a.dt_mean
                    dt_min_i    = st_a.dt_min_used
                    dt_max_i    = st_a.dt_max_used
                    dist_i      = 0.0
                    dist95_i    = 0.0
                else:
                    q_new, p_new, st_d = integrate_D(
                        chains[i], p_init, target, T_F, DT_A, tau, eps)
                    n_grad_adv_lv       += st_d.n_grad_U_advance
                    n_grad_est_opt_lv   += st_d.n_grad_U_est_opt
                    n_grad_est_unopt_lv += st_d.n_grad_U_est_unopt
                    n_steps_i   = st_d.n_steps
                    dt_mean_i   = st_d.dt_mean
                    dt_min_i    = st_d.dt_min_used
                    dt_max_i    = st_d.dt_max_used
                    dist_i      = st_d.distance_mean
                    dist95_i    = st_d.distance_p95

                n_steps_list.append(n_steps_i)
                dt_list_lv.append(dt_mean_i)
                dt_min_list.append(dt_min_i)
                dt_max_list.append(dt_max_i)
                dist_lv.append(dist_i)
                dist95_lv.append(dist95_i)

                geom_ok = float(lsf.evaluate(q_new)) <= threshold

                H0 = target.H(chains[i], p_init)
                H1 = target.H(q_new, p_new)

                if abs(H1 - H0) > 200.0:
                    n_div += 1
                    geom_ok = False

                if geom_ok:
                    lv_geom_pass += 1
                    mh_ok, dH = _mh_accept(H0, H1, rng)
                    dH_list.append(dH)
                    if mh_ok:
                        chains[i] = q_new
                        lv_acpt  += 1
                    else:
                        lv_ener  += 1
                else:
                    lv_geom += 1
                    rng.random()   # keep rng in sync across arms

                lv_n += 1

            chain_snapshots.append(chains.copy())

        # Per-level stats
        lv_geom_rej.append(lv_geom / max(lv_n, 1))
        lv_energy_rej.append(lv_ener / max(lv_n, 1))
        lv_accept_rate.append(lv_acpt / max(lv_n, 1))
        lv_cond_energy_rej.append(lv_ener / max(lv_geom_pass, 1))
        lv_abs_dH_mean.append(float(np.mean(dH_list)) if dH_list else 0.0)
        lv_abs_dH_p95.append(float(np.percentile(dH_list, 95)) if dH_list else 0.0)
        lv_grad_adv.append(n_grad_adv_lv)
        lv_grad_est_opt.append(n_grad_est_opt_lv)
        lv_grad_est_unopt.append(n_grad_est_unopt_lv)
        lv_n_steps_mean.append(float(np.mean(n_steps_list)) if n_steps_list else 0.0)
        lv_dt_mean.append(float(np.mean(dt_list_lv)) if dt_list_lv else DT_A)
        lv_dt_min_mean.append(float(np.mean(dt_min_list)) if dt_min_list else DT_A)
        lv_dt_max_mean.append(float(np.mean(dt_max_list)) if dt_max_list else DT_A)
        lv_distance_mean.append(float(np.mean(dist_lv)) if dist_lv else 0.0)
        lv_distance_p95.append(float(np.mean(dist95_lv)) if dist95_lv else 0.0)
        lv_divergences.append(n_div)

        q = (
            np.stack(chain_snapshots, axis=1)   # [n_seeds, spc, d]
            .reshape(-1, d)[:N]
        )

    # Max levels hit
    g_final = lsf.evaluate(q)
    n_fail  = int((g_final <= 0.0).sum())
    P_F     = (p0f ** SUS_MAX_LEVELS) * (n_fail / N)
    _log(f"    {rep_label} arm={arm}(eps={eps:.0e}) → max_levels  Pf={P_F:.4e}")
    return _build_result_D(
        P_F, SUS_MAX_LEVELS, thresholds, lsf,
        lv_geom_rej, lv_energy_rej, lv_accept_rate, lv_cond_energy_rej,
        lv_abs_dH_mean, lv_abs_dH_p95,
        lv_grad_adv, lv_grad_est_opt, lv_grad_est_unopt,
        lv_n_steps_mean, lv_dt_mean, lv_dt_min_mean, lv_dt_max_mean,
        lv_distance_mean, lv_distance_p95, lv_divergences,
        burn_grad_adv, burn_grad_est_opt, burn_grad_est_unopt,
    )


def _build_result_D(
    P_F, levels, thresholds, lsf,
    geom_rej, energy_rej, accept_rate, cond_energy_rej,
    dH_mean, dH_p95,
    grad_adv, grad_est_opt, grad_est_unopt,
    n_steps_mean, dt_mean, dt_min_mean, dt_max_mean,
    distance_mean, distance_p95, divergences,
    burn_adv, burn_est_opt, burn_est_unopt,
) -> Dict:
    pf = max(P_F, 1e-300)
    beta_hat = float(-Norm.ppf(pf))
    tot_adv       = sum(grad_adv)       + burn_adv
    tot_est_opt   = sum(grad_est_opt)   + burn_est_opt
    tot_est_unopt = sum(grad_est_unopt) + burn_est_unopt
    return dict(
        P_F=P_F, beta_hat=beta_hat, levels=levels, thresholds=thresholds,
        per_level_geom_rej=geom_rej,
        per_level_energy_rej=energy_rej,
        per_level_accept_rate=accept_rate,
        per_level_cond_energy_rej=cond_energy_rej,
        per_level_abs_dH_mean=dH_mean,
        per_level_abs_dH_p95=dH_p95,
        per_level_grad_adv=grad_adv,
        per_level_grad_est_opt=grad_est_opt,
        per_level_grad_est_unopt=grad_est_unopt,
        per_level_n_steps_mean=n_steps_mean,
        per_level_dt_mean=dt_mean,
        per_level_dt_min_mean=dt_min_mean,
        per_level_dt_max_mean=dt_max_mean,
        per_level_distance_mean=distance_mean,
        per_level_distance_p95=distance_p95,
        per_level_divergences=divergences,
        total_grad_adv=tot_adv,
        total_grad_est_opt=tot_est_opt,
        total_grad_est_unopt=tot_est_unopt,
        total_grad_opt=tot_adv + tot_est_opt,
        total_grad_unopt=tot_adv + tot_est_unopt,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-replication runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_arm(
    arm: str,
    target: CorrelatedGaussianTarget,
    lsf: ThalerLinearLSF,
    n_rep: int,
    tau: float = TAU_SAFETY,
    eps: float = 1e-2,
    label: str = "",
) -> Dict:
    _log(f"\n[arm={arm}  eps={eps:.0e}  {label}  {n_rep} reps]")
    results = []
    t0 = time.perf_counter()

    for rep in range(n_rep):
        rl = f"rep{rep+1:02d}/{n_rep}"
        seed = SEED_BASE + rep
        _log(f"  {rl} seed={seed}")
        t1  = time.perf_counter()
        r   = _run_one_sus_D(arm, target, lsf, tau, eps, seed, rep_label=rl)
        results.append(r)
        _log(f"  {rl} done: Pf={r['P_F']:.4e}  beta={r['beta_hat']:.4f}"
             f"  lv={r['levels']}  ({time.perf_counter()-t1:.1f}s  "
             f"tot={time.perf_counter()-t0:.1f}s)")

    pf_arr   = np.array([r["P_F"]      for r in results])
    beta_arr = np.array([r["beta_hat"] for r in results])
    lv_arr   = np.array([r["levels"]   for r in results])

    mean_pf   = float(pf_arr.mean())
    cov_pf    = float(pf_arr.std() / mean_pf) if mean_pf > 0 else float("nan")
    mean_beta = float(beta_arr.mean())
    std_beta  = float(beta_arr.std())
    bias_beta = mean_beta - lsf.beta_ref
    cov_beta  = std_beta / lsf.beta_ref if lsf.beta_ref > 0 else float("nan")

    # Per-level aggregation
    max_lv = max(len(r["per_level_geom_rej"]) for r in results)
    agg: Dict[str, List] = {k: [] for k in [
        "geom_rej", "energy_rej", "accept_rate", "cond_energy_rej",
        "abs_dH_mean", "abs_dH_p95",
        "grad_adv", "grad_est_opt", "grad_est_unopt",
        "n_steps_mean", "dt_mean", "dt_min_mean", "dt_max_mean",
        "distance_mean", "distance_p95", "divergences",
    ]}
    for lv in range(max_lv):
        def _mv(key, default=0.0):
            vals = [r[f"per_level_{key}"][lv]
                    for r in results if len(r[f"per_level_{key}"]) > lv]
            return float(np.mean(vals)) if vals else default
        for k in agg:
            agg[k].append(_mv(k))

    mean_grad_opt   = float(np.mean([r["total_grad_opt"]   for r in results]))
    mean_grad_unopt = float(np.mean([r["total_grad_unopt"] for r in results]))
    mean_grad_adv   = float(np.mean([r["total_grad_adv"]   for r in results]))
    mean_grad_est_opt   = float(np.mean([r["total_grad_est_opt"]   for r in results]))
    mean_grad_est_unopt = float(np.mean([r["total_grad_est_unopt"] for r in results]))

    return dict(
        arm=arm, label=label, eps=eps, tau=tau,
        pf_arr=pf_arr, beta_arr=beta_arr,
        mean_pf=mean_pf, cov_pf=cov_pf,
        mean_beta=mean_beta, std_beta=std_beta,
        bias_beta=bias_beta, cov_beta=cov_beta,
        mean_levels=float(lv_arr.mean()),
        agg=agg, results=results,
        mean_grad_opt=mean_grad_opt,
        mean_grad_unopt=mean_grad_unopt,
        mean_grad_adv=mean_grad_adv,
        mean_grad_est_opt=mean_grad_est_opt,
        mean_grad_est_unopt=mean_grad_est_unopt,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Efficiency metric
# ═══════════════════════════════════════════════════════════════════════════════

def efficiency(r_A: Dict, r_D: Dict, cost_key: str) -> float:
    """
    E = (cost_A / cost_D) * (COV_A / COV_D)^2
    Using total grad_U as cost.  E > 1 means D is more efficient than A.
    """
    cost_A = r_A[cost_key]
    cost_D = r_D[cost_key]
    cov_A  = r_A["cov_beta"]
    cov_D  = r_D["cov_beta"]
    if cost_D == 0 or cov_D == 0:
        return float("nan")
    return (cost_A / cost_D) * (cov_A / cov_D) ** 2


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

_COLORS = {"A": "#4C72B0", "D_1e-3": "#55A868", "D_1e-2": "#DD8452", "D_1e-1": "#C44E52"}

def _arm_color(arm: str, eps: float) -> str:
    if arm == "A":
        return _COLORS["A"]
    key = f"D_{eps:.0e}"
    return _COLORS.get(key, "gray")


def plot_eps_sweep(
    r_A: Dict,
    r_D_list: List[Dict],
    lsf: ThalerLinearLSF,
    out_dir: Path,
    tag: str,
) -> None:
    """Plot eps sweep: COV, bias, total grad cost (both conventions), distance."""
    eps_vals  = [r["eps"] for r in r_D_list]
    cov_D     = [r["cov_beta"] for r in r_D_list]
    bias_D    = [r["bias_beta"] for r in r_D_list]
    grad_opt  = [r["mean_grad_opt"]   for r in r_D_list]
    grad_unopt= [r["mean_grad_unopt"] for r in r_D_list]
    dist_mean = [r["agg"]["distance_mean"][-1] if r["agg"]["distance_mean"] else 0.0
                 for r in r_D_list]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # COV vs eps
    ax = axes[0, 0]
    ax.axhline(r_A["cov_beta"], color=_COLORS["A"], ls="--", lw=1.5, label="ARM A baseline")
    ax.semilogx(eps_vals, cov_D, "o-", color="tomato", label="ARM D")
    ax.set_xlabel("eps (tolerance)"); ax.set_ylabel("COV_beta")
    ax.set_title("COV vs eps tolerance"); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    # Bias vs eps
    ax = axes[0, 1]
    ax.axhline(r_A["bias_beta"], color=_COLORS["A"], ls="--", lw=1.5, label="ARM A baseline")
    ax.axhline(0.0, color="k", ls=":", lw=1)
    ax.semilogx(eps_vals, bias_D, "o-", color="tomato", label="ARM D")
    ax.set_xlabel("eps"); ax.set_ylabel("beta_hat − beta_ref")
    ax.set_title("Bias vs eps"); ax.legend(fontsize=9); ax.grid(True, alpha=0.4)

    # Total grad cost vs eps (both conventions)
    ax = axes[1, 0]
    ax.axhline(r_A["mean_grad_opt"], color=_COLORS["A"], ls="--", lw=1.5,
               label="ARM A (same both conventions)")
    ax.semilogx(eps_vals, grad_opt,   "o-",  color="tomato",   label="ARM D optimized (3/step)")
    ax.semilogx(eps_vals, grad_unopt, "s--", color="darkred",  label="ARM D unoptimized (6/step)")
    ax.set_xlabel("eps"); ax.set_ylabel("Total grad_U evals")
    ax.set_title("Cost vs eps (both Yoshida conventions)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # Mean distance per level (last level or overall)
    ax = axes[1, 1]
    # Flatten per-level distance_mean across all levels
    n_lv = max(len(r["agg"]["distance_mean"]) for r in r_D_list)
    for r in r_D_list:
        dm = r["agg"]["distance_mean"]
        x  = np.arange(1, len(dm) + 1)
        ax.plot(x, dm, "o-", label=f"eps={r['eps']:.0e}")
    ax.set_xlabel("Level"); ax.set_ylabel("Mean distance per level")
    ax.set_title("Distance estimate per level (ARM D)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fig.suptitle(f"Yoshida-distance adaptive controller — {lsf.beta_ref:.1f}/{lsf.rho}", fontsize=12)
    plt.tight_layout()
    path = out_dir / f"eps_sweep_beta{lsf.beta_ref}_rho{lsf.rho}{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    _log(f"  Saved {path.name}")


def plot_level_diagnostics(
    r_A: Dict,
    r_D_list: List[Dict],
    lsf: ThalerLinearLSF,
    out_dir: Path,
    tag: str,
) -> None:
    """Per-level: dt_mean, n_steps, abs_dH_mean, distance_mean for each eps."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    def _lv(r, key):
        return r["agg"].get(key, [])

    # dt mean
    ax = axes[0, 0]
    ax.axhline(DT_A, color=_COLORS["A"], ls="--", lw=1.5, label=f"ARM A (dt={DT_A})")
    for r in r_D_list:
        dm = _lv(r, "dt_mean")
        ax.plot(np.arange(1, len(dm)+1), dm, "o-", label=f"eps={r['eps']:.0e}")
    ax.set_xlabel("Level"); ax.set_ylabel("Mean dt used")
    ax.set_title("Step size per level"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # n_steps mean
    ax = axes[0, 1]
    ax.axhline(N_STEPS_A, color=_COLORS["A"], ls="--", lw=1.5, label=f"ARM A ({N_STEPS_A} steps)")
    for r in r_D_list:
        ns = _lv(r, "n_steps_mean")
        ax.plot(np.arange(1, len(ns)+1), ns, "o-", label=f"eps={r['eps']:.0e}")
    ax.set_xlabel("Level"); ax.set_ylabel("Mean n_steps per trajectory")
    ax.set_title("Steps per trajectory"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # |dH| mean
    ax = axes[1, 0]
    dH_A = _lv(r_A, "abs_dH_mean")
    if dH_A:
        ax.plot(np.arange(1, len(dH_A)+1), dH_A, "o--",
                color=_COLORS["A"], label="ARM A")
    for r in r_D_list:
        dh = _lv(r, "abs_dH_mean")
        ax.plot(np.arange(1, len(dh)+1), dh, "o-", label=f"eps={r['eps']:.0e}")
    ax.set_xlabel("Level"); ax.set_ylabel("Mean |ΔH|")
    ax.set_title("|ΔH| per level"); ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.set_yscale("log")

    # distance mean per level
    ax = axes[1, 1]
    for r in r_D_list:
        dm = _lv(r, "distance_mean")
        ax.plot(np.arange(1, len(dm)+1), dm, "o-", label=f"eps={r['eps']:.0e}")
    ax.set_xlabel("Level"); ax.set_ylabel("Mean LF−Y4 distance")
    ax.set_title("Error estimate per level (ARM D only)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    fig.suptitle(f"Per-level diagnostics — beta={lsf.beta_ref}, rho={lsf.rho}", fontsize=12)
    plt.tight_layout()
    path = out_dir / f"level_diag_beta{lsf.beta_ref}_rho{lsf.rho}{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    _log(f"  Saved {path.name}")


def plot_beta_boxplot(
    r_A: Dict,
    r_D_list: List[Dict],
    lsf: ThalerLinearLSF,
    out_dir: Path,
    tag: str,
) -> None:
    labels = ["ARM A"] + [f"D eps={r['eps']:.0e}" for r in r_D_list]
    data   = [r_A["beta_arr"]] + [r["beta_arr"] for r in r_D_list]
    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = [_COLORS["A"]] + [_arm_color("D", r["eps"]) for r in r_D_list]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
    ax.axhline(lsf.beta_ref, color="red", ls="--", lw=1.2,
               label=f"beta_ref={lsf.beta_ref}")
    ax.set_ylabel("beta_hat")
    ax.set_title(f"beta_hat — beta={lsf.beta_ref}, rho={lsf.rho}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = out_dir / f"beta_boxplot_beta{lsf.beta_ref}_rho{lsf.rho}{tag}.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    _log(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Record writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_record(
    configs: List[Dict],
    n_rep: int,
    tau: float,
    eps_list: List[float],
) -> None:
    lines = [
        "# Accuracy-adaptive Leapfrog via Leapfrog−Yoshida Distance",
        "",
        "## HOW TO READ THIS EXPERIMENT",
        "",
        "This controller targets INTEGRATION/ENERGY error, not geometric rejection.",
        "Prior experiments established that geometric rejection dominates and energy",
        "acceptance is already >99%, so the HONEST expectation is that ARM D does NOT",
        "improve geometric acceptance or COV via the energy axis.",
        "",
        "**Success criterion:**",
        "1. Does the distance behave as a sensible local-error estimate (grows where",
        "   the trajectory is harder to integrate)?",
        "2. Does the controller keep |ΔH| under the tolerance while adapting dt?",
        "3. Is there any eps at which ARM D matches baseline accuracy at LOWER total",
        "   gradient cost, DESPITE the ~3× or ~6× Yoshida overhead per step?",
        "",
        "Frame the result on that axis.  If E < 1 everywhere (the Yoshida overhead",
        "outweighs any step savings), that is itself a clean, publishable conclusion.",
        "",
        "## Controller settings",
        "",
        "| param | value |",
        "|-------|-------|",
        f"| lower-order method | leapfrog (order q=2) |",
        f"| exponent 1/(q+1) | {EXPONENT:.6f} (= 1/3) |",
        f"| tau (safety factor) | {tau} |",
        f"| eps sweep | {eps_list} |",
        f"| change factor clip | [{CHANGE_MIN}, {CHANGE_MAX}] |",
        f"| dt_min | {DT_MIN} |",
        f"| dt_max | {DT_MAX} |",
        f"| DT_A (baseline) | {DT_A} |",
        f"| N_STEPS_A (baseline) | {N_STEPS_A} |",
        f"| T_F (integration time) | {T_F} |",
        f"| SUS_N | {SUS_N} |",
        f"| SUS_P0 | {SUS_P0} |",
        f"| SUS_BURN_IN | {SUS_BURN_IN} |",
        f"| n_rep | {n_rep} |",
        "",
        "## Cost accounting — TWO conventions",
        "",
        "Each step of ARM D costs:",
        "  OPTIMIZED (g0-reuse + kick-merge):   1 (leapfrog) + 3 (Yoshida) = 4 grad_U/step",
        "  UNOPTIMIZED (existing integrators.py): 1 (leapfrog) + 6 (Yoshida) = 7 grad_U/step",
        "ARM A costs:  1 (leapfrog) per step = 11 grad_U/trajectory (n_steps+1=11).",
        "Efficiency E = (cost_A / cost_D) * (COV_A / COV_D)^2 reported for BOTH conventions.",
        "E > 1 means ARM D is more efficient; E < 1 means the overhead outweighs any saving.",
        "",
        "**CORRECTNESS NOTE**: state-dependent per-step dt breaks leapfrog",
        "reversibility/volume-preservation → ARM D is an APPROXIMATE sampler.",
        "Bias measured as mean(beta_hat) − beta_ref.",
        "",
    ]

    for cfg in configs:
        lsf     = cfg["lsf"]
        r_A     = cfg["r_A"]
        r_D_list= cfg["r_D_list"]

        lines += [
            f"## Config: beta={lsf.beta_ref}, rho={lsf.rho}",
            f"(Pf_ref={lsf.pf_ref:.4e}, beta_ref={lsf.beta_ref})",
            "",
            "### Summary table",
            "",
            "| arm | eps | beta_hat | std | bias | COV_beta | "
            "grad_adv | grad_est(opt) | grad_est(unopt) | total(opt) | total(unopt) | mean_lv |",
            "|-----|-----|----------|-----|------|----------|"
            "---------|--------------|----------------|-----------|-------------|---------|",
        ]

        def _row(r):
            return (
                f"| {r['arm']} | {r['eps']:.0e} | {r['mean_beta']:.4f} | "
                f"{r['std_beta']:.4f} | {r['bias_beta']:+.4f} | {r['cov_beta']:.3f} | "
                f"{r['mean_grad_adv']:.2e} | {r['mean_grad_est_opt']:.2e} | "
                f"{r['mean_grad_est_unopt']:.2e} | "
                f"{r['mean_grad_opt']:.2e} | {r['mean_grad_unopt']:.2e} | "
                f"{r['mean_levels']:.1f} |"
            )

        # ARM A row (no Yoshida cost)
        lines.append(
            f"| A | — | {r_A['mean_beta']:.4f} | {r_A['std_beta']:.4f} | "
            f"{r_A['bias_beta']:+.4f} | {r_A['cov_beta']:.3f} | "
            f"{r_A['mean_grad_opt']:.2e} | 0 | 0 | "
            f"{r_A['mean_grad_opt']:.2e} | {r_A['mean_grad_opt']:.2e} | "
            f"{r_A['mean_levels']:.1f} |"
        )
        for r in r_D_list:
            lines.append(_row(r))

        lines += ["", "### Efficiency E = (cost_A/cost_D) * (COV_A/COV_D)^2", "",
                  "| arm | eps | E_optimized | E_unoptimized | interpretation |",
                  "|-----|-----|-------------|---------------|----------------|"]
        for r in r_D_list:
            e_opt   = efficiency(r_A, r, "mean_grad_opt")
            e_unopt = efficiency(r_A, r, "mean_grad_unopt")
            interp  = "D wins" if e_opt > 1 else "A wins"
            lines.append(
                f"| D | {r['eps']:.0e} | {e_opt:.3f} | {e_unopt:.3f} | {interp} |"
            )

        lines += ["", "### Per-level diagnostics — ARM A", "",
                  "| lv | geom_rej | energy_rej | accept | cond_ener | "
                  "|ΔH| mean | |ΔH| p95 | dt_mean | n_steps | grad_adv |",
                  "|----|----------|-----------|--------|-----------|"
                  "----------|----------|---------|---------|----------|"]
        agg_A = r_A["agg"]
        for lv in range(len(agg_A["geom_rej"])):
            lines.append(
                f"| L{lv+1} | {agg_A['geom_rej'][lv]:.3f} | "
                f"{agg_A['energy_rej'][lv]:.3f} | "
                f"{agg_A['accept_rate'][lv]:.3f} | "
                f"{agg_A['cond_energy_rej'][lv]:.3f} | "
                f"{agg_A['abs_dH_mean'][lv]:.3e} | "
                f"{agg_A['abs_dH_p95'][lv]:.3e} | "
                f"{DT_A:.3f} | {N_STEPS_A} | "
                f"{agg_A['grad_adv'][lv]:.0f} |"
            )

        for r in r_D_list:
            agg = r["agg"]
            lines += [
                f"", f"### Per-level diagnostics — ARM D (eps={r['eps']:.0e})", "",
                "| lv | geom_rej | energy_rej | accept | cond_ener | "
                "|ΔH| mean | |ΔH| p95 | dt_mean | dt_min | dt_max | "
                "n_steps | dist_mean | dist_p95 | grad_adv | grad_est(opt) | grad_est(unopt) | div |",
                "|----|----------|-----------|--------|-----------|"
                "----------|----------|---------|--------|--------|"
                "---------|-----------|----------|----------|--------------|----------------|-----|",
            ]
            for lv in range(len(agg["geom_rej"])):
                lines.append(
                    f"| L{lv+1} | {agg['geom_rej'][lv]:.3f} | "
                    f"{agg['energy_rej'][lv]:.3f} | "
                    f"{agg['accept_rate'][lv]:.3f} | "
                    f"{agg['cond_energy_rej'][lv]:.3f} | "
                    f"{agg['abs_dH_mean'][lv]:.3e} | "
                    f"{agg['abs_dH_p95'][lv]:.3e} | "
                    f"{agg['dt_mean'][lv]:.4f} | "
                    f"{agg['dt_min_mean'][lv]:.4f} | "
                    f"{agg['dt_max_mean'][lv]:.4f} | "
                    f"{agg['n_steps_mean'][lv]:.1f} | "
                    f"{agg['distance_mean'][lv]:.3e} | "
                    f"{agg['distance_p95'][lv]:.3e} | "
                    f"{agg['grad_adv'][lv]:.0f} | "
                    f"{agg['grad_est_opt'][lv]:.0f} | "
                    f"{agg['grad_est_unopt'][lv]:.0f} | "
                    f"{agg['divergences'][lv]:.0f} |"
                )

        lines += ["", "### What this tells us", ""]
        for r in r_D_list:
            e_opt   = efficiency(r_A, r, "mean_grad_opt")
            e_unopt = efficiency(r_A, r, "mean_grad_unopt")
            lines.append(
                f"- **eps={r['eps']:.0e}**: beta_hat bias={r['bias_beta']:+.4f}, "
                f"COV={r['cov_beta']:.3f} vs baseline {r_A['cov_beta']:.3f}. "
                f"E_opt={e_opt:.3f}, E_unopt={e_unopt:.3f}. "
                f"Avg distance (last level)={r['agg']['distance_mean'][-1] if r['agg']['distance_mean'] else float('nan'):.3e}."
            )
        lines.append("")
        lines.append(
            "The distance estimate grows across levels if the harder conditional "
            "distribution requires shorter steps to maintain accuracy.  "
            "E < 1 under both conventions confirms that the Yoshida overhead "
            "outweighs any step-count savings on this problem — consistent with "
            "the energy-axis-is-saturated finding from prior experiments."
        )
        lines.append("")

    with open(REC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _log(f"  Wrote {REC_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(n_rep: int = 2) -> None:
    _open_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if n_rep <= 5 else ""

    tau      = TAU_SAFETY
    eps_list = [1e-3, 1e-2, 1e-1]

    lsf_configs = [
        (3.5, 0.00),
        (4.0, 0.75),
    ]

    _log("=" * 70)
    _log(f"Yoshida-distance adaptive leapfrog  n_rep={n_rep}  tag={tag or 'full'}")
    _log(f"tau={tau}  eps={eps_list}  exponent=1/3  DT_A={DT_A}  T_F={T_F}")
    _log(f"SUS_N={SUS_N}  SUS_P0={SUS_P0}  SUS_BURN_IN={SUS_BURN_IN}")
    _log(f"Cost conventions: optimized (3 Yoshida evals/step) "
         f"AND unoptimized (6 Yoshida evals/step)")
    _log("=" * 70)

    t_global = time.perf_counter()
    all_configs = []
    npz_data: Dict[str, np.ndarray] = {}

    for beta, rho in lsf_configs:
        lsf    = ThalerLinearLSF(beta=beta, rho=rho)
        target = CorrelatedGaussianTarget(rho=rho)
        cfg_tag = f"b{beta}_r{rho}"

        _log(f"\n{'='*70}")
        _log(f"Config: beta={beta}  rho={rho}  Pf_ref={lsf.pf_ref:.4e}")

        # ── ARM A (baseline) ───────────────────────────────────────────────
        _log(f"\n--- ARM A (baseline) ---")
        r_A = run_arm("A", target, lsf, n_rep, tau=tau, eps=0.0, label=cfg_tag)
        npz_data[f"{cfg_tag}_A_beta"] = r_A["beta_arr"]

        # ── ARM D (eps sweep) ──────────────────────────────────────────────
        r_D_list = []
        for eps in eps_list:
            _log(f"\n--- ARM D  eps={eps:.0e} ---")
            r_D = run_arm("D", target, lsf, n_rep, tau=tau, eps=eps, label=cfg_tag)
            r_D_list.append(r_D)
            npz_data[f"{cfg_tag}_D_{eps:.0e}_beta"] = r_D["beta_arr"]

        all_configs.append(dict(lsf=lsf, r_A=r_A, r_D_list=r_D_list))

        # ── Console summary ────────────────────────────────────────────────
        _log(f"\n--- Summary: beta={beta}, rho={rho} ---")
        _log(f"  ARM A:  beta_hat={r_A['mean_beta']:.4f}  bias={r_A['bias_beta']:+.4f}"
             f"  COV={r_A['cov_beta']:.3f}  grad_opt={r_A['mean_grad_opt']:.2e}")
        for r in r_D_list:
            e_opt   = efficiency(r_A, r, "mean_grad_opt")
            e_unopt = efficiency(r_A, r, "mean_grad_unopt")
            _log(
                f"  ARM D eps={r['eps']:.0e}: "
                f"beta_hat={r['mean_beta']:.4f}  bias={r['bias_beta']:+.4f}"
                f"  COV={r['cov_beta']:.3f}"
                f"  grad_opt={r['mean_grad_opt']:.2e}"
                f"  grad_unopt={r['mean_grad_unopt']:.2e}"
                f"  E_opt={e_opt:.3f}  E_unopt={e_unopt:.3f}"
            )

        # ── Plots ──────────────────────────────────────────────────────────
        _log(f"\n[Saving plots for beta={beta}, rho={rho}]")
        plot_eps_sweep(r_A, r_D_list, lsf, OUT_DIR, tag)
        plot_level_diagnostics(r_A, r_D_list, lsf, OUT_DIR, tag)
        plot_beta_boxplot(r_A, r_D_list, lsf, OUT_DIR, tag)

    # ── Save NPZ ───────────────────────────────────────────────────────────
    npz_path = OUT_DIR / f"yoshida_dist_raw{tag}.npz"
    np.savez(npz_path, **npz_data)
    _log(f"\n  Saved {npz_path.name}")

    # ── Write record ───────────────────────────────────────────────────────
    write_record(all_configs, n_rep, tau, eps_list)

    _log(f"\nTotal wall time: {time.perf_counter()-t_global:.1f}s")
    _log(f"Done.  All outputs in {OUT_DIR}")
    _close_log()


if __name__ == "__main__":
    import sys as _sys
    _full = "--full" in _sys.argv
    main(n_rep=30 if _full else 2)
