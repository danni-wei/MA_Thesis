"""
experiment_geometry_bounce.py
─────────────────────────────────────────────────────────────────────────────
3-arm HMC comparison in Subset Simulation:

  ARM A  — fixed leapfrog + end-reject  (baseline)
  ARM B  — geometry-aware adaptive step + end-reject  (tests H_B: refine alone)
  ARM C  — geometry-aware adaptive step + barrier bounce  (tests H_C: bouncing)
  ARM C' — FIXED step + barrier bounce (ablation, flag bounce_fixed_step=True)

True/analytic gradients only — HNN path not touched.
All arms use the same total integration time t_f per proposal.

CORRECTNESS NOTE
  A state-dependent per-step dt (arms B, C) breaks leapfrog map reversibility /
  volume-preservation → approximate samplers.  The dt_geom regularizer
  dt_geom = c_cfl*(m/gdot + dt_seed) ensures dt_geom >= c_cfl*dt_seed even when
  m→0, so the trajectory never stalls.  This non-zero floor increases bias vs the
  limiting "pure CFL" (dt_seed=0) and is why C' (fixed+bounce) is the reference.
  Bias measured via beta_hat vs analytic beta_ref (Exp 1).

Run from project root:
    python -u _run_v3_geometry_bounce.py               # smoke (n_rep=2)
    python -u _run_v3_geometry_bounce.py --full        # full  (n_rep=30)
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
from scipy.stats import norm as Norm, ks_2samp, truncnorm

# ── Output paths ───────────────────────────────────────────────────────────────
OUT_DIR  = Path("results/geometry_bounce")
REC_PATH = OUT_DIR / "record_geometry_bounce.md"
LOG_PATH = OUT_DIR / "geometry_bounce_run.log"

# ── Experiment hyperparameters ─────────────────────────────────────────────────
DT_A       = 0.1    # ARM A (baseline) fixed step size
N_STEPS_A  = 10     # ARM A fixed number of leapfrog steps
T_F        = DT_A * N_STEPS_A   # 1.0 — shared integration time for all arms

SUS_N          = 1000
SUS_P0         = 0.1
SUS_BURN_IN    = 200
SUS_MAX_LEVELS = 8
SEED_BASE      = 0

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
# Limit-state functions (with analytic grad_G)
# ═══════════════════════════════════════════════════════════════════════════════

class ThalerLinearLSF:
    """
    Thaler 2024 §5.1 linear LSF (2D).

        g(x) = beta * sqrt(sigma_max * d) - sum_i x_i

    Failure domain: {g <= 0}.  SuS threshold F_k = {g <= b_k}.
    Analytic reliability index = beta exactly; Pf = Phi(-beta).

    sigma_max = max eigenvalue of the 2×2 correlation matrix [[1,rho],[rho,1]]
               = 1 + |rho|.

    grad_G(x) = -ones_d  (constant — zero extra evals).

    Precision note: m(q) = b_k - g(q) = b_k - (offset - sum(q)) is a single
    float64 subtraction.  Near b_k the relative error is ~eps_mach/|m|; with
    float64 (numpy default) this is ~1e-16 and is negligible.
    """
    def __init__(self, beta: float, rho: float = 0.0, d: int = 2):
        self.beta = float(beta)
        self.rho  = float(rho)
        self.d    = d
        self.sigma_max = 1.0 + abs(rho)       # d=2 uniform correlation
        self.G_offset  = float(beta * math.sqrt(self.sigma_max * d))
        self.pf_ref    = float(Norm.cdf(-beta))
        self.beta_ref  = float(beta)

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        return self.G_offset - q.sum(axis=-1)

    def grad_G(self, q: np.ndarray) -> np.ndarray:
        return -np.ones_like(q)

    def is_failure(self, q: np.ndarray) -> np.ndarray:
        return self.evaluate(q) <= 0.0


class HalfSpaceLSF:
    """
    Exp 0 constraint: {x[0] <= b}.

    SuS convention (failure = target): G(x) = x[0] - b, failure = {G<=0}.
    b_k = 0 throughout (no threshold adaptation — single-level constrained HMC).
    grad_G = [1, 0, ..., 0]  (constant).
    """
    def __init__(self, b: float, d: int = 2):
        self.b = float(b)
        self.d = d

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        if q.ndim == 1:
            return np.array(q[0] - self.b)
        return q[:, 0] - self.b

    def grad_G(self, q: np.ndarray) -> np.ndarray:
        g = np.zeros_like(q)
        if g.ndim == 1:
            g[0] = 1.0
        else:
            g[:, 0] = 1.0
        return g


# ═══════════════════════════════════════════════════════════════════════════════
# Target distributions (HMC prior)
# ═══════════════════════════════════════════════════════════════════════════════

class CorrelatedGaussianTarget:
    """
    2D possibly-correlated Gaussian.  U(q) = 0.5 q^T Σ^{-1} q,  K(p)=0.5|p|².
    grad_U is analytic (exact, no autograd).
    """
    def __init__(self, rho: float = 0.0, d: int = 2):
        self.rho = float(rho)
        self.d   = d
        cov = np.eye(d, dtype=np.float64)
        if d == 2:
            cov[0, 1] = cov[1, 0] = rho
        self._inv_cov = np.linalg.inv(cov)
        self._chol    = np.linalg.cholesky(cov)   # for prior sampling

    def U(self, q: np.ndarray) -> float:
        return 0.5 * float(q @ self._inv_cov @ q)

    def grad_U(self, q: np.ndarray) -> np.ndarray:
        return (self._inv_cov @ q).copy()

    def K(self, p: np.ndarray) -> float:
        return 0.5 * float(p @ p)

    def H(self, q: np.ndarray, p: np.ndarray) -> float:
        return self.U(q) + self.K(p)

    def sample_momentum(self, rng: np.random.Generator) -> np.ndarray:
        return rng.standard_normal(self.d)

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        z = rng.standard_normal((n, self.d))
        return z @ self._chol.T


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry-aware controller
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ControllerConfig:
    c_cfl: float       = 0.5    # fraction of regularised time-to-boundary
    eps_E: float       = 0.05   # energy-error tolerance (phase-space units)
    tau_safety: float  = 0.85   # safety factor for energy cap
    dt_min: float      = 1e-3   # absolute minimum step size
    dt_max: float      = 0.5    # absolute maximum step size
    dt_max_change: float = 5.0  # max growth/shrink factor per step
    # bounce-specific
    n_bounce_max: int  = 5      # max reflections per trajectory (arm C)
    tol_hit: float     = 1e-6   # G-distance tolerance for bisection
    bisect_max_iter: int = 30   # bisection iteration cap
    # ablation flag
    bounce_fixed_step: bool = False  # if True → arm C' (fixed dt_seed + bounce)


def _compute_next_dt(
    q_new: np.ndarray,
    p_new: np.ndarray,
    q_eul: np.ndarray,   # SE companion position (free, shares g0)
    p_eul: np.ndarray,   # SE companion momentum
    dt_cur: float,
    G_new: float,
    grad_G_new: np.ndarray,
    b_k: float,
    dt_seed: float,      # baseline fixed dt (regularisation floor)
    cfg: ControllerConfig,
) -> float:
    """
    Geometry-primary + energy-safety combined step controller.

    Geometry (PRIMARY)
    ------------------
    When gdot = grad_G·p > 0 (heading toward b_k boundary):
        dt_geom = c_cfl * (m/max(gdot, tiny) + dt_seed)
    The +dt_seed regularises the singularity at m→0: even at the boundary
    dt_geom >= c_cfl*dt_seed (never stalls).  Hitting precision comes from
    the bisection, not from dt→0.
    When gdot <= 0 (moving away from boundary): dt_geom = dt_max.

    Energy (SAFETY CAP only — never primary)
    -----------------------------------------
    Uses the SE-Euler companion (free, no extra grad evals) to estimate
    the per-step energy error and cap dt.

    Final dt = clip(min(dt_geom, dt_energy), change_factor, [dt_min, dt_max]).
    """
    # ── Geometry ──────────────────────────────────────────────────────────────
    margin = b_k - G_new                            # >0 inside F_k
    gdot   = float(np.dot(grad_G_new, p_new))       # dG/dt = grad_G · v

    tiny = 1e-15
    if gdot > tiny:
        tau_b   = margin / gdot                     # may be negative if outside
        dt_geom = cfg.c_cfl * (max(tau_b, 0.0) + dt_seed)
    else:
        dt_geom = cfg.dt_max

    # ── Energy safety ─────────────────────────────────────────────────────────
    delta_E  = math.sqrt(
        float(np.sum((q_new - q_eul) ** 2) + np.sum((p_new - p_eul) ** 2))
    )
    dt_energy = cfg.tau_safety * math.sqrt(cfg.eps_E / max(delta_E, 1e-12)) * dt_cur
    dt_energy = min(dt_energy, cfg.dt_max)

    # ── Combine ───────────────────────────────────────────────────────────────
    dt_next = min(dt_geom, dt_energy)

    # clip growth factor
    dt_next = max(dt_cur / cfg.dt_max_change,
                  min(dt_cur * cfg.dt_max_change, dt_next))
    # absolute clip
    dt_next = max(cfg.dt_min, min(cfg.dt_max, dt_next))
    return dt_next


# ═══════════════════════════════════════════════════════════════════════════════
# Single Verlet step
# ═══════════════════════════════════════════════════════════════════════════════

def _verlet_step(
    q: np.ndarray,
    p: np.ndarray,
    g0: np.ndarray,
    grad_U_fn: Callable,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One velocity-Verlet step reusing g0.  Costs exactly 1 grad_U eval."""
    p_half = p - 0.5 * dt * g0
    q_new  = q + dt * p_half
    g1     = grad_U_fn(q_new)           # 1 grad_U eval
    p_new  = p_half - 0.5 * dt * g1
    return q_new, p_new, g1


# ═══════════════════════════════════════════════════════════════════════════════
# Barrier-bounce helpers (ARM C / C')
# ═══════════════════════════════════════════════════════════════════════════════

def _bisect_crossing(
    q_n: np.ndarray,
    p_n: np.ndarray,
    g0_n: np.ndarray,
    grad_U_fn: Callable,
    G_fn: Callable,
    b_k: float,
    dt_full: float,
    cfg: ControllerConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int, bool]:
    """
    Bisect on dt' ∈ (0, dt_full) to find G(q(dt')) = b_k.

    Uses velocity-Verlet from (q_n, p_n, g0_n) — g0 cached, no extra grad_U
    until convergence.

    Returns
    -------
    q_hit, p_hit, g1_hit, dt_hit, G_hit, n_G_evals, converged
    """
    dt_lo, dt_hi = 0.0, dt_full
    dt_mid = 0.5 * (dt_lo + dt_hi)
    G_mid  = b_k        # placeholder
    n_G    = 0
    converged = False

    for _ in range(cfg.bisect_max_iter):
        dt_mid   = 0.5 * (dt_lo + dt_hi)
        p_half   = p_n - 0.5 * dt_mid * g0_n
        q_mid    = q_n + dt_mid * p_half
        G_mid    = float(G_fn(q_mid))
        n_G     += 1
        m_mid    = b_k - G_mid
        if abs(m_mid) < cfg.tol_hit:
            converged = True
            break
        if m_mid > 0.0:
            dt_lo = dt_mid
        else:
            dt_hi = dt_mid

    # Final: full Verlet to get consistent p_hit (1 grad_U eval)
    dt_hit  = dt_mid
    p_half  = p_n - 0.5 * dt_hit * g0_n
    q_hit   = q_n + dt_hit * p_half
    g1_hit  = grad_U_fn(q_hit)          # 1 grad_U eval
    p_hit   = p_half - 0.5 * dt_hit * g1_hit
    return q_hit, p_hit, g1_hit, dt_hit, G_mid, n_G, converged


def _reflect_momentum(p_hit: np.ndarray, grad_G_hit: np.ndarray) -> np.ndarray:
    """
    Specular reflection p_ref = p - 2(p·n̂)n̂.
    Volume-, energy-preserving and involutive (reversible).
    grad_G is analytic (free for all LSFs considered).
    """
    norm_g = np.linalg.norm(grad_G_hit)
    if norm_g < 1e-30:
        return p_hit.copy()
    n_hat = grad_G_hit / norm_g
    return p_hit - 2.0 * float(np.dot(p_hit, n_hat)) * n_hat


# ═══════════════════════════════════════════════════════════════════════════════
# Per-proposal stats container
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PropStats:
    n_grad_U: int   = 0
    n_G:      int   = 0
    n_grad_G: int   = 0          # 0 for all LSFs here (grad_G is free/constant)
    n_steps:  int   = 0
    n_bounces: int  = 0
    bounce_failure: bool = False
    dt_min_used: float = math.inf
    dt_max_used: float = 0.0
    dt_mean:    float  = 0.0
    abs_dH:     float  = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ARM A — fixed leapfrog (baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_A(
    q0: np.ndarray,
    p0: np.ndarray,
    target: CorrelatedGaussianTarget,
    dt: float,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, PropStats]:
    """Fixed-step velocity-Verlet. Cost: n_steps+1 grad_U, 0 G (checked externally)."""
    q, p = q0.copy(), p0.copy()
    g = target.grad_U(q)
    n_grad = 1

    p = p - 0.5 * dt * g
    for _ in range(n_steps - 1):
        q = q + dt * p
        g = target.grad_U(q);  n_grad += 1
        p = p - dt * g
    q = q + dt * p
    g = target.grad_U(q);  n_grad += 1
    p = p - 0.5 * dt * g

    s = PropStats(n_grad_U=n_grad, n_steps=n_steps,
                  dt_min_used=dt, dt_max_used=dt, dt_mean=dt)
    return q, p, s


# ═══════════════════════════════════════════════════════════════════════════════
# ARM B — geometry-aware adaptive step, no bounce
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_B(
    q0: np.ndarray,
    p0: np.ndarray,
    target: CorrelatedGaussianTarget,
    G_fn: Callable,
    grad_G_fn: Callable,
    b_k: float,
    t_f: float,
    dt_seed: float,
    cfg: ControllerConfig,
) -> Tuple[np.ndarray, np.ndarray, PropStats]:
    """
    Adaptive velocity-Verlet without bounce (geometry + energy controller).

    NOTE: state-dependent dt breaks leapfrog reversibility → approximate
    sampler.  Bias is measured via beta_hat vs analytic beta_ref.
    grad_G is analytic (constant -ones for linear LSF) → n_grad_G = 0.
    """
    q, p = q0.copy(), p0.copy()
    g0     = target.grad_U(q)       # 1 grad_U
    G_val  = float(G_fn(q))         # 1 G
    n_grad = 1;  n_G = 1

    dt = dt_seed
    t  = 0.0;  n_steps = 0
    dt_list: List[float] = []

    while t < t_f - 1e-12:
        dt = min(dt, t_f - t)

        # SE companion (free — shares g0)
        p_eul = p - dt * g0
        q_eul = q + dt * p_eul

        q_new, p_new, g1 = _verlet_step(q, p, g0, target.grad_U, dt)
        n_grad += 1

        G_new       = float(G_fn(q_new));  n_G += 1
        grad_G_new  = grad_G_fn(q_new)    # free (constant)

        dt_next = _compute_next_dt(
            q_new, p_new, q_eul, p_eul,
            dt, G_new, grad_G_new, b_k, dt_seed, cfg,
        )

        t += dt;  dt_list.append(dt)
        q, p, g0, G_val = q_new, p_new, g1, G_new
        dt = dt_next;  n_steps += 1

    dt_arr = np.asarray(dt_list)
    s = PropStats(
        n_grad_U=n_grad, n_G=n_G, n_steps=n_steps,
        dt_min_used=float(dt_arr.min()) if n_steps else dt_seed,
        dt_max_used=float(dt_arr.max()) if n_steps else dt_seed,
        dt_mean    =float(dt_arr.mean()) if n_steps else dt_seed,
    )
    return q, p, s


# ═══════════════════════════════════════════════════════════════════════════════
# ARM C — geometry-aware adaptive step + barrier bounce
# (ARM C' = fixed step + bounce when cfg.bounce_fixed_step=True)
# ═══════════════════════════════════════════════════════════════════════════════

def integrate_C(
    q0: np.ndarray,
    p0: np.ndarray,
    target: CorrelatedGaussianTarget,
    G_fn: Callable,
    grad_G_fn: Callable,
    b_k: float,
    t_f: float,
    dt_seed: float,
    cfg: ControllerConfig,
) -> Tuple[np.ndarray, np.ndarray, PropStats]:
    """
    Adaptive velocity-Verlet + specular barrier bounce at G(q) = b_k.

    If cfg.bounce_fixed_step=True: fixed step dt_seed + bounce (ARM C'),
    the exactly-reversible reference.

    Bounce algorithm per step:
      1. Detect crossing: m_n>0 and m_{n+1}<0.
      2. Bisect to locate (q_hit, p_hit) s.t. |G(q_hit)-b_k| < tol_hit.
      3. Specular reflection: p_ref = p_hit - 2(p_hit·n̂)n̂.
      4. Continue from (q_hit, p_ref) with remaining time.
      5. If n_bounce_max exceeded: fall back to geometric rejection (bounce_failure).

    Eval cost per bounce: bisect_max_iter G evals + 1 grad_U.
    grad_G is analytic/constant → 0 extra evals.
    """
    q, p = q0.copy(), p0.copy()
    g0    = target.grad_U(q)        # 1 grad_U
    G_val = float(G_fn(q))          # 1 G
    n_grad = 1;  n_G = 1;  extra_G = 0

    dt = dt_seed
    t  = 0.0;  n_steps = 0;  n_bounces = 0
    bounce_failure = False
    dt_list: List[float] = []

    while t < t_f - 1e-12:
        dt = min(dt, t_f - t)

        # SE companion (free)
        p_eul = p - dt * g0
        q_eul = q + dt * p_eul

        q_new, p_new, g1 = _verlet_step(q, p, g0, target.grad_U, dt)
        n_grad += 1

        G_new  = float(G_fn(q_new));  n_G += 1
        m_n    = b_k - G_val
        m_n1   = b_k - G_new

        # ── Crossing detection ────────────────────────────────────────────
        if m_n > 0.0 and m_n1 < 0.0:
            if n_bounces >= cfg.n_bounce_max:
                bounce_failure = True
                break

            (q_hit, p_hit, g1_hit,
             dt_hit, G_hit, n_bis, conv) = _bisect_crossing(
                q, p, g0, target.grad_U, G_fn, b_k, dt, cfg,
            )
            n_grad  += 1             # 1 grad_U inside _bisect_crossing
            extra_G += n_bis         # bisection G evals

            grad_G_hit = grad_G_fn(q_hit)   # free (constant)
            p_ref      = _reflect_momentum(p_hit, grad_G_hit)

            t += dt_hit
            dt_list.append(dt_hit)
            n_steps += 1;  n_bounces += 1

            # Reset at boundary, heading inward after reflection
            q, p, g0 = q_hit, p_ref, g1_hit
            G_val = b_k     # at boundary by construction (within tol_hit)

            # dt after bounce: reset to dt_seed (controller re-adapts next iter)
            dt = min(dt_seed, t_f - t) if t < t_f else 0.0
            continue

        # ── Normal step ───────────────────────────────────────────────────
        grad_G_new = grad_G_fn(q_new)   # free

        if cfg.bounce_fixed_step:
            dt_next = dt_seed
        else:
            dt_next = _compute_next_dt(
                q_new, p_new, q_eul, p_eul,
                dt, G_new, grad_G_new, b_k, dt_seed, cfg,
            )

        t += dt;  dt_list.append(dt)
        q, p, g0, G_val = q_new, p_new, g1, G_new
        dt = dt_next;  n_steps += 1

    dt_arr = np.asarray(dt_list)
    s = PropStats(
        n_grad_U=n_grad, n_G=n_G + extra_G, n_steps=n_steps,
        n_bounces=n_bounces, bounce_failure=bounce_failure,
        dt_min_used=float(dt_arr.min()) if n_steps else dt_seed,
        dt_max_used=float(dt_arr.max()) if n_steps else dt_seed,
        dt_mean    =float(dt_arr.mean()) if n_steps else dt_seed,
    )
    return q, p, s


# ═══════════════════════════════════════════════════════════════════════════════
# MH acceptance (shared by all arms)
# ═══════════════════════════════════════════════════════════════════════════════

def _mh_accept(
    H0: float,
    H1: float,
    rng: np.random.Generator,
) -> Tuple[bool, float]:
    """
    Metropolis-Hastings step.  Returns (accepted, abs_dH).
    Clamps log_alpha to avoid overflow.
    """
    dH = H1 - H0
    log_alpha = -dH
    log_alpha = min(log_alpha, 80.0)
    accepted  = rng.random() < math.exp(log_alpha)
    return accepted, abs(dH)


# ═══════════════════════════════════════════════════════════════════════════════
# Single SuS run for one arm
# ═══════════════════════════════════════════════════════════════════════════════

def _run_one_sus(
    arm: str,
    target: CorrelatedGaussianTarget,
    lsf: ThalerLinearLSF,
    ctrl_cfg: ControllerConfig,
    seed: int,
    rep_label: str = "",
) -> Dict:
    """
    One complete SuS run for the given arm ("A", "B", "C", or "C'").
    Tracks per-level: geom_reject, energy_reject, accept_rate,
    cond_energy_reject, abs_dH stats, bounce stats, eval counts.
    All fractions use unconditional denominators (all proposals).
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
    lv_n_bounces_mean:      List[float] = []
    lv_bounce_traj_frac:    List[float] = []
    lv_bounce_failure_frac: List[float] = []
    lv_grad_U:              List[int]   = []
    lv_G:                   List[int]   = []
    lv_n_steps_mean:        List[float] = []
    lv_dt_min_mean:         List[float] = []
    lv_dt_max_mean:         List[float] = []
    lv_divergences:         List[int]   = []

    # ── Burn-in (unconstrained HMC, no geometric check) ───────────────────
    q = np.zeros((N, d), dtype=np.float64)
    burn_grad_U = 0;  burn_G = 0

    for _ in range(SUS_BURN_IN):
        for i in range(N):
            p_init = target.sample_momentum(rng)
            if arm == "A":
                q_new, p_new, st = integrate_A(q[i], p_init, target, DT_A, N_STEPS_A)
            elif arm == "B":
                q_new, p_new, st = integrate_B(
                    q[i], p_init, target,
                    lsf.evaluate, lsf.grad_G,
                    b_k=1e9,           # no constraint during burn-in
                    t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg,
                )
            else:  # C or C'
                q_new, p_new, st = integrate_C(
                    q[i], p_init, target,
                    lsf.evaluate, lsf.grad_G,
                    b_k=1e9, t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg,
                )
            burn_grad_U += st.n_grad_U;  burn_G += st.n_G
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
        _log(f"    {rep_label} arm={arm} L{level+1}: threshold={threshold:.4f}")

        if threshold <= 0.0:
            n_fail = int((g_vals <= 0.0).sum())
            P_F    = (p0f ** level) * (n_fail / N)
            _log(f"    {rep_label} arm={arm} → done L{level+1}  Pf={P_F:.4e}")
            return _build_sus_result(
                P_F, level + 1, thresholds,
                lv_geom_rej, lv_energy_rej, lv_accept_rate, lv_cond_energy_rej,
                lv_abs_dH_mean, lv_abs_dH_p95,
                lv_n_bounces_mean, lv_bounce_traj_frac, lv_bounce_failure_frac,
                lv_grad_U, lv_G, lv_n_steps_mean,
                lv_dt_min_mean, lv_dt_max_mean, lv_divergences,
                burn_grad_U, burn_G,
            )

        # Seeds: n_seeds samples with smallest (closest-to-failure) g
        seed_idx = np.argsort(g_vals)[:n_seeds]
        chains   = q[seed_idx].copy()
        chain_snapshots = [chains.copy()]

        lv_n = lv_geom = lv_ener = lv_acpt = lv_geom_pass = 0
        dH_list: List[float] = []
        bounce_counts: List[int] = []
        bounce_fail_n = 0
        n_steps_list: List[int] = []
        dt_min_list: List[float] = []
        dt_max_list: List[float] = []
        n_grad_U_lv = 0;  n_G_lv = 0
        n_div = 0

        for _ in range(spc - 1):
            for i in range(n_seeds):
                p_init = target.sample_momentum(rng)
                if arm == "A":
                    q_new, p_new, st = integrate_A(
                        chains[i], p_init, target, DT_A, N_STEPS_A)
                elif arm == "B":
                    q_new, p_new, st = integrate_B(
                        chains[i], p_init, target,
                        lsf.evaluate, lsf.grad_G,
                        b_k=threshold, t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg,
                    )
                else:  # C or C'
                    q_new, p_new, st = integrate_C(
                        chains[i], p_init, target,
                        lsf.evaluate, lsf.grad_G,
                        b_k=threshold, t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg,
                    )

                n_grad_U_lv += st.n_grad_U
                n_G_lv      += st.n_G
                n_steps_list.append(st.n_steps)
                dt_min_list.append(st.dt_min_used)
                dt_max_list.append(st.dt_max_used)
                bounce_counts.append(st.n_bounces)
                if st.bounce_failure:
                    bounce_fail_n += 1

                # Geometric check
                if st.bounce_failure:
                    geom_ok = False
                else:
                    geom_ok = float(lsf.evaluate(q_new)) <= threshold

                H0 = target.H(chains[i], p_init)
                H1 = target.H(q_new, p_new)

                # Divergence guard (|ΔH| > 200 indicates numerical blow-up)
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
                    # still draw u to keep rng in sync across arms
                    rng.random()

                lv_n += 1

            chain_snapshots.append(chains.copy())

        # Per-level aggregated stats
        lv_geom_rej.append(lv_geom / max(lv_n, 1))
        lv_energy_rej.append(lv_ener / max(lv_n, 1))
        lv_accept_rate.append(lv_acpt / max(lv_n, 1))
        lv_cond_energy_rej.append(
            lv_ener / max(lv_geom_pass, 1)
        )
        lv_abs_dH_mean.append(float(np.mean(dH_list)) if dH_list else 0.0)
        lv_abs_dH_p95.append(float(np.percentile(dH_list, 95)) if dH_list else 0.0)
        lv_n_bounces_mean.append(float(np.mean(bounce_counts)) if bounce_counts else 0.0)
        lv_bounce_traj_frac.append(
            float(np.mean([b > 0 for b in bounce_counts])) if bounce_counts else 0.0
        )
        lv_bounce_failure_frac.append(bounce_fail_n / max(lv_n, 1))
        lv_grad_U.append(n_grad_U_lv)
        lv_G.append(n_G_lv)
        lv_n_steps_mean.append(float(np.mean(n_steps_list)) if n_steps_list else 0.0)
        lv_dt_min_mean.append(float(np.mean(dt_min_list)) if dt_min_list else DT_A)
        lv_dt_max_mean.append(float(np.mean(dt_max_list)) if dt_max_list else DT_A)
        lv_divergences.append(n_div)

        # Rebuild q from chain snapshots
        q = (
            np.stack(chain_snapshots, axis=1)   # [n_seeds, spc, d]
            .reshape(-1, d)[:N]
        )

    # Max levels hit
    g_final = lsf.evaluate(q)
    n_fail  = int((g_final <= 0.0).sum())
    P_F     = (p0f ** SUS_MAX_LEVELS) * (n_fail / N)
    _log(f"    {rep_label} arm={arm} → max_levels  Pf={P_F:.4e}")
    return _build_sus_result(
        P_F, SUS_MAX_LEVELS, thresholds,
        lv_geom_rej, lv_energy_rej, lv_accept_rate, lv_cond_energy_rej,
        lv_abs_dH_mean, lv_abs_dH_p95,
        lv_n_bounces_mean, lv_bounce_traj_frac, lv_bounce_failure_frac,
        lv_grad_U, lv_G, lv_n_steps_mean,
        lv_dt_min_mean, lv_dt_max_mean, lv_divergences,
        burn_grad_U, burn_G,
    )


def _build_sus_result(P_F, levels, thresholds,
                      geom_rej, energy_rej, accept_rate, cond_energy_rej,
                      dH_mean, dH_p95,
                      bounces_mean, bounce_traj, bounce_fail,
                      grad_U_lv, G_lv, n_steps_mean,
                      dt_min_mean, dt_max_mean, divergences,
                      burn_grad_U, burn_G) -> Dict:
    pf = max(P_F, 1e-300)
    beta_hat = float(-Norm.ppf(pf))
    return dict(
        P_F=P_F, beta_hat=beta_hat,
        levels=levels, thresholds=thresholds,
        per_level_geom_rej=geom_rej,
        per_level_energy_rej=energy_rej,
        per_level_accept_rate=accept_rate,
        per_level_cond_energy_rej=cond_energy_rej,
        per_level_abs_dH_mean=dH_mean,
        per_level_abs_dH_p95=dH_p95,
        per_level_n_bounces_mean=bounces_mean,
        per_level_bounce_traj_frac=bounce_traj,
        per_level_bounce_fail_frac=bounce_fail,
        per_level_grad_U=grad_U_lv,
        per_level_G=G_lv,
        per_level_n_steps_mean=n_steps_mean,
        per_level_dt_min_mean=dt_min_mean,
        per_level_dt_max_mean=dt_max_mean,
        per_level_divergences=divergences,
        total_grad_U=sum(grad_U_lv) + burn_grad_U,
        total_G=sum(G_lv) + burn_G,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-replication runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_replications(
    arm: str,
    target: CorrelatedGaussianTarget,
    lsf: ThalerLinearLSF,
    ctrl_cfg: ControllerConfig,
    n_rep: int,
    label: str = "",
) -> Dict:
    _log(f"\n[arm={arm}  {label}  {n_rep} reps]")
    results = []
    t0 = time.perf_counter()

    for rep in range(n_rep):
        rl = f"rep{rep+1:02d}/{n_rep}"
        _log(f"  {rl} seed={SEED_BASE+rep}")
        t1 = time.perf_counter()
        r  = _run_one_sus(arm, target, lsf, ctrl_cfg,
                          seed=SEED_BASE + rep, rep_label=rl)
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

    # Per-level aggregation (across reps)
    max_lv = max(len(r["per_level_geom_rej"]) for r in results)
    agg: Dict[str, List] = {k: [] for k in [
        "geom_rej", "energy_rej", "accept_rate", "cond_energy_rej",
        "abs_dH_mean", "abs_dH_p95",
        "n_bounces_mean", "bounce_traj_frac", "bounce_fail_frac",
        "grad_U", "G", "n_steps_mean",
    ]}
    for lv in range(max_lv):
        def _mvals(key, default=0.0):
            vals = [r[f"per_level_{key}"][lv]
                    for r in results if len(r[f"per_level_{key}"]) > lv]
            return float(np.mean(vals)) if vals else default
        agg["geom_rej"].append(_mvals("geom_rej"))
        agg["energy_rej"].append(_mvals("energy_rej"))
        agg["accept_rate"].append(_mvals("accept_rate"))
        agg["cond_energy_rej"].append(_mvals("cond_energy_rej"))
        agg["abs_dH_mean"].append(_mvals("abs_dH_mean"))
        agg["abs_dH_p95"].append(_mvals("abs_dH_p95"))
        agg["n_bounces_mean"].append(_mvals("n_bounces_mean"))
        agg["bounce_traj_frac"].append(_mvals("bounce_traj_frac"))
        agg["bounce_fail_frac"].append(_mvals("bounce_fail_frac"))
        agg["grad_U"].append(_mvals("grad_U", 0))
        agg["G"].append(_mvals("G", 0))
        agg["n_steps_mean"].append(_mvals("n_steps_mean"))

    mean_total_grad_U = float(np.mean([r["total_grad_U"] for r in results]))
    mean_total_G      = float(np.mean([r["total_G"]      for r in results]))

    return dict(
        arm=arm, label=label,
        pf_arr=pf_arr, beta_arr=beta_arr,
        mean_pf=mean_pf, cov_pf=cov_pf,
        mean_beta=mean_beta, std_beta=std_beta,
        bias_beta=bias_beta, cov_beta=cov_beta,
        mean_levels=float(lv_arr.mean()),
        agg=agg, results=results,
        mean_total_grad_U=mean_total_grad_U,
        mean_total_G=mean_total_G,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 0 — bounce sanity on truncated Gaussian
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp0_truncated_gaussian(
    b: float = 2.0,
    n_chains: int = 100,
    n_burn: int = 500,
    n_test: int = 1000,
    ctrl_cfg: Optional[ControllerConfig] = None,
    seed: int = 42,
) -> Dict:
    """
    Constrained HMC on N(0,I₂) truncated to {x₁ ≤ b}.

    Arms A/B/C run as constrained MCMC (b_k = 0 fixed, no SuS).
    Analytic reference: x₁ ~ TruncN(0,1,(-∞,b]), x₂ ~ N(0,1).

    Checks:
      (i)  Leak rate A/B: fraction of proposals landing outside {x₁≤b}.
      (ii) ARM C leak rate: should be ≈ 0.
      (iii) KS on x₁, x₂ marginals of ARM C samples vs analytic.
      (iv) Bounce counts, eval costs.
    """
    if ctrl_cfg is None:
        ctrl_cfg = ControllerConfig()

    lsf    = HalfSpaceLSF(b=b, d=2)
    target = CorrelatedGaussianTarget(rho=0.0, d=2)
    rng    = np.random.default_rng(seed)

    # Analytic reference samples (large N for KS)
    N_ref = 50_000
    x1_ref = truncnorm.rvs(-1e3, b, loc=0.0, scale=1.0, size=N_ref,
                            random_state=np.random.default_rng(seed + 1))
    x2_ref = rng.standard_normal(N_ref)

    def _run_constrained(arm_label: str) -> Dict:
        rng_arm   = np.random.default_rng(seed + hash(arm_label) % 10000)
        chains    = np.zeros((n_chains, 2), dtype=np.float64)
        b_k       = 0.0          # fixed: {G <= 0} = {x1 <= b}

        n_leak = 0;  n_prop = 0
        bounce_counts: List[int] = []
        n_grad_U_tot = 0;  n_G_tot = 0

        for phase in ("burn", "test"):
            n_iters   = n_burn if phase == "burn" else n_test
            samples   = [] if phase == "test" else None

            for _ in range(n_iters):
                for i in range(n_chains):
                    p_init = target.sample_momentum(rng_arm)

                    if arm_label == "A":
                        q_new, p_new, st = integrate_A(
                            chains[i], p_init, target, DT_A, N_STEPS_A)
                    elif arm_label == "B":
                        q_new, p_new, st = integrate_B(
                            chains[i], p_init, target,
                            lsf.evaluate, lsf.grad_G,
                            b_k=b_k, t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg,
                        )
                    else:  # C
                        q_new, p_new, st = integrate_C(
                            chains[i], p_init, target,
                            lsf.evaluate, lsf.grad_G,
                            b_k=b_k, t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg,
                        )

                    if phase == "test":
                        n_grad_U_tot += st.n_grad_U
                        n_G_tot      += st.n_G
                        bounce_counts.append(st.n_bounces)

                    geom_ok = float(lsf.evaluate(q_new)) <= b_k
                    H0  = target.H(chains[i], p_init)
                    H1  = target.H(q_new, p_new)
                    ok, _ = _mh_accept(H0, H1, rng_arm)

                    if phase == "test":
                        n_prop += 1
                        if not geom_ok:
                            n_leak += 1

                    if geom_ok and ok:
                        chains[i] = q_new

                if phase == "test" and samples is not None:
                    samples.append(chains.copy())

            if phase == "test":
                all_samples = np.concatenate(samples, axis=0)   # [n_chains*n_test, 2]

        leak_rate = n_leak / max(n_prop, 1)

        ks_x1 = ks_2samp(all_samples[:, 0], x1_ref)
        ks_x2 = ks_2samp(all_samples[:, 1], x2_ref)

        return dict(
            arm=arm_label,
            leak_rate=leak_rate,
            ks_x1_stat=ks_x1.statistic, ks_x1_pval=ks_x1.pvalue,
            ks_x2_stat=ks_x2.statistic, ks_x2_pval=ks_x2.pvalue,
            mean_bounces=float(np.mean(bounce_counts)) if bounce_counts else 0.0,
            frac_bounced=float(np.mean([b > 0 for b in bounce_counts])) if bounce_counts else 0.0,
            n_grad_U=n_grad_U_tot,
            n_G=n_G_tot,
            n_samples=len(all_samples),
        )

    _log("\n[Exp 0] Truncated Gaussian sanity check")
    results_0 = {}
    for arm in ("A", "B", "C"):
        _log(f"  arm={arm} ...")
        results_0[arm] = _run_constrained(arm)

    _log("\n  Exp 0 summary:")
    _log(f"  {'arm':>3}  {'leak_rate':>10}  {'KS_x1':>8}  {'KS_x2':>8}  "
         f"{'bounces_mean':>13}  {'frac_bounced':>13}  {'grad_U':>8}  {'G':>6}")
    for arm in ("A", "B", "C"):
        r = results_0[arm]
        _log(f"  {arm:>3}  {r['leak_rate']:>10.4f}  {r['ks_x1_stat']:>8.4f}  "
             f"{r['ks_x2_stat']:>8.4f}  {r['mean_bounces']:>13.3f}  "
             f"{r['frac_bounced']:>13.3f}  {r['n_grad_U']:>8d}  {r['n_G']:>6d}")

    return results_0


# ═══════════════════════════════════════════════════════════════════════════════
# Exp 1 — 3-arm SuS on Thaler linear LSF
# ═══════════════════════════════════════════════════════════════════════════════

def run_exp1_linear_lsf(
    beta: float,
    rho: float,
    n_rep: int,
    ctrl_cfg: ControllerConfig,
    arms: Tuple[str, ...] = ("A", "B", "C"),
    d: int = 2,
) -> Dict:
    label = f"beta={beta}_rho={rho}_d={d}"
    _log(f"\n[Exp 1  {label}  n_rep={n_rep}]")
    lsf    = ThalerLinearLSF(beta=beta, rho=rho, d=d)
    target = CorrelatedGaussianTarget(rho=rho, d=d)
    all_arm_results = {}
    for arm in arms:
        all_arm_results[arm] = run_replications(
            arm, target, lsf, ctrl_cfg, n_rep, label=label)
    return dict(beta=beta, rho=rho, d=d, lsf=lsf, arms=all_arm_results)


# ═══════════════════════════════════════════════════════════════════════════════
# Console output helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_main_table(exp1_results: List[Dict], lsf_ref_map: Dict) -> None:
    sep = "─" * 80
    _log("\n" + sep)
    _log(f"{'Config':<12} {'label':<20} {'beta_hat':>9} {'beta_ref':>9} "
         f"{'bias':>7} {'COV_beta':>9} {'grad_U/run':>11} {'G/run':>9} {'lv':>5}")
    _log(sep)
    for exp in exp1_results:
        key  = f"b{exp['beta']}_r{exp['rho']}"
        bref = lsf_ref_map.get(key, float("nan"))
        for arm, r in exp["arms"].items():
            _log(
                f"  arm {arm:<8} {r['label']:<20} "
                f"{r['mean_beta']:>9.4f} {bref:>9.4f} "
                f"{r['bias_beta']:>+7.4f} {r['cov_beta']:>9.3f} "
                f"{r['mean_total_grad_U']:>11.3e} {r['mean_total_G']:>9.3e} "
                f"{r['mean_levels']:>5.1f}"
            )
    _log(sep)


def print_mechanism_table(exp1_results: List[Dict]) -> None:
    _log("\nMechanism table — per-level fractions (mean over reps):")
    _log(f"  {'arm':<4} {'label':<18} {'lv':>3}  "
         f"{'geom_rej':>9} {'ener_rej':>9} {'accept':>8} "
         f"{'cond_ener':>10} {'bounces_mean':>13} {'bounce_traj':>12}")
    for exp in exp1_results:
        for arm, r in exp["arms"].items():
            agg = r["agg"]
            for lv in range(len(agg["geom_rej"])):
                _log(
                    f"  {arm:<4} {r['label']:<18} L{lv+1:>1}  "
                    f"{agg['geom_rej'][lv]:>9.3f} "
                    f"{agg['energy_rej'][lv]:>9.3f} "
                    f"{agg['accept_rate'][lv]:>8.3f} "
                    f"{agg['cond_energy_rej'][lv]:>10.3f} "
                    f"{agg['n_bounces_mean'][lv]:>13.3f} "
                    f"{agg['bounce_traj_frac'][lv]:>12.3f}"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def _arm_color(arm: str) -> str:
    return {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "C'": "#C44E52"}.get(arm, "gray")


def plot_mechanism_bars(exp1_results: List[Dict], out_dir: Path, tag: str) -> None:
    for exp in exp1_results:
        lbl = f"beta{exp['beta']}_rho{exp['rho']}"
        arms = list(exp["arms"].keys())
        n_arms = len(arms)
        max_lv = max(len(exp["arms"][a]["agg"]["geom_rej"]) for a in arms)
        fig, axes = plt.subplots(1, n_arms, figsize=(5 * n_arms, 4), squeeze=False)
        for col, arm in enumerate(arms):
            ax  = axes[0][col]
            agg = exp["arms"][arm]["agg"]
            x   = np.arange(max_lv)
            g   = [agg["geom_rej"][lv]   if lv < len(agg["geom_rej"])   else 0 for lv in x]
            e   = [agg["energy_rej"][lv] if lv < len(agg["energy_rej"]) else 0 for lv in x]
            a   = [agg["accept_rate"][lv] if lv < len(agg["accept_rate"]) else 0 for lv in x]
            ax.bar(x, g, color="#DD8452", alpha=0.85, label="geom reject")
            ax.bar(x, e, bottom=g, color="#4C72B0", alpha=0.85, label="energy reject")
            ax.bar(x, a, bottom=[gi + ei for gi, ei in zip(g, e)],
                   color="#55A868", alpha=0.85, label="accepted")
            ax.set_xticks(x); ax.set_xticklabels([f"L{i+1}" for i in x])
            ax.set_ylim(0, 1.05); ax.set_title(f"Arm {arm}  {lbl}", fontsize=9)
            ax.set_ylabel("fraction")
            if col == 0:
                ax.legend(fontsize=8)
        plt.tight_layout()
        path = out_dir / f"mechanism_bars_{lbl}{tag}.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        _log(f"  Saved {path.name}")


def plot_beta_boxplot(exp1_results: List[Dict], out_dir: Path, tag: str) -> None:
    for exp in exp1_results:
        lbl  = f"beta{exp['beta']}_rho{exp['rho']}"
        arms = list(exp["arms"].keys())
        data = [exp["arms"][a]["beta_arr"] for a in arms]
        fig, ax = plt.subplots(figsize=(5, 5))
        bp = ax.boxplot(data, labels=arms, patch_artist=True)
        for patch, arm in zip(bp["boxes"], arms):
            patch.set_facecolor(_arm_color(arm)); patch.set_alpha(0.7)
        ax.axhline(exp["beta"], color="red", ls="--", lw=1.2,
                   label=f"beta_ref={exp['beta']}")
        ax.set_ylabel("beta_hat"); ax.set_title(f"beta_hat distribution — {lbl}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = out_dir / f"beta_boxplot_{lbl}{tag}.png"
        fig.savefig(path, dpi=150); plt.close(fig)
        _log(f"  Saved {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# Record writer
# ═══════════════════════════════════════════════════════════════════════════════

def write_record(
    exp0_results: Dict,
    exp1_results: List[Dict],
    ctrl_cfg: ControllerConfig,
    n_rep: int,
) -> None:
    lines = [
        f"# Geometry-aware adaptive step + Barrier Bouncing in HMC-SuS",
        f"",
        f"## Controller settings",
        f"",
        f"| param | value |",
        f"|-------|-------|",
        f"| c_cfl | {ctrl_cfg.c_cfl} |",
        f"| eps_E | {ctrl_cfg.eps_E} |",
        f"| tau_safety | {ctrl_cfg.tau_safety} |",
        f"| dt_min | {ctrl_cfg.dt_min} |",
        f"| dt_max | {ctrl_cfg.dt_max} |",
        f"| dt_max_change | {ctrl_cfg.dt_max_change} |",
        f"| n_bounce_max | {ctrl_cfg.n_bounce_max} |",
        f"| tol_hit | {ctrl_cfg.tol_hit} |",
        f"| bisect_max_iter | {ctrl_cfg.bisect_max_iter} |",
        f"| dt_seed (baseline dt) | {DT_A} |",
        f"| t_f (integration time) | {T_F} |",
        f"| SUS_N | {SUS_N} |  SUS_P0 | {SUS_P0} |  SUS_BURN_IN | {SUS_BURN_IN} |",
        f"",
        f"**Correctness note**: arms B and C use state-dependent per-step dt,",
        f"breaking leapfrog reversibility/volume-preservation → approximate samplers.",
        f"The dt_geom regulariser (dt_seed floor) ensures the step never stalls at",
        f"m→0 but increases bias vs the pure CFL limit. C' (fixed+bounce) is the",
        f"exactly-correct reference for isolating adaptivity bias.",
        f"",
        f"## Exp 0 — Bounce sanity on truncated Gaussian",
        f"",
        f"Target: N(0,I₂) truncated to {{x₁ ≤ {2.0}}}.",
        f"",
        f"| arm | leak_rate | KS_x1 | KS_x1_pval | KS_x2 | KS_x2_pval | bounces_mean | frac_bounced | grad_U | G |",
        f"|-----|-----------|-------|------------|-------|------------|-------------|--------------|--------|---|",
    ]
    for arm in ("A", "B", "C"):
        if arm not in exp0_results:
            continue
        r = exp0_results[arm]
        lines.append(
            f"| {arm} | {r['leak_rate']:.4f} | {r['ks_x1_stat']:.4f} | "
            f"{r['ks_x1_pval']:.3f} | {r['ks_x2_stat']:.4f} | "
            f"{r['ks_x2_pval']:.3f} | {r['mean_bounces']:.3f} | "
            f"{r['frac_bounced']:.3f} | {r['n_grad_U']} | {r['n_G']} |"
        )

    lines += ["", f"## Exp 1 — 3-arm SuS on Thaler linear LSF  (N_rep={n_rep})", ""]

    for exp in exp1_results:
        beta = exp["beta"];  rho = exp["rho"]
        lsf  = exp["lsf"]
        lines += [
            f"### beta={beta}, rho={rho}  (Pf_ref={lsf.pf_ref:.4e}, beta_ref={beta})",
            f"",
            f"#### Summary table",
            f"",
            f"| arm | beta_hat | std | bias | COV_beta | grad_U/run | G/run | mean_lv |",
            f"|-----|----------|-----|------|----------|------------|-------|---------|",
        ]
        for arm, r in exp["arms"].items():
            lines.append(
                f"| {arm} | {r['mean_beta']:.4f} | {r['std_beta']:.4f} | "
                f"{r['bias_beta']:+.4f} | {r['cov_beta']:.3f} | "
                f"{r['mean_total_grad_U']:.3e} | {r['mean_total_G']:.3e} | "
                f"{r['mean_levels']:.2f} |"
            )

        lines += ["", f"#### Per-level mechanism", ""]
        for arm, r in exp["arms"].items():
            agg = r["agg"]
            lines += [
                f"**Arm {arm}**",
                f"",
                f"| lv | geom_rej | energy_rej | accept | cond_ener_rej | bounces_mean | bounce_traj | grad_U | G | n_steps_mean |",
                f"|----|----------|------------|--------|---------------|-------------|-------------|--------|---|--------------|",
            ]
            for lv in range(len(agg["geom_rej"])):
                lines.append(
                    f"| L{lv+1} | {agg['geom_rej'][lv]:.3f} | "
                    f"{agg['energy_rej'][lv]:.3f} | {agg['accept_rate'][lv]:.3f} | "
                    f"{agg['cond_energy_rej'][lv]:.3f} | "
                    f"{agg['n_bounces_mean'][lv]:.3f} | {agg['bounce_traj_frac'][lv]:.3f} | "
                    f"{agg['grad_U'][lv]:.0f} | {agg['G'][lv]:.0f} | "
                    f"{agg['n_steps_mean'][lv]:.1f} |"
                )
            lines.append("")

        lines += [
            f"#### What this tells us (H_B and H_C)",
            f"",
            f"H_B (refinement alone): ARM B vs ARM A — if geom_rej rates are similar,",
            f"refinement did NOT reduce geometric rejection (consistent with the hypothesis",
            f"that the true Hamiltonian trajectory exits F_k regardless of step size).",
            f"",
            f"H_C (bouncing): ARM C vs ARM A/B — if ARM C's geom_rej ≈ 0 and A/B's are",
            f"substantial, bouncing successfully prevents geometric rejection.",
            f"Cost comparison: C pays extra G/grad_G evals for bisection; net efficiency",
            f"= (COV improvement) / (extra eval cost).",
            f"",
        ]

    with open(REC_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    _log(f"  Wrote {REC_PATH}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main(n_rep: int = 2, skip_exp0: bool = False, d: int = 2) -> None:
    _open_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if n_rep <= 5 else ""

    ctrl_cfg = ControllerConfig(
        c_cfl=0.5, eps_E=0.05, dt_min=1e-3, dt_max=0.5,
        dt_max_change=5.0, n_bounce_max=5, tol_hit=1e-6, bisect_max_iter=30,
        bounce_fixed_step=False,  # C' ablation: default OFF
    )

    _log("=" * 70)
    _log(f"Geometry-bounce experiment  n_rep={n_rep}  tag={tag or 'full'}")
    _log(f"DT_A={DT_A}  N_STEPS_A={N_STEPS_A}  T_F={T_F}")
    _log(f"SUS_N={SUS_N}  SUS_P0={SUS_P0}  SUS_BURN_IN={SUS_BURN_IN}")
    _log(f"ctrl: c_cfl={ctrl_cfg.c_cfl}  dt_seed=DT_A={DT_A}  "
         f"dt_min={ctrl_cfg.dt_min}  dt_max={ctrl_cfg.dt_max}")
    _log("=" * 70)

    t_start = time.perf_counter()

    # ── Exp 0 ─────────────────────────────────────────────────────────────
    if not skip_exp0:
        exp0_results = run_exp0_truncated_gaussian(
            b=2.0, n_chains=100, n_burn=200, n_test=500, ctrl_cfg=ctrl_cfg)
    else:
        exp0_results = {}

    # ── Exp 1 ─────────────────────────────────────────────────────────────
    exp1_configs = [
        (3.5, 0.00),   # Thaler §5.1 base case
        (4.0, 0.75),   # higher beta, mild correlation
    ]
    exp1_results = []
    lsf_ref_map  = {}

    for beta, rho in exp1_configs:
        _log(f"\n{'='*70}")
        _log(f"Exp 1: beta={beta}  rho={rho}")
        r = run_exp1_linear_lsf(beta=beta, rho=rho, n_rep=n_rep,
                                ctrl_cfg=ctrl_cfg, arms=("A", "B", "C"), d=d)
        exp1_results.append(r)
        lsf_ref_map[f"b{beta}_r{rho}"] = float(beta)

    _log(f"\nTotal wall time: {time.perf_counter()-t_start:.1f}s")

    # ── Print tables ───────────────────────────────────────────────────────
    print_main_table(exp1_results, lsf_ref_map)
    print_mechanism_table(exp1_results)

    # ── Plots + save ───────────────────────────────────────────────────────
    _log("\n[Saving outputs]")
    plot_mechanism_bars(exp1_results, OUT_DIR, tag)
    plot_beta_boxplot(exp1_results, OUT_DIR, tag)

    np.savez(
        OUT_DIR / f"exp1_raw{tag}.npz",
        **{
            f"{exp['beta']}_{exp['rho']}_{arm}_beta": exp["arms"][arm]["beta_arr"]
            for exp in exp1_results
            for arm in exp["arms"]
        },
    )
    _log(f"  Saved exp1_raw{tag}.npz")

    if not skip_exp0:
        import json
        with open(OUT_DIR / f"exp0_results{tag}.json", "w") as fj:
            json.dump(
                {arm: {k: v for k, v in r.items() if not isinstance(v, np.ndarray)}
                 for arm, r in exp0_results.items()},
                fj, indent=2,
            )
        _log(f"  Saved exp0_results{tag}.json")

    if n_rep > 2:
        write_record(exp0_results, exp1_results, ctrl_cfg, n_rep)

    _log(f"\nDone.  All outputs in {OUT_DIR}")
    _close_log()


if __name__ == "__main__":
    full = "--full" in sys.argv
    skip = "--skip-exp0" in sys.argv
    main(n_rep=30 if full else 2, skip_exp0=skip)
