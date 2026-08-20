"""
experiment_bb_hnn.py
─────────────────────────────────────────────────────────────────────────────
Barrier bouncing driven by an HNN gradient (design a-1).

Two arms, linear LSF, beta=3.5, rho=0, d=2 (matches Chapter 6's Arm C setup
exactly, for direct comparison):

  Arm C      : analytic gradient + bouncing (existing integrate_C, unchanged,
               imported from experiment_geometry_bounce.py)
  Arm C_HNN  : SAME bouncing mechanism (bisection hitting-time on the true G,
               specular reflection, both analytic and unchanged), but the
               leapfrog force/kinetic terms come from the trained HNN's FULL
               vector field:

                   dq/dt = dH_theta/dp
                   dp/dt = -dH_theta/dq

               both re-differentiated at every micro-step -- exactly matching
               leapfrog_hnn / hnn_vector_field in compare_pinn_hnn.py (the
               Chapter 4/5 HNN integrator). This is NOT the force-only
               variant: dq/dt does not use p directly.

MH acceptance uses the TRUE analytic Hamiltonian (target.H) in both arms --
only the proposal-generating dynamics differ.

Two judgment calls made where "match leapfrog_hnn" doesn't fully specify
behaviour (leapfrog_hnn is fixed-step, unconstrained, and has no controller
or bounce event -- neither exists in the Chapter 4/5 code to copy from):

  1. SE-Euler energy-safety companion (used only to cap dt as a safety net,
     never primary -- see _compute_next_dt). Kept "free" (no extra HNN
     calls) by using q_eul = q + dt*p_eul (p as drift velocity for this
     diagnostic only), mirroring the analytic companion's math with dpdt0
     substituted for -grad_U. Does not touch the main HNN trajectory update.
  2. After a bounce/reflection, dp/dt is recomputed FRESH at the
     post-reflection (q_hit, p_ref) rather than reusing the pre-reflection
     value -- because unlike the analytic separable case (where dp/dt
     depends only on q, so reuse across the reflection is exact), the HNN's
     dp/dt genuinely depends on p, so a cached pre-reflection value would be
     stale. This costs one extra HNN call per bounce and is the more
     "honest" choice.

Run from project root:
    python -u pinn_hmc/experiment_bb_hnn.py --stage1
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch
from scipy.stats import norm as Norm

from pinn_hmc.model import HamiltonianNN, HNNConfig
from pinn_hmc.experiment_geometry_bounce import (
    ThalerLinearLSF,
    CorrelatedGaussianTarget,
    ControllerConfig,
    PropStats,
    integrate_C,
    _compute_next_dt,
    _reflect_momentum,
    _mh_accept,
    DT_A,
    T_F,
    SUS_N,
    SUS_P0,
    SUS_BURN_IN,
    SUS_MAX_LEVELS,
)

OUT_DIR = Path("results/bb_hnn")
HNN_CHECKPOINT = "results/smoke_yoshida_hnn/hnn_gaussian.pt"


def make_ctrl_cfg() -> ControllerConfig:
    """Identical to the Chapter 6 beta=3.5 rho=0 three-arm study."""
    return ControllerConfig(
        c_cfl=0.5, eps_E=0.05, dt_min=1e-3, dt_max=0.5,
        dt_max_change=5.0, n_bounce_max=5, tol_hit=1e-6, bisect_max_iter=30,
        bounce_fixed_step=False,
    )


def load_hnn(checkpoint: str = HNN_CHECKPOINT,
             device: Optional[torch.device] = None) -> torch.nn.Module:
    if device is None:
        device = torch.device("cpu")
    model = HamiltonianNN(HNNConfig(dim=2, hidden_dim=128, depth=4)).to(device)
    sd = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# HNN full vector field (exactly matching compare_pinn_hnn.py::hnn_vector_field)
# ═══════════════════════════════════════════════════════════════════════════════

def hnn_vector_field(model, q: np.ndarray, p: np.ndarray, device: torch.device
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """dq/dt = dH_theta/dp, dp/dt = -dH_theta/dq. Single-vector numpy in/out."""
    q_t = torch.as_tensor(q, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    p_t = torch.as_tensor(p, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
    H_hat = model(q_t, p_t)
    dH_dq, = torch.autograd.grad(H_hat.sum(), q_t, retain_graph=True)
    dH_dp, = torch.autograd.grad(H_hat.sum(), p_t)
    dqdt = dH_dp.detach().squeeze(0).numpy().astype(np.float64)
    dpdt = (-dH_dq).detach().squeeze(0).numpy().astype(np.float64)
    return dqdt, dpdt


def _verlet_step_hnn(q, p, dpdt0, model, dt, device):
    """
    Single adaptive-dt step, full HNN vector field. Same kick-drift-kick
    template as leapfrog_hnn's inner loop (dpdt0 cached from the previous
    step's end -- valid because nothing external touches p between
    consecutive non-bounce steps, exactly mirroring how leapfrog_hnn reuses
    the interior kick across its fixed-size steps).
    """
    p_half = p + 0.5 * dt * dpdt0
    dqdt, _ = hnn_vector_field(model, q, p_half, device)
    q_new = q + dt * dqdt
    _, dpdt1 = hnn_vector_field(model, q_new, p_half, device)
    p_new = p_half + 0.5 * dt * dpdt1
    return q_new, p_new, dpdt1


def _bisect_crossing_hnn(q_n, p_n, dpdt0_n, model, G_fn, b_k, dt_full, cfg, device):
    """
    HNN analog of _bisect_crossing. Unlike the analytic case (where q_mid(dt)
    is free once g0 is cached, since dq/dt=p_half needs no extra evaluation),
    here dq/dt = dH/dp(q, p_half) is a nonlinear function of p_half that must
    be re-evaluated at every candidate dt_mid -- one extra HNN call per
    bisection iteration, NOT free. This is exactly the honest extra cost
    Arm C_HNN is meant to expose.
    """
    dt_lo, dt_hi = 0.0, dt_full
    dt_mid = 0.5 * (dt_lo + dt_hi)
    G_mid = b_k
    n_G = 0
    n_hnn = 0
    converged = False

    for _ in range(cfg.bisect_max_iter):
        dt_mid = 0.5 * (dt_lo + dt_hi)
        p_half = p_n + 0.5 * dt_mid * dpdt0_n
        dqdt, _ = hnn_vector_field(model, q_n, p_half, device); n_hnn += 1
        q_mid = q_n + dt_mid * dqdt
        G_mid = float(G_fn(q_mid)); n_G += 1
        m_mid = b_k - G_mid
        if abs(m_mid) < cfg.tol_hit:
            converged = True
            break
        if m_mid > 0.0:
            dt_lo = dt_mid
        else:
            dt_hi = dt_mid

    dt_hit = dt_mid
    p_half = p_n + 0.5 * dt_hit * dpdt0_n
    dqdt, _ = hnn_vector_field(model, q_n, p_half, device); n_hnn += 1
    q_hit = q_n + dt_hit * dqdt
    _, dpdt_hit = hnn_vector_field(model, q_hit, p_half, device); n_hnn += 1
    p_hit = p_half + 0.5 * dt_hit * dpdt_hit
    return q_hit, p_hit, dt_hit, G_mid, n_G, n_hnn, converged


def integrate_C_hnn(q0, p0, model, G_fn, grad_G_fn, b_k, t_f, dt_seed, cfg, device):
    """
    HNN analog of integrate_C. Bounce/reflection and G/grad_G checks are
    unchanged (analytic, identical calls to _reflect_momentum and the LSF's
    grad_G); only the force/kinetic terms driving the leapfrog come from the
    HNN's full vector field.
    """
    q, p = q0.copy(), p0.copy()
    _, dpdt0 = hnn_vector_field(model, q, p, device)
    n_hnn = 1
    G_val = float(G_fn(q)); n_G = 1; extra_G = 0

    dt = dt_seed
    t = 0.0; n_steps = 0; n_bounces = 0
    bounce_failure = False
    dt_list: List[float] = []

    while t < t_f - 1e-12:
        dt = min(dt, t_f - t)

        # SE companion (free -- see module docstring, judgment call #1)
        p_eul = p + dt * dpdt0
        q_eul = q + dt * p_eul

        q_new, p_new, dpdt1 = _verlet_step_hnn(q, p, dpdt0, model, dt, device)
        n_hnn += 2

        G_new = float(G_fn(q_new)); n_G += 1
        m_n = b_k - G_val
        m_n1 = b_k - G_new

        if m_n > 0.0 and m_n1 < 0.0:
            if n_bounces >= cfg.n_bounce_max:
                bounce_failure = True
                break

            (q_hit, p_hit, dt_hit, G_hit,
             n_bis_G, n_bis_hnn, conv) = _bisect_crossing_hnn(
                q, p, dpdt0, model, G_fn, b_k, dt, cfg, device,
            )
            n_hnn += n_bis_hnn
            extra_G += n_bis_G

            grad_G_hit = grad_G_fn(q_hit)
            p_ref = _reflect_momentum(p_hit, grad_G_hit)

            # Judgment call #2: recompute fresh post-reflection (see docstring)
            _, dpdt_post = hnn_vector_field(model, q_hit, p_ref, device)
            n_hnn += 1

            t += dt_hit
            dt_list.append(dt_hit)
            n_steps += 1; n_bounces += 1

            q, p, dpdt0 = q_hit, p_ref, dpdt_post
            G_val = b_k

            dt = min(dt_seed, t_f - t) if t < t_f else 0.0
            continue

        grad_G_new = grad_G_fn(q_new)
        if cfg.bounce_fixed_step:
            dt_next = dt_seed
        else:
            dt_next = _compute_next_dt(
                q_new, p_new, q_eul, p_eul, dt, G_new, grad_G_new, b_k, dt_seed, cfg,
            )

        t += dt; dt_list.append(dt)
        q, p, dpdt0, G_val = q_new, p_new, dpdt1, G_new
        dt = dt_next; n_steps += 1

    dt_arr = np.asarray(dt_list)
    s = PropStats(
        n_grad_U=n_hnn, n_G=n_G + extra_G, n_steps=n_steps,
        n_bounces=n_bounces, bounce_failure=bounce_failure,
        dt_min_used=float(dt_arr.min()) if n_steps else dt_seed,
        dt_max_used=float(dt_arr.max()) if n_steps else dt_seed,
        dt_mean=float(dt_arr.mean()) if n_steps else dt_seed,
    )
    return q, p, s


# ═══════════════════════════════════════════════════════════════════════════════
# SuS driver for Arm C vs Arm C_HNN (timed)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_bb_result(P_F, levels, thresholds, geom_rej, accept_rate, abs_dH_mean,
                      n_bounces_mean, proposal_time_mean, proposal_times_all) -> Dict:
    pf = max(P_F, 1e-300)
    beta_hat = float(-Norm.ppf(pf))
    times_arr = np.asarray(proposal_times_all, dtype=np.float64)
    return dict(
        P_F=P_F, beta_hat=beta_hat, levels=levels, thresholds=thresholds,
        per_level_geom_rej=geom_rej,
        per_level_accept_rate=accept_rate,
        per_level_abs_dH_mean=abs_dH_mean,
        per_level_n_bounces_mean=n_bounces_mean,
        per_level_proposal_time_mean=proposal_time_mean,
        proposal_times=times_arr,
        mean_proposal_time=float(times_arr.mean()) if times_arr.size else 0.0,
        std_proposal_time=float(times_arr.std()) if times_arr.size else 0.0,
        n_proposals=int(times_arr.size),
    )


def _run_one_sus_bb(
    arm: str,                          # "C" or "C_HNN"
    target: CorrelatedGaussianTarget,
    lsf: ThalerLinearLSF,
    ctrl_cfg: ControllerConfig,
    seed: int,
    model=None,
    device=None,
    rep_label: str = "",
    verbose: bool = True,
) -> Dict:
    rng = np.random.default_rng(seed)
    d = target.d
    N = SUS_N
    p0f = SUS_P0
    n_seeds = max(1, int(round(p0f * N)))
    spc = math.ceil(N / n_seeds)

    thresholds: List[float] = []
    lv_geom_rej: List[float] = []
    lv_accept_rate: List[float] = []
    lv_abs_dH_mean: List[float] = []
    lv_n_bounces_mean: List[float] = []
    lv_proposal_time_mean: List[float] = []
    proposal_times_all: List[float] = []

    def _propose(q_i, p_init, b_k):
        t0 = time.perf_counter()
        if arm == "C":
            q_new, p_new, st = integrate_C(
                q_i, p_init, target, lsf.evaluate, lsf.grad_G,
                b_k=b_k, t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg,
            )
        else:
            q_new, p_new, st = integrate_C_hnn(
                q_i, p_init, model, lsf.evaluate, lsf.grad_G,
                b_k=b_k, t_f=T_F, dt_seed=DT_A, cfg=ctrl_cfg, device=device,
            )
        proposal_times_all.append(time.perf_counter() - t0)
        return q_new, p_new, st

    # ── Burn-in (unconstrained, no geometric check) ─────────────────────────
    q = np.zeros((N, d), dtype=np.float64)
    for _ in range(SUS_BURN_IN):
        for i in range(N):
            p_init = target.sample_momentum(rng)
            q_new, p_new, st = _propose(q[i], p_init, b_k=1e9)
            H0 = target.H(q[i], p_init)
            H1 = target.H(q_new, p_new)
            ok, _ = _mh_accept(H0, H1, rng)
            if ok:
                q[i] = q_new

    # ── Conditional levels ───────────────────────────────────────────────────
    for level in range(SUS_MAX_LEVELS):
        g_vals = lsf.evaluate(q)
        threshold = float(np.quantile(g_vals, p0f))
        thresholds.append(threshold)
        if verbose:
            print(f"    {rep_label} arm={arm} L{level+1}: threshold={threshold:.4f}", flush=True)

        if threshold <= 0.0:
            n_fail = int((g_vals <= 0.0).sum())
            P_F = (p0f ** level) * (n_fail / N)
            return _build_bb_result(
                P_F, level + 1, thresholds, lv_geom_rej, lv_accept_rate,
                lv_abs_dH_mean, lv_n_bounces_mean, lv_proposal_time_mean,
                proposal_times_all,
            )

        seed_idx = np.argsort(g_vals)[:n_seeds]
        chains = q[seed_idx].copy()
        chain_snapshots = [chains.copy()]

        lv_n = lv_geom = lv_acpt = 0
        dH_list: List[float] = []
        bounce_counts: List[int] = []
        lv_times: List[float] = []

        for _ in range(spc - 1):
            for i in range(n_seeds):
                p_init = target.sample_momentum(rng)
                q_new, p_new, st = _propose(chains[i], p_init, b_k=threshold)
                lv_times.append(proposal_times_all[-1])
                bounce_counts.append(st.n_bounces)

                if st.bounce_failure:
                    geom_ok = False
                else:
                    geom_ok = float(lsf.evaluate(q_new)) <= threshold

                H0 = target.H(chains[i], p_init)
                H1 = target.H(q_new, p_new)

                if abs(H1 - H0) > 200.0:
                    geom_ok = False

                if geom_ok:
                    mh_ok, dH = _mh_accept(H0, H1, rng)
                    dH_list.append(dH)
                    if mh_ok:
                        chains[i] = q_new
                        lv_acpt += 1
                else:
                    lv_geom += 1
                    rng.random()

                lv_n += 1

            chain_snapshots.append(chains.copy())

        lv_geom_rej.append(lv_geom / max(lv_n, 1))
        lv_accept_rate.append(lv_acpt / max(lv_n, 1))
        lv_abs_dH_mean.append(float(np.mean(dH_list)) if dH_list else 0.0)
        lv_n_bounces_mean.append(float(np.mean(bounce_counts)) if bounce_counts else 0.0)
        lv_proposal_time_mean.append(float(np.mean(lv_times)) if lv_times else 0.0)

        q = np.stack(chain_snapshots, axis=1).reshape(-1, d)[:N]

    g_final = lsf.evaluate(q)
    n_fail = int((g_final <= 0.0).sum())
    P_F = (p0f ** SUS_MAX_LEVELS) * (n_fail / N)
    return _build_bb_result(
        P_F, SUS_MAX_LEVELS, thresholds, lv_geom_rej, lv_accept_rate,
        lv_abs_dH_mean, lv_n_bounces_mean, lv_proposal_time_mean,
        proposal_times_all,
    )


def stage1(beta: float = 3.5, seed: int = 0) -> None:
    print("=" * 70)
    print("STAGE 1 — Arm C vs Arm C_HNN smoke test (n_rep=1)")
    print("=" * 70)

    ctrl_cfg = make_ctrl_cfg()
    target = CorrelatedGaussianTarget(rho=0.0)   # d=2 default
    lsf = ThalerLinearLSF(beta=beta, rho=0.0)    # d=2 default
    device = torch.device("cpu")
    model = load_hnn(device=device)
    print(f"\nbeta={beta}  rho=0  d=2  pf_ref={lsf.pf_ref:.4e}")
    print(f"HNN checkpoint: {HNN_CHECKPOINT}")

    results = {}
    for arm in ("C", "C_HNN"):
        print(f"\n--- Arm {arm} ---")
        t0 = time.perf_counter()
        r = _run_one_sus_bb(arm, target, lsf, ctrl_cfg, seed=seed,
                             model=model, device=device, rep_label="rep01/1")
        wall = time.perf_counter() - t0
        r["wall_time"] = wall
        results[arm] = r
        print(f"  done in {wall:.1f}s  beta_hat={r['beta_hat']:.4f}  "
              f"P_F={r['P_F']:.4e}  levels={r['levels']}")

    print("\n" + "=" * 100)
    print("SMOKE COMPARISON")
    print("=" * 100)
    print(f"{'arm':<8} {'beta_hat':>10} {'P_F':>12} {'levels':>7} "
          f"{'mean|dH|':>10} {'mean_prop_t(ms)':>16} {'wall(s)':>9}")
    for arm, r in results.items():
        mean_dH = float(np.mean(r["per_level_abs_dH_mean"])) if r["per_level_abs_dH_mean"] else 0.0
        print(f"{arm:<8} {r['beta_hat']:>10.4f} {r['P_F']:>12.4e} {r['levels']:>7} "
              f"{mean_dH:>10.4e} {r['mean_proposal_time']*1e3:>16.3f} {r['wall_time']:>9.1f}")

    print("\nPer-level constraint rejection:")
    for arm, r in results.items():
        gr = ", ".join(f"{g:.4f}" for g in r["per_level_geom_rej"])
        print(f"  {arm:<8} geom_rej = [{gr}]")

    print("\nPer-level mean proposal time (ms):")
    for arm, r in results.items():
        tt = ", ".join(f"{t*1e3:.2f}" for t in r["per_level_proposal_time_mean"])
        print(f"  {arm:<8} = [{tt}]")

    print("\nPer-level n_bounces_mean:")
    for arm, r in results.items():
        bb = ", ".join(f"{b:.3f}" for b in r["per_level_n_bounces_mean"])
        print(f"  {arm:<8} = [{bb}]")

    speedup = results["C_HNN"]["mean_proposal_time"] / results["C"]["mean_proposal_time"]
    print(f"\nArm C_HNN is {speedup:.1f}x slower per proposal than Arm C "
          f"({results['C_HNN']['mean_proposal_time']*1e3:.3f}ms vs "
          f"{results['C']['mean_proposal_time']*1e3:.3f}ms)")

    print("\nStage 1 complete — no errors.")



# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2 — full n_rep run, incremental + resumable
# ═══════════════════════════════════════════════════════════════════════════════

REPS_DIR = OUT_DIR / "reps"
SEED_BASE = 0


def _rep_path(arm: str, rep: int) -> Path:
    return REPS_DIR / f"{arm}_rep{rep:02d}.npz"


def _save_rep_atomic(path: Path, r: Dict) -> None:
    """Write-to-temp-then-rename so a crash mid-write never leaves a rep file
    that looks complete but is corrupt / half-written."""
    # np.savez auto-appends ".npz" if the given name doesn't already end in
    # it, so the temp name must end in ".npz" itself (not ".npz.tmp") to
    # avoid silently writing "<name>.npz.tmp.npz" instead.
    tmp_path = path.with_name(path.stem + ".tmp.npz")
    np.savez(
        tmp_path,
        beta_hat=r["beta_hat"], P_F=r["P_F"], levels=r["levels"],
        thresholds=np.array(r["thresholds"]),
        geom_rej=np.array(r["per_level_geom_rej"]),
        accept_rate=np.array(r["per_level_accept_rate"]),
        abs_dH_mean=np.array(r["per_level_abs_dH_mean"]),
        n_bounces_mean=np.array(r["per_level_n_bounces_mean"]),
        proposal_time_mean_per_level=np.array(r["per_level_proposal_time_mean"]),
        proposal_times=r["proposal_times"],
        mean_proposal_time=r["mean_proposal_time"],
        std_proposal_time=r["std_proposal_time"],
        n_proposals=r["n_proposals"],
    )
    # np.savez appends .npz if missing; tmp_path already ends in .npz.tmp so it
    # writes exactly that name.
    os.replace(tmp_path, path)


def _load_rep(path: Path) -> Dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def run_stage2(n_rep: int = 30) -> None:
    print("=" * 70)
    print(f"STAGE 2 — full run  n_rep={n_rep}  (incremental + resumable)")
    print("=" * 70)

    REPS_DIR.mkdir(parents=True, exist_ok=True)
    ctrl_cfg = make_ctrl_cfg()
    target = CorrelatedGaussianTarget(rho=0.0)
    lsf = ThalerLinearLSF(beta=3.5, rho=0.0)
    device = torch.device("cpu")
    model = load_hnn(device=device)
    print(f"\nbeta=3.5  rho=0  d=2  pf_ref={lsf.pf_ref:.4e}  n_rep={n_rep}")

    for arm in ("C", "C_HNN"):
        print(f"\n{'='*70}\nArm {arm}\n{'='*70}")
        for rep in range(n_rep):
            path = _rep_path(arm, rep)
            if path.exists():
                print(f"  [skip] {arm} rep{rep:02d} already done ({path.name})")
                continue
            seed = SEED_BASE + rep
            t0 = time.perf_counter()
            r = _run_one_sus_bb(
                arm, target, lsf, ctrl_cfg, seed=seed, model=model, device=device,
                rep_label=f"rep{rep+1:02d}/{n_rep}", verbose=False,
            )
            wall = time.perf_counter() - t0
            _save_rep_atomic(path, r)
            print(f"  rep{rep:02d}/{n_rep-1}  beta_hat={r['beta_hat']:.4f}  "
                  f"P_F={r['P_F']:.4e}  levels={r['levels']}  "
                  f"mean_prop_t={r['mean_proposal_time']*1e3:.3f}ms  "
                  f"n_prop={r['n_proposals']}  wall={wall:.1f}s  "
                  f"[saved {path.name}]", flush=True)

    print("\nAll reps done (or already present). Aggregating ...")
    aggregate(n_rep=n_rep)
    print("\nStage 2 complete — no errors.")


def aggregate(n_rep: int = 30) -> None:
    """
    Build the consolidated raw npz + markdown record from however many rep
    files currently exist under REPS_DIR (safe to run any time, including on
    a partially-completed run, to inspect progress).
    """
    npz_payload = {}
    summary = {}

    for arm in ("C", "C_HNN"):
        reps = []
        for rep in range(n_rep):
            path = _rep_path(arm, rep)
            if path.exists():
                reps.append(_load_rep(path))
        if not reps:
            print(f"  [aggregate] no completed reps for arm={arm} yet, skipping")
            continue

        beta_arr = np.array([r["beta_hat"] for r in reps])
        pf_arr = np.array([r["P_F"] for r in reps])
        levels_arr = np.array([r["levels"] for r in reps])
        geom_rej_raw = _pad_stack_bb([r["geom_rej"] for r in reps])
        accept_rate_raw = _pad_stack_bb([r["accept_rate"] for r in reps])
        abs_dH_raw = _pad_stack_bb([r["abs_dH_mean"] for r in reps])
        n_bounces_raw = _pad_stack_bb([r["n_bounces_mean"] for r in reps])
        all_proposal_times = np.concatenate([r["proposal_times"] for r in reps])

        npz_payload[f"{arm}_beta"] = beta_arr
        npz_payload[f"{arm}_pf"] = pf_arr
        npz_payload[f"{arm}_levels"] = levels_arr
        npz_payload[f"{arm}_geom_rej"] = geom_rej_raw
        npz_payload[f"{arm}_accept_rate"] = accept_rate_raw
        npz_payload[f"{arm}_abs_dH"] = abs_dH_raw
        npz_payload[f"{arm}_n_bounces"] = n_bounces_raw
        npz_payload[f"{arm}_proposal_times"] = all_proposal_times

        summary[arm] = dict(
            n_reps=len(reps),
            mean_beta=float(beta_arr.mean()), std_beta=float(beta_arr.std()),
            bias_beta=float(beta_arr.mean() - 3.5), cov_beta=float(beta_arr.std() / 3.5),
            mean_levels=float(levels_arr.mean()),
            geom_rej_per_level=np.nanmean(geom_rej_raw, axis=0),
            accept_rate_per_level=np.nanmean(accept_rate_raw, axis=0),
            abs_dH_per_level=np.nanmean(abs_dH_raw, axis=0),
            n_bounces_per_level=np.nanmean(n_bounces_raw, axis=0),
            mean_abs_dH=float(np.nanmean(abs_dH_raw)),
            mean_prop_time=float(all_proposal_times.mean()),
            std_prop_time=float(all_proposal_times.std()),
            n_proposals=int(all_proposal_times.size),
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUT_DIR / "bb_hnn_raw.npz"
    if npz_payload:
        np.savez(npz_path, **npz_payload)
        print(f"  Saved raw data: {npz_path}")

    _write_record_bb(summary, n_rep)


def _pad_stack_bb(list_of_lists) -> np.ndarray:
    max_len = max(len(l) for l in list_of_lists)
    out = np.full((len(list_of_lists), max_len), np.nan, dtype=np.float64)
    for i, l in enumerate(list_of_lists):
        out[i, :len(l)] = l
    return out


def _write_record_bb(summary: Dict, n_rep: int) -> None:
    lines = [
        "# Barrier Bouncing Driven by an HNN Gradient (Arm C vs Arm C_HNN)",
        "",
        "Linear LSF, beta=3.5, rho=0, d=2 (matches Chapter 6's Arm C setup "
        "exactly). Bounce/reflection and the true-G constraint checks are "
        "analytic and identical between arms; only the leapfrog force/kinetic "
        "terms differ:",
        "",
        "- Arm C: analytic gradient (existing integrate_C, unchanged)",
        "- Arm C_HNN: full HNN vector field (dq/dt=dH_theta/dp, "
        "dp/dt=-dH_theta/dq), exactly matching leapfrog_hnn / "
        "hnn_vector_field (Chapter 4/5)",
        "",
        "MH acceptance uses the true analytic Hamiltonian in both arms.",
        "",
        "## Controller settings (identical to Chapter 6)",
        "",
        "| param | value |",
        "|-------|-------|",
    ]
    ctrl_cfg = make_ctrl_cfg()
    for k in ("c_cfl", "eps_E", "tau_safety", "dt_min", "dt_max",
              "dt_max_change", "n_bounce_max", "tol_hit", "bisect_max_iter"):
        lines.append(f"| {k} | {getattr(ctrl_cfg, k)} |")
    lines += [
        f"| N (SUS_N) | {SUS_N} |  p0 (SUS_P0) | {SUS_P0} |",
        f"| n_rep (requested) | {n_rep} |",
        "",
        "## Summary",
        "",
        "| arm | n_reps | beta_hat | std | bias | COV_beta | mean_lv | "
        "mean\\|dH\\| | mean_prop_t (ms) | std_prop_t (ms) | n_proposals |",
        "|-----|--------|----------|-----|------|----------|---------|"
        "-----------|------------------|-----------------|-------------|",
    ]
    for arm, s in summary.items():
        lines.append(
            f"| {arm} | {s['n_reps']} | {s['mean_beta']:.4f} | {s['std_beta']:.4f} | "
            f"{s['bias_beta']:+.4f} | {s['cov_beta']:.4f} | {s['mean_levels']:.2f} | "
            f"{s['mean_abs_dH']:.4e} | {s['mean_prop_time']*1e3:.4f} | "
            f"{s['std_prop_time']*1e3:.4f} | {s['n_proposals']} |"
        )

    if "C" in summary and "C_HNN" in summary:
        speedup = summary["C_HNN"]["mean_prop_time"] / summary["C"]["mean_prop_time"]
        lines += ["", f"**Arm C_HNN is {speedup:.1f}x slower per proposal than Arm C** "
                       f"(over {summary['C']['n_proposals']} vs "
                       f"{summary['C_HNN']['n_proposals']} timed proposals)."]

    lines += ["", "## Per-level mechanism (mean across reps)", ""]
    for arm, s in summary.items():
        lines += [
            f"### Arm {arm}", "",
            "| lv | geom_rej | accept_rate | abs_dH_mean | n_bounces_mean |",
            "|----|----------|-------------|-------------|----------------|",
        ]
        n_lv = len(s["geom_rej_per_level"])
        for lv in range(n_lv):
            lines.append(
                f"| L{lv+1} | {s['geom_rej_per_level'][lv]:.4f} | "
                f"{s['accept_rate_per_level'][lv]:.4f} | "
                f"{s['abs_dH_per_level'][lv]:.4e} | "
                f"{s['n_bounces_per_level'][lv]:.3f} |"
            )
        lines.append("")

    record_path = OUT_DIR / "record_bb_hnn.md"
    with open(record_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved record: {record_path}")


def progress(n_rep: int = 30) -> None:
    """Lightweight status check: how many reps are done per arm, safe to run
    at any time against a running or finished Stage 2 job."""
    print("=" * 60)
    print("STAGE 2 PROGRESS")
    print("=" * 60)
    for arm in ("C", "C_HNN"):
        done = [rep for rep in range(n_rep) if _rep_path(arm, rep).exists()]
        latest_mtime = None
        if done:
            latest_path = max((_rep_path(arm, r) for r in done), key=lambda p: p.stat().st_mtime)
            latest_mtime = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(latest_path.stat().st_mtime))
        print(f"  Arm {arm:<7} {len(done):>3}/{n_rep} reps done"
              + (f"   (latest: {latest_mtime})" if latest_mtime else "   (none yet)"))
    print("=" * 60)


if __name__ == "__main__":
    if "--stage1" in sys.argv:
        stage1()
    if "--stage2" in sys.argv:
        run_stage2()
    if "--aggregate" in sys.argv:
        aggregate()
    if "--progress" in sys.argv:
        progress()
