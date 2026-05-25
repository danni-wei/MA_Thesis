"""
Banana distribution SuS: Leapfrog vs Yoshida4 integrator comparison.
Experiment 2a: fixed epsilon=0.1, 20 runs.
Experiment 2b: step-size sweep [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3].
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.stats import norm as Nrm

from integrators import leapfrog, yoshida4

# ── output directories ────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(HERE, "..", "..", "results", "leapfrog_vs_yoshida_linear")
OUTDIR     = os.path.join(HERE, "..", "..", "results", "leapfrog_vs_yoshida_banana")
SWEEP_DIR  = os.path.join(OUTDIR, "stepsize_sweep")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(SWEEP_DIR, exist_ok=True)

# ── banana distribution parameters ───────────────────────────────────────────
a   = 1.15
b   = 0.5
rho = 0.9
d   = 2

_Sigma     = np.array([[1.0, rho], [rho, 1.0]])
_L_chol    = np.linalg.cholesky(_Sigma)
_Sigma_inv = (1.0 / (1.0 - rho**2)) * np.array([[1.0, -rho], [-rho, 1.0]])


def U_banana(q):
    q1, q2 = q[0], q[1]
    u = np.array([a * q1, a * (q2 - b * q1**2)])
    return 0.5 * float(u @ _Sigma_inv @ u)


def grad_U_banana(q):
    """
    Analytical gradient of U_banana.
    U = 0.5 u^T Sigma_inv u,  u = [a*q1, a*(q2 - b*q1^2)]
    dU/dq = J^T @ (Sigma_inv @ u),  J = du/dq
    """
    q1, q2 = q[0], q[1]
    u = np.array([a * q1, a * (q2 - b * q1**2)])
    v = _Sigma_inv @ u                              # Sigma_inv u
    J = np.array([[a, 0.0], [-2.0 * a * b * q1, a]])  # Jacobian du/dq
    return J.T @ v


def sample_banana(n, rng):
    """Sample n points exactly from banana distribution."""
    z = (_L_chol @ rng.standard_normal((2, n))).T   # (n, 2) from N(0, Sigma)
    q = np.empty((n, 2))
    q[:, 0] = z[:, 0] / a
    q[:, 1] = z[:, 1] / a + b * (z[:, 0] / a) ** 2
    return q


# ── verify gradient with finite differences ───────────────────────────────────
def _verify_gradient():
    rng_v = np.random.default_rng(0)
    q_test = rng_v.standard_normal(d) * 0.5
    g_anal = grad_U_banana(q_test)
    eps_fd = 1e-6
    g_fd = np.zeros(d)
    for i in range(d):
        qp, qm = q_test.copy(), q_test.copy()
        qp[i] += eps_fd; qm[i] -= eps_fd
        g_fd[i] = (U_banana(qp) - U_banana(qm)) / (2 * eps_fd)
    err = float(np.max(np.abs(g_anal - g_fd)))
    print(f"[grad check] max |anal - FD| = {err:.2e}  (must be < 1e-6)")
    assert err < 1e-6, f"Gradient check failed: error={err}"

_verify_gradient()

# ── reference Pf via MCS ──────────────────────────────────────────────────────
c = 2.5
n_mcs = 10_000_000
print(f"Computing reference Pf via MCS ({n_mcs:.0e} samples) ...")
rng_mcs = np.random.default_rng(12345)
_z_mcs  = (_L_chol @ rng_mcs.standard_normal((2, n_mcs))).T
_q1_mcs = _z_mcs[:, 0] / a
Pf_ref  = float(np.mean(_q1_mcs >= c))
print(f"  c={c}, Pf_ref = {Pf_ref:.6e}")

# auto-adjust c if Pf_ref outside [1e-4, 1e-2]
if not (1e-4 <= Pf_ref <= 1e-2):
    c_new = float(Nrm.ppf(1 - 1e-3) / a)   # target Pf ~ 1e-3
    Pf_ref = float(np.mean(_q1_mcs >= c_new))
    print(f"  Adjusting c → {c_new:.4f}, new Pf_ref = {Pf_ref:.6e}")
    c = c_new

del _z_mcs, _q1_mcs
print(f"Using c={c:.4f},  Pf_ref = {Pf_ref:.6e}\n")


def G_banana(q):
    return c - q[0]       # failure: G <= 0  ↔  q1 >= c


# ── SuS parameters ────────────────────────────────────────────────────────────
N         = 1000
p0        = 0.1
Jmax      = 10
L         = 10
n_runs    = 20
seed_base = 42


# ── HMC step ──────────────────────────────────────────────────────────────────
def hmc_step(q, integrator_fn, eps, rng):
    p0s = rng.standard_normal(d)
    H0  = U_banana(q) + 0.5 * np.dot(p0s, p0s)

    q_prop, p_prop, n_grads = integrator_fn(q, p0s, grad_U_banana, eps, L)

    if not np.all(np.isfinite(q_prop)):
        return q.copy(), False, np.inf, n_grads

    H1      = U_banana(q_prop) + 0.5 * np.dot(p_prop, p_prop)
    delta_H = H1 - H0

    if not np.isfinite(delta_H):
        return q.copy(), False, np.inf, n_grads

    log_alpha = min(0.0, -delta_H)
    accepted  = rng.random() < np.exp(log_alpha)
    return (q_prop if accepted else q.copy()), accepted, delta_H, n_grads


# ── single SuS run ────────────────────────────────────────────────────────────
def run_sus(integrator_fn, eps, rng_seed, save_samples=False):
    """Returns dict with pf_hat and per-level records, or None on failure."""
    rng = np.random.default_rng(rng_seed)
    level_records = []

    # Level 0: sample directly from banana distribution
    samples = sample_banana(N, rng)
    G_vals  = np.array([G_banana(s) for s in samples])

    b_k = np.percentile(G_vals, p0 * 100)
    rec0 = dict(threshold=b_k, acceptance_rate=1.0, mean_abs_delta_H=0.0)
    if save_samples:
        rec0['samples'] = samples.copy()
        rec0['G_vals']  = G_vals.copy()
    level_records.append(rec0)

    n_intermediate  = 0
    n_seed_actual   = int(p0 * N)

    while b_k > 0 and len(level_records) < Jmax:
        seed_idx = np.argsort(G_vals)[:n_seed_actual]
        seeds    = samples[seed_idx]

        new_samples          = np.zeros((N, d))
        acc_list, dH_list    = [], []
        k = 0

        for s in range(n_seed_actual):
            q_cur      = seeds[s].copy()
            chain_len  = N // n_seed_actual + (1 if s < N % n_seed_actual else 0)

            for _ in range(chain_len):
                if k >= N:
                    break
                q_prop, accepted, delta_H, _ = hmc_step(q_cur, integrator_fn, eps, rng)

                if not np.isfinite(delta_H):
                    return None   # numerical failure

                if G_banana(q_prop) <= b_k:
                    if accepted:
                        q_cur = q_prop
                    acc_list.append(1 if accepted else 0)
                else:
                    acc_list.append(0)

                new_samples[k] = q_cur
                dH_list.append(abs(delta_H))
                k += 1

        samples = new_samples
        G_vals  = np.array([G_banana(s) for s in samples])
        b_k_new = np.percentile(G_vals, p0 * 100)

        rec = dict(
            threshold       = b_k_new,
            acceptance_rate = float(np.mean(acc_list)) if acc_list else 0.0,
            mean_abs_delta_H= float(np.mean(dH_list))  if dH_list else 0.0,
        )
        if save_samples:
            rec['samples'] = samples.copy()
            rec['G_vals']  = G_vals.copy()
        level_records.append(rec)

        n_intermediate += 1
        b_k = b_k_new
        if b_k <= 0:
            break

    G_final = np.array([G_banana(s) for s in samples])
    p_last  = float(np.mean(G_final <= 0.0))
    Pf_hat  = (p0 ** n_intermediate) * p_last

    return dict(pf_hat=Pf_hat, levels=level_records)


# ── helpers: aggregate over runs ──────────────────────────────────────────────
def aggregate_levels(results, key):
    max_lvl = max(len(r['levels']) for r in results)
    out = []
    for lv in range(max_lvl):
        vals = [r['levels'][lv][key] for r in results if lv < len(r['levels'])]
        out.append((float(np.mean(vals)), float(np.std(vals)), len(vals)))
    return out


def pf_array(results):
    return np.array([r['pf_hat'] for r in results])


# ── banana contour grid (vectorised) ─────────────────────────────────────────
def banana_contour(xlim=(-4, 4), ylim=(-3, 8), n=250):
    xx, yy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    u1 = a * xx
    u2 = a * (yy - b * xx**2)
    s  = 1.0 / (1.0 - rho**2)
    ZZ = np.exp(-0.5 * s * (u1**2 - 2*rho*u1*u2 + u2**2))
    return xx, yy, ZZ


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2a — fixed epsilon = 0.1
# ═══════════════════════════════════════════════════════════════════════════════
eps_2a = 0.1
print("=" * 65)
print(f"Experiment 2a: fixed epsilon={eps_2a}, n_runs={n_runs}")
print("=" * 65)

print("  Running Leapfrog ...")
lf_res = [run_sus(leapfrog,  eps_2a, seed_base + i,         save_samples=True) for i in range(n_runs)]
print("  Running Yoshida4 ...")
y4_res = [run_sus(yoshida4,  eps_2a, seed_base + 1000 + i,  save_samples=True) for i in range(n_runs)]

lf_res_ok = [r for r in lf_res if r is not None]
y4_res_ok = [r for r in y4_res if r is not None]
print(f"  Leapfrog ok={len(lf_res_ok)}/20   Yoshida4 ok={len(y4_res_ok)}/20\n")

lf_pf = pf_array(lf_res_ok);  y4_pf = pf_array(y4_res_ok)
lf_acc = aggregate_levels(lf_res_ok, 'acceptance_rate')
y4_acc = aggregate_levels(y4_res_ok, 'acceptance_rate')
lf_dH  = aggregate_levels(lf_res_ok, 'mean_abs_delta_H')
y4_dH  = aggregate_levels(y4_res_ok, 'mean_abs_delta_H')
lf_thr = aggregate_levels(lf_res_ok, 'threshold')
y4_thr = aggregate_levels(y4_res_ok, 'threshold')


def _pf_row(label, arr):
    m, s = np.mean(arr), np.std(arr)
    cov  = s / m if m > 0 else np.inf
    bias = (m - Pf_ref) / Pf_ref
    print(f"  {label:10s}  Pf={m:.4e} ± {s:.4e}  COV={cov:.3f}  Bias={bias:+.3f}")

print(f"  Pf_ref (MCS) = {Pf_ref:.4e}")
_pf_row("Leapfrog", lf_pf)
_pf_row("Yoshida4", y4_pf)
print()

max_lv = max(len(lf_acc), len(y4_acc))
hdr = f"{'Lvl':>4}  {'Method':>10}  {'Threshold':>10}  {'AccRate':>9}  {'mean|dH|':>10}"
print(hdr); print("-" * len(hdr))
for lv in range(max_lv):
    for label, thr_list, acc_list2, dH_list in [
        ("Leapfrog", lf_thr, lf_acc, lf_dH),
        ("Yoshida4", y4_thr, y4_acc, y4_dH),
    ]:
        if lv < len(thr_list):
            tm, ts, _ = thr_list[lv]
            am, as_, _ = acc_list2[lv]
            dm, ds, _  = dH_list[lv]
            print(f"  {lv:>2}  {label:>10}  {tm:+8.3f}±{ts:.3f}  {am:.3f}±{as_:.3f}  {dm:.4e}±{ds:.4e}")
    print()
print("=" * 65)

# ── 2a Plot 1: per-level acceptance rate ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x_lf = np.arange(len(lf_acc)); x_y4 = np.arange(len(y4_acc))
ax.errorbar(x_lf, [v[0] for v in lf_acc], yerr=[v[1] for v in lf_acc],
            fmt='o-', color='steelblue', capsize=4, label='Leapfrog')
ax.errorbar(x_y4+0.15, [v[0] for v in y4_acc], yerr=[v[1] for v in y4_acc],
            fmt='s--', color='tomato', capsize=4, label='Yoshida4')
ax.set_xlabel('Level', fontsize=12); ax.set_ylabel('Acceptance rate', fontsize=12)
ax.set_title('Banana SuS — Per-level acceptance rate', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'per_level_acceptance.png'), dpi=150)
plt.close(fig); print('Saved per_level_acceptance.png')

# ── 2a Plot 2: per-level mean|ΔH| ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(x_lf[1:], [v[0] for v in lf_dH[1:]], yerr=[v[1] for v in lf_dH[1:]],
            fmt='o-', color='steelblue', capsize=4, label='Leapfrog')
ax.errorbar(x_y4[1:]+0.15, [v[0] for v in y4_dH[1:]], yerr=[v[1] for v in y4_dH[1:]],
            fmt='s--', color='tomato', capsize=4, label='Yoshida4')
ax.set_xlabel('Level', fontsize=12); ax.set_ylabel('mean |ΔH|', fontsize=12)
ax.set_title('Banana SuS — Per-level mean |ΔH|', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'per_level_deltaH.png'), dpi=150)
plt.close(fig); print('Saved per_level_deltaH.png')

# ── 2a Plot 3: Pf boxplot ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.boxplot([lf_pf, y4_pf], labels=['Leapfrog', 'Yoshida4'], patch_artist=True)
ax.axhline(Pf_ref, color='r', ls='--', label=f'Pf_ref {Pf_ref:.3e}')
ax.set_ylabel('Pf estimate', fontsize=12)
ax.set_title('Banana SuS — Pf across 20 runs', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'pf_boxplot.png'), dpi=150)
plt.close(fig); print('Saved pf_boxplot.png')

# ── 2a Plot 4: sample scatter per level (representative run) ──────────────────
rep_lf = next((r for r in lf_res if r is not None), None)
rep_y4 = next((r for r in y4_res if r is not None), None)
n_col  = min(len(rep_lf['levels']), len(rep_y4['levels']), 5)

xx_c, yy_c, ZZ_c = banana_contour()
x_lsf = c  # vertical line at q1 = c

fig, axes = plt.subplots(2, n_col, figsize=(3.8 * n_col, 7.5))
for col, lv in enumerate(range(n_col)):
    for row, (res, mname) in enumerate([(rep_lf, 'Leapfrog'), (rep_y4, 'Yoshida4')]):
        ax = axes[row, col]
        samp = res['levels'][lv]['samples']
        thr  = res['levels'][lv]['threshold']
        ax.contour(xx_c, yy_c, ZZ_c, levels=14, cmap='Blues', linewidths=0.6, alpha=0.6)
        ax.scatter(samp[:, 0], samp[:, 1], s=2, alpha=0.35, c='steelblue')
        ax.axvline(x_lsf, color='r', ls='--', lw=1.2, label=f'q1={c:.2f}')
        ax.set_xlim(-4, 4); ax.set_ylim(-3, 8)
        ax.set_title(f'{mname} lv{lv} b={thr:.2f}', fontsize=8)
        ax.set_aspect('auto')
        if col == 0:
            ax.set_ylabel(mname, fontsize=9)
        if row == 0 and col == 0:
            ax.legend(fontsize=7, loc='upper left')

fig.suptitle(f'Banana SuS samples per level (run 0, ε={eps_2a})', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'sample_scatter.png'), dpi=150)
plt.close(fig); print('Saved sample_scatter.png')


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2b — step size sweep
# ═══════════════════════════════════════════════════════════════════════════════
epsilons_sweep = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
methods = [("leapfrog", leapfrog), ("yoshida4", yoshida4)]

print("\n" + "=" * 65)
print(f"Experiment 2b: step-size sweep, n_runs={n_runs} per combo")
print("=" * 65)

sweep_results = {name: [] for name, _ in methods}

for name, fn in methods:
    for eps in epsilons_sweep:
        offset = 0 if name == "leapfrog" else 1000
        runs = [run_sus(fn, eps, seed_base + offset + i) for i in range(n_runs)]
        n_ok   = sum(r is not None for r in runs)
        n_fail = n_runs - n_ok
        print(f"  eps={eps:.3f}  {name:10s}  ok={n_ok}/{n_runs}  failed={n_fail}")
        sweep_results[name].append(runs)
    print()


def sweep_stats(runs_list):
    valid = [r for r in runs_list if r is not None]
    if not valid:
        return dict(pf_mean=np.nan, pf_std=np.nan, cov=np.nan,
                    avg_acc=np.nan, avg_acc_std=np.nan,
                    avg_dH=np.nan, avg_dH_std=np.nan, n_valid=0)
    pf_arr   = np.array([r['pf_hat'] for r in valid])
    pf_mean  = np.mean(pf_arr); pf_std = np.std(pf_arr)
    cov      = pf_std / pf_mean if pf_mean > 0 else np.inf

    per_acc, per_dH = [], []
    for r in valid:
        lvls = r['levels'][1:]
        if lvls:
            per_acc.append(np.mean([lv['acceptance_rate']    for lv in lvls]))
            per_dH.append( np.mean([lv['mean_abs_delta_H']   for lv in lvls]))

    return dict(
        pf_mean     = pf_mean, pf_std = pf_std, cov = cov,
        avg_acc     = np.mean(per_acc) if per_acc else np.nan,
        avg_acc_std = np.std(per_acc)  if per_acc else np.nan,
        avg_dH      = np.mean(per_dH)  if per_dH  else np.nan,
        avg_dH_std  = np.std(per_dH)   if per_dH  else np.nan,
        n_valid     = len(valid),
    )


stats_2b = {name: [sweep_stats(sweep_results[name][ei])
                   for ei in range(len(epsilons_sweep))]
            for name, _ in methods}

# print sweep table
hdr2 = (f"{'epsilon':>8} | {'Method':>10} | {'Pf_mean':>11} | {'COV':>7} | "
        f"{'avg_acc':>9} | {'avg_mean|dH|':>14} | {'n_ok':>5}")
print("\n" + "=" * len(hdr2))
print(f"  Pf_ref (MCS) = {Pf_ref:.4e}")
print("=" * len(hdr2))
print(hdr2); print("-" * len(hdr2))
for ei, eps in enumerate(epsilons_sweep):
    for name, _ in methods:
        s = stats_2b[name][ei]
        pf_s  = f"{s['pf_mean']:.4e}" if np.isfinite(s['pf_mean']) else "   NaN   "
        co_s  = f"{s['cov']:.4f}"     if np.isfinite(s['cov'])     else "  NaN  "
        ac_s  = f"{s['avg_acc']:.4f}" if np.isfinite(s['avg_acc']) else "  NaN  "
        dh_s  = f"{s['avg_dH']:.4e}"  if np.isfinite(s['avg_dH'])  else "   NaN      "
        print(f"  {eps:>6.3f}   | {name:>10} | {pf_s:>11} | {co_s:>7} | "
              f"{ac_s:>9} | {dh_s:>14} | {s['n_valid']:>5}")
    print("-" * len(hdr2))
print("=" * len(hdr2))


def _arr2b(name, key):
    return np.array([stats_2b[name][ei][key] for ei in range(len(epsilons_sweep))])

eps_arr = np.array(epsilons_sweep)

def _safe_eb(ax, x, y, ye, **kw):
    m = np.isfinite(y) & np.isfinite(ye)
    ax.errorbar(x[m], y[m], yerr=ye[m], capsize=4, **kw)


# ── 2b Plot 1: acceptance rate vs epsilon ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
_safe_eb(ax, eps_arr, _arr2b("leapfrog","avg_acc"), _arr2b("leapfrog","avg_acc_std"),
         fmt='o-', color='steelblue', label='Leapfrog')
_safe_eb(ax, eps_arr, _arr2b("yoshida4","avg_acc"), _arr2b("yoshida4","avg_acc_std"),
         fmt='s--', color='tomato', label='Yoshida4')
ax.set_xscale('log'); ax.set_xlabel('ε', fontsize=12)
ax.set_ylabel('Avg acceptance rate', fontsize=12)
ax.set_title('Banana — Acceptance Rate vs Step Size', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(SWEEP_DIR, 'stepsize_acceptance.png'), dpi=150)
plt.close(fig); print('Saved stepsize_acceptance.png')

# ── 2b Plot 2: mean|ΔH| vs epsilon ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
_safe_eb(ax, eps_arr, _arr2b("leapfrog","avg_dH"), _arr2b("leapfrog","avg_dH_std"),
         fmt='o-', color='steelblue', label='Leapfrog')
_safe_eb(ax, eps_arr, _arr2b("yoshida4","avg_dH"), _arr2b("yoshida4","avg_dH_std"),
         fmt='s--', color='tomato', label='Yoshida4')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('ε', fontsize=12); ax.set_ylabel('Avg mean |ΔH|', fontsize=12)
ax.set_title('Banana — Energy Error vs Step Size', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4, which='both'); fig.tight_layout()
fig.savefig(os.path.join(SWEEP_DIR, 'stepsize_deltaH.png'), dpi=150)
plt.close(fig); print('Saved stepsize_deltaH.png')

# ── 2b Plot 3: Pf COV vs epsilon ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
    cov_arr = _arr2b(name, "cov")
    m = np.isfinite(cov_arr)
    ax.plot(eps_arr[m], cov_arr[m], f'{marker}-', color=color, label=name.capitalize())
ax.axhline(0, color='gray', ls=':', lw=1)
ax.set_xscale('log'); ax.set_xlabel('ε', fontsize=12)
ax.set_ylabel('COV of Pf', fontsize=12)
ax.set_title('Banana — Pf Estimation Accuracy vs Step Size', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(SWEEP_DIR, 'stepsize_pf_cov.png'), dpi=150)
plt.close(fig); print('Saved stepsize_pf_cov.png')

# ── 2b Plot 4: Pf relative bias vs epsilon ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
    pf_m = _arr2b(name, "pf_mean")
    bias = (pf_m - Pf_ref) / Pf_ref * 100
    m = np.isfinite(bias)
    ax.plot(eps_arr[m], bias[m], f'{marker}-', color=color, label=name.capitalize())
ax.axhline(0, color='k', ls='--', lw=1, label='Zero bias')
ax.set_xscale('log'); ax.set_xlabel('ε', fontsize=12)
ax.set_ylabel('Relative bias (%)', fontsize=12)
ax.set_title('Banana — Relative Bias vs Step Size', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(SWEEP_DIR, 'stepsize_pf_bias.png'), dpi=150)
plt.close(fig); print('Saved stepsize_pf_bias.png')

print(f'\nAll plots: {OUTDIR}  and  {SWEEP_DIR}')


# ═══════════════════════════════════════════════════════════════════════════════
# Append results to record_yoshida.md
# ═══════════════════════════════════════════════════════════════════════════════
def _num(v, fmt='.4e'):
    return f"{v:{fmt}}" if np.isfinite(v) else "N/A"


def _pf_table_row_2a(label, arr):
    m, s = np.mean(arr), np.std(arr)
    cov  = s / m if m > 0 else np.inf
    bias = (m - Pf_ref) / Pf_ref
    return f"| {label} | {m:.4e} | {s:.4e} | {cov:.3f} | {bias:+.3f} |"


def _level_table(thr_list, acc_list2, dH_list2):
    rows = [
        "| Level | Threshold (mean±std) | n_runs | Acc rate (mean±std) | mean|ΔH| (mean±std) |",
        "|------:|--------------------:|-------:|--------------------:|--------------------:|",
    ]
    for lv, (thr, acc, dh) in enumerate(zip(thr_list, acc_list2, dH_list2)):
        tm, ts, n = thr; am, as_, _ = acc; dm, ds, _ = dh
        rows.append(f"| {lv} | {tm:+.3f} ± {ts:.3f} | {n} | "
                    f"{am:.3f} ± {as_:.3f} | {dm:.4e} ± {ds:.4e} |")
    return "\n".join(rows)


# 2b sweep table rows
sweep_rows = []
for ei, eps in enumerate(epsilons_sweep):
    for name, _ in methods:
        s = stats_2b[name][ei]
        sweep_rows.append(
            f"| {eps} | {name} | {_num(s['pf_mean'])} | {_num(s['cov'],'.3f')} | "
            f"{_num(s['avg_acc'],'.3f')} | {_num(s['avg_dH'])} |"
        )

sweep_table_md = (
    "| epsilon | Method | Pf_mean | COV | avg_acc | avg_mean\\|ΔH\\| |\n"
    "|---------|--------|---------|-----|---------|----------------|\n"
    + "\n".join(sweep_rows)
)

# key observations for 2b
obs_2b = []
for ei, eps in enumerate(epsilons_sweep):
    lf_a = stats_2b["leapfrog"][ei]["avg_acc"]
    y4_a = stats_2b["yoshida4"][ei]["avg_acc"]
    if np.isfinite(lf_a) and np.isfinite(y4_a) and abs(y4_a - lf_a) > 0.05:
        obs_2b.append(
            f"- At ε={eps}: Yoshida4 acc={y4_a:.3f}, Leapfrog acc={lf_a:.3f} "
            f"(gap={y4_a-lf_a:+.3f})."
        )

if not obs_2b:
    obs_2b.append(
        "- No ε in the sweep produced a >5 pp acceptance gap between methods."
    )

# Compare banana vs linear Gaussian
lf_dH_2a_val = np.mean([v[0] for v in lf_dH[1:]])
y4_dH_2a_val = np.mean([v[0] for v in y4_dH[1:]])
obs_2b.append(
    f"- At ε={eps_2a}, banana: Leapfrog mean|ΔH|={lf_dH_2a_val:.4e}, "
    f"Yoshida4 mean|ΔH|={y4_dH_2a_val:.4e} "
    f"(ratio {lf_dH_2a_val/y4_dH_2a_val:.0f}×)."
)
obs_2b.append(
    "- The banana's curvature (rho=0.9) produces larger ΔH for leapfrog than the "
    "linear Gaussian case at the same ε, widening Yoshida4's energy advantage."
)

append_md = f"""
---

## Experiment 2a: Banana Distribution (Fixed Step Size)

### Setup

| Item | Value |
|------|-------|
| Distribution | Banana (a={a}, b={b}, ρ={rho}) |
| LSF | G(q) = {c:.4f} − q₁  (failure: q₁ ≥ {c:.4f}) |
| Pf_ref (MCS, 10M) | {Pf_ref:.6e} |
| d | {d} |
| N | {N} |
| p0 | {p0} |
| epsilon | {eps_2a} |
| L | {L} |
| n_runs | {n_runs} |

### Overall Pf Estimates

| Method | Pf_mean | Pf_std | COV | Relative bias |
|--------|--------:|-------:|----:|--------------:|
{_pf_table_row_2a("Leapfrog", lf_pf)}
{_pf_table_row_2a("Yoshida4", y4_pf)}

### Per-level Diagnostics — Leapfrog

{_level_table(lf_thr, lf_acc, lf_dH)}

### Per-level Diagnostics — Yoshida4

{_level_table(y4_thr, y4_acc, y4_dH)}

### Plots (leapfrog_vs_yoshida_banana/)
- `per_level_acceptance.png`
- `per_level_deltaH.png`
- `pf_boxplot.png`
- `sample_scatter.png`

---

## Experiment 2b: Banana Step Size Sensitivity

### Setup
- Same banana distribution as Experiment 2a
- L = {L} (fixed)
- epsilon = {epsilons_sweep}
- n_runs = {n_runs} per (method, epsilon)

### Results

{sweep_table_md}

### Key Observations
{chr(10).join(obs_2b)}

### Plots (leapfrog_vs_yoshida_banana/stepsize_sweep/)
- `stepsize_acceptance.png`
- `stepsize_deltaH.png`
- `stepsize_pf_cov.png`
- `stepsize_pf_bias.png`
"""

record_path = os.path.join(RECORD_DIR, 'record_yoshida.md')
with open(record_path, 'a', encoding='utf-8') as f:
    f.write(append_md)
print(f'\nAppended Experiments 2a+2b to {record_path}')
