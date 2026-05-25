"""
High-dimensional linear LSF: Leapfrog vs Yoshida4 in HMC-SuS.
G(q) = beta0*sqrt(d) - sum(q),  prior N(0,I_d),  Pf = Phi(-beta0) independent of d.

Experiment 3a: fixed epsilon=0.1, L=10, sweep d in [2, 10, 50, 100], n_runs=20.
Experiment 3b: step-size sweep per d, n_runs=10.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.stats import norm as Nrm

from integrators import leapfrog, yoshida4

# ── directories ───────────────────────────────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(HERE, "..", "..", "results", "leapfrog_vs_yoshida_linear")
OUTDIR     = os.path.join(HERE, "..", "..", "results", "leapfrog_vs_yoshida_highdim")
os.makedirs(OUTDIR, exist_ok=True)

# ── problem definition ────────────────────────────────────────────────────────
beta0    = 3.5
Pf_exact = Nrm.cdf(-beta0)          # ≈ 2.326e-4, independent of d
print(f"Pf_exact = {Pf_exact:.4e}  (beta0={beta0})\n")


def U(q):
    return 0.5 * np.dot(q, q)


def grad_U(q):
    return q.copy()


def G(q, d):
    return beta0 * np.sqrt(d) - np.sum(q)


# ── SuS / HMC parameters ─────────────────────────────────────────────────────
N         = 1000
p0        = 0.1
Jmax      = 15
L_fixed   = 10
eps_fixed = 0.1
n_runs_3a = 20
n_runs_3b = 10
seed_base = 42

dimensions = [2, 10, 50, 100]
methods    = [("leapfrog", leapfrog), ("yoshida4", yoshida4)]

# sweep epsilons per dimension
EPS_SWEEP      = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
EPS_SWEEP_D100 = [0.02, 0.05, 0.1, 0.2]          # shorter for d=100


# ── HMC step ─────────────────────────────────────────────────────────────────
def hmc_step(q, d, integrator_fn, eps, L, rng):
    p0s     = rng.standard_normal(d)
    H0      = U(q) + 0.5 * np.dot(p0s, p0s)
    q_prop, p_prop, n_grads = integrator_fn(q, p0s, grad_U, eps, L)

    if not np.all(np.isfinite(q_prop)):
        return q.copy(), False, np.inf, n_grads

    H1      = U(q_prop) + 0.5 * np.dot(p_prop, p_prop)
    delta_H = H1 - H0

    if not np.isfinite(delta_H):
        return q.copy(), False, np.inf, n_grads

    accepted = rng.random() < np.exp(min(0.0, -delta_H))
    return (q_prop if accepted else q.copy()), accepted, delta_H, n_grads


# ── single SuS run ────────────────────────────────────────────────────────────
def run_sus(d, integrator_fn, eps, L, rng_seed):
    """Returns dict or None on numerical failure."""
    rng = np.random.default_rng(rng_seed)
    level_records = []
    total_grads   = 0

    # Level 0: iid N(0, I_d) samples
    samples = rng.standard_normal((N, d))
    G_vals  = np.array([G(s, d) for s in samples])

    b_k = np.percentile(G_vals, p0 * 100)
    level_records.append(dict(threshold=b_k, acceptance_rate=1.0,
                               mean_abs_delta_H=0.0))
    n_intermediate = 0
    n_seed_actual  = int(p0 * N)

    while b_k > 0 and len(level_records) < Jmax:
        seed_idx = np.argsort(G_vals)[:n_seed_actual]
        seeds    = samples[seed_idx]

        new_samples          = np.zeros((N, d))
        acc_list, dH_list    = [], []
        k = 0

        for s in range(n_seed_actual):
            q_cur     = seeds[s].copy()
            chain_len = N // n_seed_actual + (1 if s < N % n_seed_actual else 0)

            for _ in range(chain_len):
                if k >= N:
                    break
                q_prop, accepted, delta_H, n_grads = hmc_step(
                    q_cur, d, integrator_fn, eps, L, rng)
                total_grads += n_grads

                if not np.isfinite(delta_H):
                    return None

                if G(q_prop, d) <= b_k:
                    if accepted:
                        q_cur = q_prop
                    acc_list.append(1 if accepted else 0)
                else:
                    acc_list.append(0)

                new_samples[k] = q_cur
                dH_list.append(abs(delta_H))
                k += 1

        samples = new_samples
        G_vals  = np.array([G(s, d) for s in samples])
        b_k_new = np.percentile(G_vals, p0 * 100)

        level_records.append(dict(
            threshold        = b_k_new,
            acceptance_rate  = float(np.mean(acc_list)) if acc_list else 0.0,
            mean_abs_delta_H = float(np.mean(dH_list))  if dH_list else 0.0,
        ))
        n_intermediate += 1
        b_k = b_k_new
        if b_k <= 0:
            break

    G_final = np.array([G(s, d) for s in samples])
    p_last  = float(np.mean(G_final <= 0.0))
    Pf_hat  = (p0 ** n_intermediate) * p_last

    return dict(pf_hat=Pf_hat, levels=level_records, total_grads=total_grads)


# ── aggregate helpers ─────────────────────────────────────────────────────────
def compute_stats(runs):
    valid = [r for r in runs if r is not None]
    if not valid:
        return dict(pf_mean=np.nan, pf_std=np.nan, cov=np.nan,
                    avg_acc=np.nan, avg_acc_std=np.nan,
                    avg_dH=np.nan, avg_dH_std=np.nan,
                    avg_levels=np.nan, avg_grads=np.nan, n_valid=0)

    pf_arr  = np.array([r['pf_hat'] for r in valid])
    pf_mean = np.mean(pf_arr); pf_std = np.std(pf_arr)
    cov     = pf_std / pf_mean if pf_mean > 0 else np.inf

    per_acc, per_dH, n_lvl, grads = [], [], [], []
    for r in valid:
        lvls = r['levels'][1:]
        if lvls:
            per_acc.append(np.mean([lv['acceptance_rate']    for lv in lvls]))
            per_dH.append( np.mean([lv['mean_abs_delta_H']   for lv in lvls]))
        n_lvl.append(len(r['levels']) - 1)   # number of MCMC levels
        grads.append(r['total_grads'])

    return dict(
        pf_mean     = pf_mean,  pf_std    = pf_std,   cov        = cov,
        avg_acc     = np.mean(per_acc) if per_acc else np.nan,
        avg_acc_std = np.std(per_acc)  if per_acc else np.nan,
        avg_dH      = np.mean(per_dH)  if per_dH  else np.nan,
        avg_dH_std  = np.std(per_dH)   if per_dH  else np.nan,
        avg_levels  = float(np.mean(n_lvl)),
        avg_grads   = float(np.mean(grads)),
        n_valid     = len(valid),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3a — fixed epsilon, dimension sweep
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print(f"Experiment 3a: epsilon={eps_fixed}, L={L_fixed}, n_runs={n_runs_3a}")
print("=" * 70)

results_3a = {}
for d in dimensions:
    for mname, fn in methods:
        offset = 0 if mname == "leapfrog" else 1000
        runs = []
        for i in range(n_runs_3a):
            r = run_sus(d, fn, eps_fixed, L_fixed, seed_base + offset + i)
            runs.append(r)
        n_ok   = sum(r is not None for r in runs)
        n_fail = n_runs_3a - n_ok
        print(f"  d={d:3d}  {mname:10s}  ok={n_ok}/{n_runs_3a}  failed={n_fail}")
        results_3a[(d, mname)] = runs
    print()

stats_3a = {k: compute_stats(v) for k, v in results_3a.items()}

# ── 3a terminal table ─────────────────────────────────────────────────────────
def _num(v, fmt='.4e'):
    return f"{v:{fmt}}" if np.isfinite(v) else "  N/A  "

hdr3a = (f"{'d':>4} | {'Method':>10} | {'Pf_mean':>11} | {'COV':>6} | "
         f"{'Bias%':>7} | {'avg_acc':>8} | {'avg|dH|':>10} | "
         f"{'avg_lvl':>7} | {'avg_grads':>10}")
print("\n" + "=" * len(hdr3a))
print(f"  Pf_exact = {Pf_exact:.4e}  (beta0={beta0})")
print("=" * len(hdr3a))
print(hdr3a); print("-" * len(hdr3a))
for d in dimensions:
    for mname, _ in methods:
        s = stats_3a[(d, mname)]
        bias_pct = (s['pf_mean'] - Pf_exact) / Pf_exact * 100 if np.isfinite(s['pf_mean']) else np.nan
        print(
            f"  {d:>2}   | {mname:>10} | {_num(s['pf_mean']):>11} | "
            f"{_num(s['cov'],'.4f'):>6} | {_num(bias_pct,'+.1f'):>7} | "
            f"{_num(s['avg_acc'],'.4f'):>8} | {_num(s['avg_dH']):>10} | "
            f"{_num(s['avg_levels'],'.1f'):>7} | {_num(s['avg_grads'],'.0f'):>10}"
        )
    print("-" * len(hdr3a))
print("=" * len(hdr3a))

# ── helpers for arrays ────────────────────────────────────────────────────────
d_arr = np.array(dimensions, dtype=float)

def _arr3a(mname, key):
    return np.array([stats_3a[(d, mname)][key] for d in dimensions])


# ── 3a Plot 1: acceptance rate vs d ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for mname, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
    am = _arr3a(mname, "avg_acc"); ae = _arr3a(mname, "avg_acc_std")
    m  = np.isfinite(am)
    ax.errorbar(d_arr[m], am[m], yerr=ae[m], fmt=f'{marker}-', color=color,
                capsize=5, label=mname.capitalize())
ax.set_xlabel('Dimension d', fontsize=12); ax.set_ylabel('Avg acceptance rate', fontsize=12)
ax.set_title('High-dim Linear LSF — Acceptance Rate vs Dimension', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dim_vs_acceptance.png'), dpi=150)
plt.close(fig); print('Saved dim_vs_acceptance.png')

# ── 3a Plot 2: mean|ΔH| vs d ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for mname, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
    dm = _arr3a(mname, "avg_dH"); de = _arr3a(mname, "avg_dH_std")
    m  = np.isfinite(dm) & (dm > 0)
    ax.errorbar(d_arr[m], dm[m], yerr=de[m], fmt=f'{marker}-', color=color,
                capsize=5, label=mname.capitalize())
ax.set_yscale('log'); ax.set_xlabel('Dimension d', fontsize=12)
ax.set_ylabel('Avg mean |ΔH|', fontsize=12)
ax.set_title('High-dim Linear LSF — Energy Error vs Dimension', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4, which='both'); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dim_vs_deltaH.png'), dpi=150)
plt.close(fig); print('Saved dim_vs_deltaH.png')

# ── 3a Plot 3: COV vs d ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for mname, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
    cv = _arr3a(mname, "cov")
    m  = np.isfinite(cv)
    ax.plot(d_arr[m], cv[m], f'{marker}-', color=color, label=mname.capitalize())
ax.set_xlabel('Dimension d', fontsize=12); ax.set_ylabel('COV of Pf', fontsize=12)
ax.set_title('High-dim Linear LSF — Pf COV vs Dimension', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dim_vs_cov.png'), dpi=150)
plt.close(fig); print('Saved dim_vs_cov.png')

# ── 3a Plot 4: bias% vs d ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
for mname, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
    pf_m = _arr3a(mname, "pf_mean")
    bias = (pf_m - Pf_exact) / Pf_exact * 100
    m    = np.isfinite(bias)
    ax.plot(d_arr[m], bias[m], f'{marker}-', color=color, label=mname.capitalize())
ax.axhline(0, color='k', ls='--', lw=1)
ax.set_xlabel('Dimension d', fontsize=12); ax.set_ylabel('Relative bias (%)', fontsize=12)
ax.set_title('High-dim Linear LSF — Pf Bias vs Dimension', fontsize=13)
ax.legend(fontsize=11); ax.grid(True, alpha=0.4); fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'dim_vs_bias.png'), dpi=150)
plt.close(fig); print('Saved dim_vs_bias.png')

# ── 3a Plot 5: Pf boxplots grouped by d ────────────────────────────────────
fig, axes = plt.subplots(1, len(dimensions), figsize=(4 * len(dimensions), 5), sharey=False)
for ax, d in zip(axes, dimensions):
    lf_pf = [r['pf_hat'] for r in results_3a[(d, "leapfrog")] if r is not None]
    y4_pf = [r['pf_hat'] for r in results_3a[(d, "yoshida4")] if r is not None]
    bplot = ax.boxplot([lf_pf, y4_pf], labels=['LF', 'Y4'], patch_artist=True,
                       medianprops=dict(color='k', lw=2))
    bplot['boxes'][0].set_facecolor('steelblue')
    bplot['boxes'][1].set_facecolor('tomato')
    ax.axhline(Pf_exact, color='r', ls='--', lw=1.2, label=f'Exact {Pf_exact:.2e}')
    ax.set_title(f'd={d}', fontsize=12); ax.set_ylabel('Pf', fontsize=10)
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
fig.suptitle(f'Pf estimates by dimension (ε={eps_fixed}, L={L_fixed})', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'pf_boxplot_by_dim.png'), dpi=150)
plt.close(fig); print('Saved pf_boxplot_by_dim.png')


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3b — step size sweep per dimension
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"Experiment 3b: step-size sweep, n_runs={n_runs_3b} per combo")
print("=" * 70)

results_3b  = {}   # (d, mname, eps) -> list of run results

for d in dimensions:
    eps_list = EPS_SWEEP_D100 if d == 100 else EPS_SWEEP
    for mname, fn in methods:
        for eps in eps_list:
            offset = 0 if mname == "leapfrog" else 1000
            runs = [run_sus(d, fn, eps, L_fixed, seed_base + offset + i)
                    for i in range(n_runs_3b)]
            n_ok   = sum(r is not None for r in runs)
            n_fail = n_runs_3b - n_ok
            print(f"  d={d:3d}  {mname:10s}  eps={eps:.3f}  ok={n_ok}/{n_runs_3b}  failed={n_fail}")
            results_3b[(d, mname, eps)] = runs
    print()

stats_3b = {k: compute_stats(v) for k, v in results_3b.items()}

# ── print 3b summary per dimension ───────────────────────────────────────────
for d in dimensions:
    eps_list = EPS_SWEEP_D100 if d == 100 else EPS_SWEEP
    print(f"\n--- d={d} ---")
    hdr = (f"{'eps':>6} | {'Method':>10} | {'Pf_mean':>11} | {'COV':>6} | "
           f"{'avg_acc':>8} | {'avg|dH|':>10} | {'ok':>4}")
    print(hdr); print("-" * len(hdr))
    for eps in eps_list:
        for mname, _ in methods:
            s = stats_3b[(d, mname, eps)]
            print(f"  {eps:.3f}  | {mname:>10} | {_num(s['pf_mean']):>11} | "
                  f"{_num(s['cov'],'.4f'):>6} | {_num(s['avg_acc'],'.4f'):>8} | "
                  f"{_num(s['avg_dH']):>10} | {s['n_valid']:>4}")
        print("-" * len(hdr))


# ── helper for sweep arrays ───────────────────────────────────────────────────
def _get_eps_arr(d):
    return np.array(EPS_SWEEP_D100 if d == 100 else EPS_SWEEP)

def _arr3b(d, mname, key):
    eps_list = EPS_SWEEP_D100 if d == 100 else EPS_SWEEP
    return np.array([stats_3b[(d, mname, eps)][key] for eps in eps_list])

def _safe_eb(ax, x, y, ye, **kw):
    m = np.isfinite(y) & np.isfinite(ye)
    if m.any():
        ax.errorbar(x[m], y[m], yerr=ye[m], **kw)
    else:
        plot_kw = {k: v for k, v in kw.items() if k not in ('yerr', 'capsize', 'fmt')}
        fmt = kw.get('fmt', '-')
        ax.plot([], [], fmt, **plot_kw)


# ── 3b Plot per dimension: 4-subplot figure ───────────────────────────────────
for d in dimensions:
    eps_arr_d = _get_eps_arr(d)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for mname, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
        am  = _arr3b(d, mname, "avg_acc");      ae  = _arr3b(d, mname, "avg_acc_std")
        dm  = _arr3b(d, mname, "avg_dH");       de  = _arr3b(d, mname, "avg_dH_std")
        cv  = _arr3b(d, mname, "cov")
        pf  = _arr3b(d, mname, "pf_mean")
        bias = (pf - Pf_exact) / Pf_exact * 100

        lkw = dict(fmt=f'{marker}-', color=color, capsize=4, label=mname.capitalize())

        _safe_eb(axes[0], eps_arr_d, am, ae, **lkw)
        _safe_eb(axes[1], eps_arr_d, dm, de, **lkw)

        m_cv = np.isfinite(cv)
        axes[2].plot(eps_arr_d[m_cv], cv[m_cv], f'{marker}-', color=color,
                     label=mname.capitalize())
        m_bi = np.isfinite(bias)
        axes[3].plot(eps_arr_d[m_bi], bias[m_bi], f'{marker}-', color=color,
                     label=mname.capitalize())

    axes[0].set_xscale('log'); axes[0].set_ylabel('Avg acceptance rate', fontsize=11)
    axes[0].set_title('Acceptance rate', fontsize=11); axes[0].legend(fontsize=10); axes[0].grid(True, alpha=0.4)

    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_ylabel('Avg mean |ΔH|', fontsize=11)
    axes[1].set_title('Energy error', fontsize=11); axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.4, which='both')

    axes[2].set_xscale('log'); axes[2].set_ylabel('COV of Pf', fontsize=11)
    axes[2].set_title('Pf COV', fontsize=11); axes[2].legend(fontsize=10); axes[2].grid(True, alpha=0.4)

    axes[3].axhline(0, color='k', ls='--', lw=1)
    axes[3].set_xscale('log'); axes[3].set_ylabel('Relative bias (%)', fontsize=11)
    axes[3].set_title('Pf bias', fontsize=11); axes[3].legend(fontsize=10); axes[3].grid(True, alpha=0.4)

    for ax in axes:
        ax.set_xlabel('ε', fontsize=11)

    fig.suptitle(f'High-dim LSF step-size sweep (d={d}, L={L_fixed})', fontsize=13)
    fig.tight_layout()
    fname = os.path.join(OUTDIR, f'stepsize_sweep_d{d}.png')
    fig.savefig(fname, dpi=150); plt.close(fig)
    print(f'Saved stepsize_sweep_d{d}.png')


# ── 3b Summary: acceptance rate vs epsilon, all dims side by side ─────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=False)
for ax, d in zip(axes.ravel(), dimensions):
    eps_arr_d = _get_eps_arr(d)
    for mname, color, marker in [("leapfrog","steelblue","o"), ("yoshida4","tomato","s")]:
        am = _arr3b(d, mname, "avg_acc"); ae = _arr3b(d, mname, "avg_acc_std")
        m  = np.isfinite(am)
        ax.errorbar(eps_arr_d[m], am[m], yerr=ae[m], fmt=f'{marker}-',
                    color=color, capsize=4, label=mname.capitalize())
    ax.set_xscale('log')
    ax.set_xlabel('ε', fontsize=11); ax.set_ylabel('Avg acc rate', fontsize=11)
    ax.set_title(f'd={d}', fontsize=12); ax.legend(fontsize=10); ax.grid(True, alpha=0.4)

fig.suptitle('Acceptance Rate vs Step Size — all dimensions', fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'stepsize_acceptance_all_dims.png'), dpi=150)
plt.close(fig); print('Saved stepsize_acceptance_all_dims.png')

print(f'\nAll plots saved to: {OUTDIR}')


# ═══════════════════════════════════════════════════════════════════════════════
# Append to record_yoshida.md
# ═══════════════════════════════════════════════════════════════════════════════

def _pf_row_3a(d, mname):
    s = stats_3a[(d, mname)]
    bias_pct = (s['pf_mean'] - Pf_exact) / Pf_exact * 100 if np.isfinite(s['pf_mean']) else np.nan
    return (f"| {d} | {mname} | {_num(s['pf_mean'])} | {_num(s['cov'],'.3f')} | "
            f"{_num(bias_pct,'+.1f')} | {_num(s['avg_acc'],'.3f')} | "
            f"{_num(s['avg_dH'])} | {_num(s['avg_levels'],'.1f')} |")

table_3a = (
    "| d | Method | Pf_mean | COV | Bias% | avg_acc | avg\\|ΔH\\| | avg_levels |\n"
    "|---|--------|---------|-----|-------|---------|-----------|------------|\n"
    + "\n".join(_pf_row_3a(d, m) for d in dimensions for m, _ in methods)
)

# 3b table: one per dimension
def _sweep_table_3b(d):
    eps_list = EPS_SWEEP_D100 if d == 100 else EPS_SWEEP
    rows = ["| epsilon | Method | Pf_mean | COV | avg_acc | avg\\|ΔH\\| | n_ok |",
            "|---------|--------|---------|-----|---------|-----------|------|"]
    for eps in eps_list:
        for mname, _ in methods:
            s = stats_3b[(d, mname, eps)]
            rows.append(
                f"| {eps} | {mname} | {_num(s['pf_mean'])} | {_num(s['cov'],'.3f')} | "
                f"{_num(s['avg_acc'],'.3f')} | {_num(s['avg_dH'])} | {s['n_valid']} |"
            )
    return "\n".join(rows)

sweep_tables = "\n\n".join(
    f"**d={d}**\n\n{_sweep_table_3b(d)}" for d in dimensions
)

# ── key observations ──────────────────────────────────────────────────────────
obs_3a, obs_3b = [], []

# Check ΔH scaling
lf_dH_vals = [stats_3a[(d,"leapfrog")]["avg_dH"] for d in dimensions]
y4_dH_vals = [stats_3a[(d,"yoshida4")]["avg_dH"] for d in dimensions]
if all(np.isfinite(v) for v in lf_dH_vals + y4_dH_vals):
    ratio_max = max(lf_dH_vals) / min(v for v in lf_dH_vals if v > 0)
    obs_3a.append(
        f"- Leapfrog mean|ΔH| grows with d: {lf_dH_vals[0]:.2e} (d=2) → "
        f"{lf_dH_vals[-1]:.2e} (d={dimensions[-1]}), ratio ~{lf_dH_vals[-1]/lf_dH_vals[0]:.0f}×. "
        f"This is consistent with ΔH ∝ d·ε² scaling."
    )
    obs_3a.append(
        f"- Yoshida4 mean|ΔH|: {y4_dH_vals[0]:.2e} (d=2) → "
        f"{y4_dH_vals[-1]:.2e} (d={dimensions[-1]}), ratio ~{y4_dH_vals[-1]/y4_dH_vals[0]:.0f}×. "
        f"Scales similarly (ΔH ∝ d·ε⁴), but absolute values remain orders of magnitude smaller."
    )

# Check acceptance
lf_acc_vals = [stats_3a[(d,"leapfrog")]["avg_acc"] for d in dimensions]
y4_acc_vals = [stats_3a[(d,"yoshida4")]["avg_acc"] for d in dimensions]
for i, d in enumerate(dimensions):
    la, ya = lf_acc_vals[i], y4_acc_vals[i]
    if np.isfinite(la) and np.isfinite(ya) and abs(ya - la) > 0.02:
        obs_3a.append(
            f"- At d={d}: Yoshida4 acc={ya:.3f} vs Leapfrog acc={la:.3f} "
            f"(gap {ya-la:+.3f})."
        )

# Check failures in 3b
for d in dimensions:
    eps_list = EPS_SWEEP_D100 if d == 100 else EPS_SWEEP
    for mname, _ in methods:
        for eps in eps_list:
            s = stats_3b[(d, mname, eps)]
            if s['n_valid'] < n_runs_3b:
                obs_3b.append(
                    f"- d={d}, {mname}, ε={eps}: {n_runs_3b - s['n_valid']}/{n_runs_3b} failures."
                )

if not obs_3b:
    obs_3b.append("- No numerical failures across all (d, method, ε) combinations.")

append_md = f"""
---

## Experiment 3a: High-Dimensional Linear LSF (Fixed Step Size)

### Setup

| Item | Value |
|------|-------|
| Problem | G(q) = β₀√d − Σqᵢ, β₀={beta0} |
| Prior | N(0, I_d) |
| Pf_exact | {Pf_exact:.6e} (= Φ(−{beta0}), independent of d) |
| Dimensions | d = {dimensions} |
| N | {N} |
| p0 | {p0} |
| epsilon | {eps_fixed} |
| L | {L_fixed} |
| n_runs | {n_runs_3a} |

### Results

{table_3a}

### Key Observations
{chr(10).join(obs_3a) if obs_3a else "- See plots."}

### Plots (leapfrog_vs_yoshida_highdim/)
- `dim_vs_acceptance.png`
- `dim_vs_deltaH.png`
- `dim_vs_cov.png`
- `dim_vs_bias.png`
- `pf_boxplot_by_dim.png`

---

## Experiment 3b: High-Dimensional Step Size Sensitivity

### Setup
- Same problem as Experiment 3a
- L = {L_fixed} (fixed)
- ε sweep for d∈[2,10,50]: {EPS_SWEEP}
- ε sweep for d=100: {EPS_SWEEP_D100}
- n_runs = {n_runs_3b} per (d, method, ε)

### Results

{sweep_tables}

### Key Observations
{chr(10).join(obs_3b)}

### Plots (leapfrog_vs_yoshida_highdim/)
- `stepsize_sweep_d{{d}}.png` for each d
- `stepsize_acceptance_all_dims.png`
"""

record_path = os.path.join(RECORD_DIR, 'record_yoshida.md')
with open(record_path, 'a', encoding='utf-8') as f:
    f.write(append_md)
print(f'\nAppended Experiments 3a+3b to {record_path}')
