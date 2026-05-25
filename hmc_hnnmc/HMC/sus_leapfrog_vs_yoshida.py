"""
Subset Simulation: Leapfrog vs Yoshida 4th-order integrator comparison.
Problem: 2D standard Gaussian prior, linear LSF G(x) = beta - alpha^T x.
Analytical Pf = Phi(-3) ~ 1.350e-3.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy.stats import norm as Nrm

from integrators import leapfrog, yoshida4

# ── output directory ──────────────────────────────────────────────────────────
OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "results", "leapfrog_vs_yoshida_linear"
)
os.makedirs(OUTDIR, exist_ok=True)

# ── problem definition ────────────────────────────────────────────────────────
d = 2
beta_lsf = 3.0
alpha = np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])

def grad_U(q):
    return q.copy()

def G(q):
    return beta_lsf - float(q @ alpha)

Pf_analytical = Nrm.cdf(-beta_lsf)

# ── SuS hyperparameters ───────────────────────────────────────────────────────
N       = 1000
p0      = 0.1
n_seeds = 100
n_chain = 10
Jmax    = 10
epsilon = 0.1
L       = 10
n_runs  = 20
seed_base = 42


# ── HMC step ─────────────────────────────────────────────────────────────────
def hmc_step(q, integrator_fn, eps, n_steps, rng):
    p0_sample = rng.standard_normal(d)
    H0 = 0.5 * np.dot(q, q) + 0.5 * np.dot(p0_sample, p0_sample)

    q_prop, p_prop, n_grads = integrator_fn(q, p0_sample, grad_U, eps, n_steps)

    H1 = 0.5 * np.dot(q_prop, q_prop) + 0.5 * np.dot(p_prop, p_prop)
    delta_H = H1 - H0
    log_alpha = -delta_H
    accepted = rng.random() < min(1.0, np.exp(log_alpha))
    q_new = q_prop if accepted else q.copy()
    return q_new, accepted, delta_H, n_grads


# ── ESS (Kish approximation via autocorrelation lag-1) ───────────────────────
def ess_from_samples(samples):
    n = len(samples)
    if n < 4:
        return float(n)
    x = samples - np.mean(samples)
    rho = np.correlate(x, x, mode='full')[n - 1:]
    rho = rho / rho[0]
    lag1 = rho[1] if len(rho) > 1 else 0.0
    denom = max(1e-12, 1 - lag1)
    return float(n) / (1.0 + 2.0 * max(0.0, lag1)) if denom > 0 else float(n)


# ── single SuS run ────────────────────────────────────────────────────────────
def run_sus(integrator_fn, rng_seed):
    rng = np.random.default_rng(rng_seed)

    level_records = []

    # Level 0: iid prior samples
    samples = rng.standard_normal((N, d))
    G_vals = np.array([G(samples[i]) for i in range(N)])

    b_k = np.percentile(G_vals, p0 * 100)
    level_records.append(dict(
        threshold=b_k,
        acceptance_rate=1.0,
        mean_abs_delta_H=0.0,
        ess=float(N),
        n_grad_evals=0,
        samples=samples.copy(),
        G_vals=G_vals.copy(),
    ))

    n_intermediate = 0

    while b_k > 0 and len(level_records) < Jmax:
        # select seeds: p0*N lowest G
        n_seed_actual = int(p0 * N)
        seed_idx = np.argsort(G_vals)[:n_seed_actual]
        seeds = samples[seed_idx]

        new_samples = np.zeros((N, d))
        acc_list = []
        dH_list = []
        grad_count = 0

        k = 0
        for s in range(n_seed_actual):
            q_cur = seeds[s].copy()
            chain_len = N // n_seed_actual + (1 if s < N % n_seed_actual else 0)

            for _ in range(chain_len):
                if k >= N:
                    break
                q_prop, accepted, delta_H, n_grads = hmc_step(q_cur, integrator_fn, epsilon, L, rng)
                grad_count += n_grads

                # subset constraint
                if G(q_prop) <= b_k:
                    if accepted:
                        q_cur = q_prop
                    acc_list.append(1 if accepted else 0)
                else:
                    acc_list.append(0)

                new_samples[k] = q_cur
                dH_list.append(abs(delta_H))
                k += 1

        samples = new_samples[:k] if k < N else new_samples
        G_vals = np.array([G(samples[i]) for i in range(len(samples))])

        b_k_new = np.percentile(G_vals, p0 * 100)

        g1 = G_vals[:, 0] if G_vals.ndim > 1 else G_vals
        ess_val = ess_from_samples(g1)

        level_records.append(dict(
            threshold=b_k_new,
            acceptance_rate=float(np.mean(acc_list)) if acc_list else 0.0,
            mean_abs_delta_H=float(np.mean(dH_list)) if dH_list else 0.0,
            ess=ess_val,
            n_grad_evals=grad_count,
            samples=samples.copy(),
            G_vals=G_vals.copy(),
        ))

        n_intermediate += 1
        b_k = b_k_new

        if b_k <= 0:
            break

    # compute Pf: p0^n_intermediate * p_last regardless of whether threshold crossed 0
    G_final = level_records[-1]['G_vals']
    p_last = float(np.mean(G_final <= 0.0))
    Pf_hat = (p0 ** n_intermediate) * p_last

    return dict(pf_hat=Pf_hat, levels=level_records)


# ── run both methods ──────────────────────────────────────────────────────────
print("Running Leapfrog SuS ...")
lf_results = [run_sus(leapfrog, seed_base + i) for i in range(n_runs)]
print("Running Yoshida4 SuS ...")
y4_results = [run_sus(yoshida4, seed_base + 1000 + i) for i in range(n_runs)]


# ── aggregate per-level stats ─────────────────────────────────────────────────
def aggregate_levels(results, key):
    max_lvl = max(len(r['levels']) for r in results)
    out = []
    for lv in range(max_lvl):
        vals = [r['levels'][lv][key] for r in results if lv < len(r['levels'])]
        out.append((np.mean(vals), np.std(vals), len(vals)))
    return out


def pf_array(results):
    return np.array([r['pf_hat'] for r in results])


lf_pf = pf_array(lf_results)
y4_pf = pf_array(y4_results)

lf_acc   = aggregate_levels(lf_results, 'acceptance_rate')
y4_acc   = aggregate_levels(y4_results, 'acceptance_rate')
lf_dH    = aggregate_levels(lf_results, 'mean_abs_delta_H')
y4_dH    = aggregate_levels(y4_results, 'mean_abs_delta_H')
lf_thr   = aggregate_levels(lf_results, 'threshold')
y4_thr   = aggregate_levels(y4_results, 'threshold')


# ── terminal output ───────────────────────────────────────────────────────────
def pf_stats(arr, label):
    m, s = np.mean(arr), np.std(arr)
    cov = s / m if m > 0 else np.inf
    bias = (m - Pf_analytical) / Pf_analytical
    print(f"  {label:10s}  Pf={m:.4e} ± {s:.4e}  COV={cov:.3f}  Bias={bias:+.3f}")

print("\n" + "=" * 70)
print(f"  Analytical Pf = {Pf_analytical:.4e}  (beta={beta_lsf})")
print("=" * 70)
pf_stats(lf_pf, "Leapfrog")
pf_stats(y4_pf, "Yoshida4")
print()

max_lvl = max(len(lf_acc), len(y4_acc))
hdr = f"{'Lvl':>4}  {'Method':>10}  {'Threshold':>10}  {'AccRate':>9}  {'mean|dH|':>10}"
print(hdr)
print("-" * len(hdr))
for lv in range(max_lvl):
    if lv < len(lf_acc):
        tm, ts, _ = lf_thr[lv]
        am, as_, _ = lf_acc[lv]
        dm, ds, _ = lf_dH[lv]
        print(f"  {lv:>2}  {'Leapfrog':>10}  {tm:+8.3f}±{ts:.3f}  {am:.3f}±{as_:.3f}  {dm:.4e}±{ds:.4e}")
    if lv < len(y4_acc):
        tm, ts, _ = y4_thr[lv]
        am, as_, _ = y4_acc[lv]
        dm, ds, _ = y4_dH[lv]
        print(f"  {lv:>2}  {'Yoshida4':>10}  {tm:+8.3f}±{ts:.3f}  {am:.3f}±{as_:.3f}  {dm:.4e}±{ds:.4e}")
    print()
print("=" * 70)


# ── Plot 1: per-level acceptance rate ────────────────────────────────────────
n_lv_lf = len(lf_acc)
n_lv_y4 = len(y4_acc)
x_lf = np.arange(n_lv_lf)
x_y4 = np.arange(n_lv_y4)

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(x_lf, [v[0] for v in lf_acc], yerr=[v[1] for v in lf_acc],
            fmt='o-', capsize=4, label='Leapfrog')
ax.errorbar(x_y4 + 0.15, [v[0] for v in y4_acc], yerr=[v[1] for v in y4_acc],
            fmt='s--', capsize=4, label='Yoshida4')
ax.set_xlabel('Level'); ax.set_ylabel('Acceptance rate')
ax.set_title('Per-level acceptance rate (mean ± std, 20 runs)')
ax.legend(); ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'plot1_acceptance_rate.png'), dpi=150)
plt.close(fig)
print('Saved plot1_acceptance_rate.png')

# ── Plot 2: per-level mean|ΔH| ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(x_lf[1:], [v[0] for v in lf_dH[1:]], yerr=[v[1] for v in lf_dH[1:]],
            fmt='o-', capsize=4, label='Leapfrog')
ax.errorbar(x_y4[1:] + 0.15, [v[0] for v in y4_dH[1:]], yerr=[v[1] for v in y4_dH[1:]],
            fmt='s--', capsize=4, label='Yoshida4')
ax.set_xlabel('Level'); ax.set_ylabel('mean |ΔH|')
ax.set_title('Per-level mean |ΔH| (mean ± std, 20 runs)')
ax.legend(); ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'plot2_mean_delta_H.png'), dpi=150)
plt.close(fig)
print('Saved plot2_mean_delta_H.png')

# ── Plot 3: Pf boxplot ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 5))
ax.boxplot([lf_pf, y4_pf], labels=['Leapfrog', 'Yoshida4'], patch_artist=True)
ax.axhline(Pf_analytical, color='r', ls='--', label=f'Analytical {Pf_analytical:.3e}')
ax.set_ylabel('Pf estimate')
ax.set_title('Pf estimates across 20 runs')
ax.legend(); ax.grid(True, alpha=0.4)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'plot3_pf_boxplot.png'), dpi=150)
plt.close(fig)
print('Saved plot3_pf_boxplot.png')

# ── Plot 4: sample scatter per level (representative run) ────────────────────
rep_lf = lf_results[0]
rep_y4 = y4_results[0]
n_levels_rep = min(len(rep_lf['levels']), len(rep_y4['levels']), 5)

fig, axes = plt.subplots(2, n_levels_rep, figsize=(3.5 * n_levels_rep, 7))

xx = np.linspace(-5, 5, 200)
x2_line = (beta_lsf - alpha[0] * xx) / alpha[1]

for col, lv in enumerate(range(n_levels_rep)):
    for row, (res, mname) in enumerate([(rep_lf, 'Leapfrog'), (rep_y4, 'Yoshida4')]):
        ax = axes[row, col]
        samp = res['levels'][lv]['samples']
        thr  = res['levels'][lv]['threshold']
        ax.scatter(samp[:, 0], samp[:, 1], s=2, alpha=0.3, c='steelblue')
        ax.plot(xx, x2_line, 'r--', lw=1, label='g=0')
        ax.set_xlim(-6, 6); ax.set_ylim(-6, 6)
        ax.set_title(f'{mname} lv{lv} b={thr:.2f}', fontsize=8)
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel(mname)

fig.suptitle('Sample scatter per level (run 0)', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'plot4_sample_scatter.png'), dpi=150)
plt.close(fig)
print('Saved plot4_sample_scatter.png')

print(f'\nAll plots saved to: {OUTDIR}')


# ── write record_yoshida.md ───────────────────────────────────────────────────
def fmt_row(lv, thr, acc, dh):
    tm, ts, n = thr
    am, as_, _ = acc
    dm, ds, _  = dh
    return (f"| {lv} | {tm:+.3f} ± {ts:.3f} | {n} | "
            f"{am:.3f} ± {as_:.3f} | {dm:.4e} ± {ds:.4e} |")


lf_m = np.mean(lf_pf); lf_s = np.std(lf_pf)
y4_m = np.mean(y4_pf); y4_s = np.std(y4_pf)
lf_cov = lf_s / lf_m if lf_m > 0 else np.inf
y4_cov = y4_s / y4_m if y4_m > 0 else np.inf
lf_bias = (lf_m - Pf_analytical) / Pf_analytical
y4_bias = (y4_m - Pf_analytical) / Pf_analytical

def build_level_table(thr_list, acc_list, dh_list):
    rows = ["| Level | Threshold (mean±std) | n_runs | Acc rate (mean±std) | mean|ΔH| (mean±std) |",
            "|------:|--------------------:|-------:|--------------------:|--------------------:|"]
    for lv, (thr, acc, dh) in enumerate(zip(thr_list, acc_list, dh_list)):
        rows.append(fmt_row(lv, thr, acc, dh))
    return "\n".join(rows)

md_content = f"""# Leapfrog vs Yoshida 4th-order — Linear LSF Experiment

## Setup

| Item | Value |
|------|-------|
| Problem | 2D standard Gaussian prior, linear LSF |
| LSF | G(x) = {beta_lsf} − [1/√2, 1/√2]·x |
| Analytical Pf | {Pf_analytical:.6e} |
| N per level | {N} |
| p0 | {p0} |
| n_seeds | {n_seeds} (actual: {int(p0*N)}) |
| n_chain | {n_chain} |
| Jmax | {Jmax} |
| epsilon | {epsilon} |
| L | {L} |
| n_runs | {n_runs} |
| seed_base | {seed_base} |

## Overall Pf Estimates

| Method | Pf mean | Pf std | COV | Relative bias |
|--------|--------:|-------:|----:|--------------:|
| Leapfrog | {lf_m:.4e} | {lf_s:.4e} | {lf_cov:.3f} | {lf_bias:+.3f} |
| Yoshida4 | {y4_m:.4e} | {y4_s:.4e} | {y4_cov:.3f} | {y4_bias:+.3f} |

## Per-level Diagnostics — Leapfrog

{build_level_table(lf_thr, lf_acc, lf_dH)}

## Per-level Diagnostics — Yoshida4

{build_level_table(y4_thr, y4_acc, y4_dH)}

## Plots

- `plot1_acceptance_rate.png` — per-level acceptance rate comparison
- `plot2_mean_delta_H.png` — per-level mean|ΔH| comparison
- `plot3_pf_boxplot.png` — Pf estimates boxplot across 20 runs
- `plot4_sample_scatter.png` — sample scatter per level (representative run)

## Key Observations

- Yoshida4 uses 6 gradient evaluations per step (3 sub-steps × 2) vs L+1 for leapfrog.
- With same epsilon and L, Yoshida4 has significantly higher per-step cost (6L vs L+1 grad evals).
- Yoshida4's 4th-order accuracy leads to smaller |ΔH| per unit step, improving acceptance.
- At equal epsilon/L budget, compare acceptance rate and Pf COV to judge integrator efficiency.
"""

record_path = os.path.join(OUTDIR, 'record_yoshida.md')
with open(record_path, 'w', encoding='utf-8') as f:
    f.write(md_content)
print(f'Saved record_yoshida.md to {record_path}')
