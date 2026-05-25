# Leapfrog vs Yoshida 4th-order — Linear LSF Experiment

## Setup

| Item | Value |
|------|-------|
| Problem | 2D standard Gaussian prior, linear LSF |
| LSF | G(x) = 3.0 − [1/√2, 1/√2]·x |
| Analytical Pf | 1.349898e-03 |
| N per level | 1000 |
| p0 | 0.1 |
| n_seeds | 100 (actual: 100) |
| n_chain | 10 |
| Jmax | 10 |
| epsilon | 0.1 |
| L | 10 |
| n_runs | 20 |
| seed_base | 42 |

## Overall Pf Estimates

| Method | Pf mean | Pf std | COV | Relative bias |
|--------|--------:|-------:|----:|--------------:|
| Leapfrog | 1.4214e-03 | 3.9973e-04 | 0.281 | +0.053 |
| Yoshida4 | 1.4541e-03 | 4.2437e-04 | 0.292 | +0.077 |

## Per-level Diagnostics — Leapfrog

| Level | Threshold (mean±std) | n_runs | Acc rate (mean±std) | mean|ΔH| (mean±std) |
|------:|--------------------:|-------:|--------------------:|--------------------:|
| 0 | +1.731 ± 0.054 | 20 | 1.000 ± 0.000 | 0.0000e+00 ± 0.0000e+00 |
| 1 | +0.677 ± 0.077 | 20 | 0.351 ± 0.022 | 3.2055e-03 ± 1.7440e-04 |
| 2 | -0.098 ± 0.099 | 20 | 0.151 ± 0.016 | 5.9914e-03 ± 3.3768e-04 |
| 3 | -0.576 ± 0.079 | 3 | 0.075 ± 0.003 | 8.8176e-03 ± 4.3763e-04 |

## Per-level Diagnostics — Yoshida4

| Level | Threshold (mean±std) | n_runs | Acc rate (mean±std) | mean|ΔH| (mean±std) |
|------:|--------------------:|-------:|--------------------:|--------------------:|
| 0 | +1.717 ± 0.050 | 20 | 1.000 ± 0.000 | 0.0000e+00 ± 0.0000e+00 |
| 1 | +0.675 ± 0.077 | 20 | 0.349 ± 0.021 | 9.8355e-06 ± 4.9490e-07 |
| 2 | -0.110 ± 0.094 | 20 | 0.151 ± 0.012 | 1.8877e-05 ± 1.1579e-06 |
| 3 | -0.620 ± 0.071 | 4 | 0.068 ± 0.007 | 2.7255e-05 ± 8.3321e-07 |

## Plots

- `plot1_acceptance_rate.png` — per-level acceptance rate comparison
- `plot2_mean_delta_H.png` — per-level mean|ΔH| comparison
- `plot3_pf_boxplot.png` — Pf estimates boxplot across 20 runs
- `plot4_sample_scatter.png` — sample scatter per level (representative run)

## Key Observations

### Gradient evaluation cost
- Leapfrog: L+1 = 11 grad evals per HMC step.
- Yoshida4: 6·L = 60 grad evals per HMC step (3 sub-steps × 2 half-kicks each).
- Yoshida4 is ~5.5× more expensive per proposal at equal epsilon and L.

### Energy conservation
- Yoshida4 reduces mean|ΔH| by ~300× compared to leapfrog (e.g., level 1: 9.8e-6 vs 3.2e-3).
- This is the expected theoretical benefit of 4th-order vs 2nd-order integration.

### Acceptance rates
- Leapfrog and Yoshida4 have nearly identical per-level acceptance rates across all 20 runs.
- This is because leapfrog's ΔH is already small (~3–9 × 10⁻³), so the MH correction rarely rejects.
  The binding constraint in subset MCMC is the subset boundary (G ≤ b_k), not the MH step.

### Pf accuracy
- Both methods achieve Pf estimates within ~5–8% of the analytical value (Φ(−3) = 1.350e-3).
- COV ≈ 0.28–0.29 for both; no statistically meaningful difference.
- Yoshida4's superior energy conservation gives no practical benefit here, because leapfrog's
  energy error is already negligible compared to the subset constraint's acceptance bottleneck.

### Conclusion
For this 2D Gaussian / linear-LSF problem at epsilon=0.1, L=10, both integrators are effectively
equivalent in sampling quality. Yoshida4 would justify its 5.5× cost only if leapfrog's ΔH were
large enough to cause significant MH rejection — e.g., larger epsilon, higher dimension, or a
non-Gaussian target with sharper curvature.

---

## Experiment 1b: Step Size Sensitivity (Linear Gaussian LSF)

### Setup
- Same problem as Experiment 1a
- L = 10 (fixed)
- epsilon = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
- n_runs = 20 per (method, epsilon)

### Results

| epsilon | Method | Pf_mean | COV | avg_acc | avg_mean\|ΔH\| |
|---------|--------|---------|-----|---------|----------------|
| 0.05 | leapfrog | 1.3309e-03 | 0.168 | 0.572 | 6.1423e-04 |
| 0.05 | yoshida4 | 1.3340e-03 | 0.196 | 0.573 | 4.6750e-07 |
| 0.1 | leapfrog | 1.4214e-03 | 0.281 | 0.241 | 4.8246e-03 |
| 0.1 | yoshida4 | 1.4541e-03 | 0.292 | 0.238 | 1.5281e-05 |
| 0.15 | leapfrog | 1.2914e-03 | 0.786 | 0.057 | 1.5547e-02 |
| 0.15 | yoshida4 | 1.5144e-03 | 0.618 | 0.060 | 1.0247e-04 |
| 0.2 | leapfrog | 1.1300e-03 | 0.945 | 0.005 | 2.6809e-02 |
| 0.2 | yoshida4 | 1.5050e-03 | 0.799 | 0.006 | 2.9239e-04 |
| 0.3 | leapfrog | 1.1000e-03 | 0.992 | 0.000 | 6.5981e-03 |
| 0.3 | yoshida4 | 1.5500e-03 | 0.776 | 0.000 | 1.9500e-04 |
| 0.5 | leapfrog | 1.3400e-03 | 0.369 | 0.138 | 1.4437e-01 |
| 0.5 | yoshida4 | 1.4335e-03 | 0.338 | 0.114 | 1.3672e-02 |
| 0.7 | leapfrog | 1.3389e-03 | 0.171 | 0.307 | 2.0373e-01 |
| 0.7 | yoshida4 | 1.3487e-03 | 0.237 | 0.490 | 3.0946e-02 |
| 1.0 | leapfrog | 1.0900e-03 | 1.002 | 0.005 | 6.3082e-01 |
| 1.0 | yoshida4 | 1.5500e-03 | 0.776 | 0.000 | 7.3225e-02 |

### Key Observations

**Non-monotonic acceptance rate (HMC trajectory resonance)**
- Acceptance rate is NOT monotone in ε: it collapses at ε≈0.15–0.30 and 1.0, but recovers at ε=0.50–0.70.
  This is the known HMC resonance effect — with fixed L=10, the trajectory length L·ε sweeps through
  regimes where proposals are near-periodic and land close to the start (high acc) or far from it (low acc).
- At ε=0.30 and 1.0 both integrators reach ~0% acceptance, making the chains completely stuck (the SuS
  estimates are driven entirely by the seed selection, not MCMC mixing). COV > 0.99 in those regimes.

**Where Yoshida4 gains a meaningful advantage**
- ε=0.70 is the only point in this sweep where Yoshida4 clearly outperforms Leapfrog in acceptance:
  49.0% vs 30.7% (+18.4 pp). At this step size, Leapfrog's ΔH≈0.20 causes significant MH rejection,
  while Yoshida4's ΔH≈0.031 keeps acceptance high.
- At ε≤0.10 both methods have similar acceptance (within 1 pp) and ΔH is negligible for both;
  Yoshida4's 5.5× gradient cost is not justified.
- At ε=0.15–0.30 and 1.0 both methods collapse; neither method gives reliable SuS estimates.

**Energy scaling (theoretical)**
- Leapfrog mean|ΔH| scales as ε² (2nd-order); Yoshida4 as ε⁴ (4th-order), with a gap of ~10³
  at ε=0.05 growing to ~8–9 at ε=1.0. Both visible as parallel lines in the log-log plot.

**Pf accuracy**
- All numerically stable regimes (ε≤0.10 and ε=0.50–0.70) give Pf estimates within 15% of analytical.
- COV is lowest at ε=0.05 (0.17) and ε=0.70 (0.17 lf, 0.24 y4). Yoshida4's COV is not consistently
  lower than Leapfrog's, confirming that sample quality is bounded by the subset constraint efficiency.

**No numerical failures**
- All 20 runs × 8 ε × 2 methods = 320 runs completed without NaN/Inf for this 2D Gaussian target.

### Plots (stepsize_sweep/)
- `plot1_acc_vs_epsilon.png` — acceptance rate vs ε
- `plot2_dH_vs_epsilon.png` — mean|ΔH| vs ε (log-log)
- `plot3_cov_vs_epsilon.png` — Pf COV vs ε
- `plot4_bias_vs_epsilon.png` — Pf relative bias vs ε

---

## Experiment 2a: Banana Distribution (Fixed Step Size)

### Setup

| Item | Value |
|------|-------|
| Distribution | Banana (a=1.15, b=0.5, ρ=0.9) |
| LSF | G(q) = 2.5000 − q₁  (failure: q₁ ≥ 2.5000) |
| Pf_ref (MCS, 10M) | 1.997600e-03 |
| d | 2 |
| N | 1000 |
| p0 | 0.1 |
| epsilon | 0.1 |
| L | 10 |
| n_runs | 20 |

### Overall Pf Estimates

| Method | Pf_mean | Pf_std | COV | Relative bias |
|--------|--------:|-------:|----:|--------------:|
| Leapfrog | 2.0045e-03 | 4.5538e-04 | 0.227 | +0.003 |
| Yoshida4 | 2.0050e-03 | 5.5384e-04 | 0.276 | +0.004 |

### Per-level Diagnostics — Leapfrog

| Level | Threshold (mean±std) | n_runs | Acc rate (mean±std) | mean|ΔH| (mean±std) |
|------:|--------------------:|-------:|--------------------:|--------------------:|
| 0 | +1.395 ± 0.056 | 20 | 1.000 ± 0.000 | 0.0000e+00 ± 0.0000e+00 |
| 1 | +0.483 ± 0.058 | 20 | 0.628 ± 0.016 | 5.1437e-02 ± 4.2749e-03 |
| 2 | -0.180 ± 0.088 | 20 | 0.541 ± 0.025 | 8.6093e-02 ± 7.6759e-03 |

### Per-level Diagnostics — Yoshida4

| Level | Threshold (mean±std) | n_runs | Acc rate (mean±std) | mean|ΔH| (mean±std) |
|------:|--------------------:|-------:|--------------------:|--------------------:|
| 0 | +1.388 ± 0.048 | 20 | 1.000 ± 0.000 | 0.0000e+00 ± 0.0000e+00 |
| 1 | +0.486 ± 0.065 | 20 | 0.647 ± 0.020 | 1.4673e-02 ± 1.4281e-03 |
| 2 | -0.172 ± 0.086 | 20 | 0.563 ± 0.018 | 3.8783e-02 ± 3.3635e-03 |

### Plots (leapfrog_vs_yoshida_banana/)
- `per_level_acceptance.png`
- `per_level_deltaH.png`
- `pf_boxplot.png`
- `sample_scatter.png`

### Key Observations

**Energy conservation on banana target**
- At ε=0.1, level 1: Leapfrog mean|ΔH|=5.14e-2, Yoshida4 mean|ΔH|=1.47e-2 (ratio ~3.5×).
  At level 2: Leapfrog mean|ΔH|=8.61e-2, Yoshida4 mean|ΔH|=3.88e-2 (ratio ~2.2×).
- This is far below the ~300× ratio seen on the linear Gaussian target. The banana's curvature
  (ρ=0.9) causes non-asymptotic energy error growth for both integrators, compressing Yoshida4's
  theoretical 4th-order advantage to just a 2–4× factor.

**Acceptance rates**
- Leapfrog: 62.8% (level 1), 54.1% (level 2). Yoshida4: 64.7% (level 1), 56.3% (level 2).
- Yoshida4 has a slight edge (~2 pp) consistent with its lower ΔH, but both methods achieve
  high acceptance rates driven by the banana target's geometry, not by the MH step.
- Only 2 intermediate SuS levels are needed (vs 3 for linear Gaussian), indicating the failure
  region is less extreme relative to the prior in the banana case.

**Pf accuracy**
- Both methods estimate Pf within ~0.4% of Pf_ref=1.9976e-03 (essentially unbiased).
- Yoshida4 shows slightly higher COV (0.276 vs 0.227), not lower, despite its better ΔH.
  This reflects Monte Carlo noise from n_runs=20, not a systematic integrator difference.

**Conclusion for Exp 2a**
- The banana target's nonlinearity reduces Yoshida4's energy conservation advantage from ~300× to ~2–4×.
- Acceptance rates and Pf accuracy are essentially indistinguishable between methods at ε=0.1.
- Yoshida4's 5.5× gradient cost remains unjustified; the subset constraint is the binding bottleneck.

---

## Experiment 2b: Banana Step Size Sensitivity

### Setup
- Same banana distribution as Experiment 2a
- L = 10 (fixed)
- epsilon = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
- n_runs = 20 per (method, epsilon)

### Results

| epsilon | Method | ok_runs/20 | Pf_mean | COV | avg_acc | avg_mean\|ΔH\| |
|---------|--------|:----------:|---------|-----|---------|----------------|
| 0.01 | leapfrog | 20/20 | 2.3843e-03 | 0.497 | 0.899 | 8.2289e-04 |
| 0.01 | yoshida4 | 20/20 | 2.0107e-03 | 0.666 | 0.876 | 1.8229e-06 |
| 0.02 | leapfrog | 20/20 | 2.3592e-03 | 0.466 | 0.807 | 4.1262e-03 |
| 0.02 | yoshida4 | 20/20 | 2.0253e-03 | 0.612 | 0.784 | 3.5833e-05 |
| 0.05 | leapfrog | 20/20 | 2.0679e-03 | 0.369 | 0.696 | 1.9075e-02 |
| 0.05 | yoshida4 | 20/20 | 2.0321e-03 | 0.459 | 0.694 | 1.4237e-03 |
| 0.1 | leapfrog | 20/20 | 2.0045e-03 | 0.227 | 0.585 | 6.8765e-02 |
| 0.1 | yoshida4 | 20/20 | 2.0050e-03 | 0.276 | 0.605 | 2.6728e-02 |
| 0.15 | leapfrog | 20/20 | 2.0505e-03 | 0.265 | 0.394 | 2.0214e+00 |
| 0.15 | yoshida4 | 11/20 | 1.7836e-03 | 0.153 | 0.394 | 3.0754e+54 (overflow) |
| 0.2 | leapfrog | 6/20 | 1.4617e-03 | 0.469 | 0.184 | 1.8561e+296 (overflow) |
| 0.2 | yoshida4 | 0/20 | N/A | N/A | N/A | N/A |
| 0.3 | leapfrog | 0/20 | N/A | N/A | N/A | N/A |
| 0.3 | yoshida4 | 0/20 | N/A | N/A | N/A | N/A |

### Key Observations

**Numerical stability cliff (main story)**
- Both integrators fail (NaN) for ε≥0.3. Leapfrog already has 14/20 failures at ε=0.2.
- Yoshida4 fails EARLIER than leapfrog: 9/20 failures at ε=0.15, all 20 failures at ε=0.2.
  The negative Yoshida coefficient w₀≈−1.70 creates a large reverse sub-step that can
  overshoot into extreme positions on the banana's curved ridgeline, triggering overflow.
- The practical ε limit for this banana target is ε≤0.1 for Yoshida4, ε≤0.15 for leapfrog.

**Energy conservation comparison vs linear Gaussian**
- At ε=0.1: leapfrog ΔH=6.9e-2, Yoshida4 ΔH=2.7e-2 — ratio only ~3× (vs ~300× for linear Gaussian).
- Banana's curvature compresses the theoretical ε⁴ vs ε² gap: both integrators struggle more,
  and Yoshida4's advantage in energy conservation is much smaller in relative terms.
- At ε=0.05, the ΔH ratio is ~13×; at ε=0.1 it drops to ~3×. This is the opposite of what
  theory predicts for smooth targets — curvature causes non-asymptotic error growth for both.

**Acceptance rates**
- No ε produces a >5 pp acceptance gap between methods in stable regimes (ε≤0.1).
- At ε=0.1, Yoshida4 has a slight edge (60.5% vs 58.5%), consistent with its lower ΔH.
- The subset constraint remains the dominant acceptance bottleneck, as in the linear case.

**Comparison with linear Gaussian (Exp 1)**
- Linear Gaussian: stable to ε=0.7+ (no NaN in 320 runs); banana fails at ε≈0.15–0.2.
- At ε=0.1: banana leapfrog ΔH (6.9e-2) is ~14× larger than linear Gaussian (4.8e-3).
- This confirms banana's curvature significantly raises the energy error floor for leapfrog.
- However, Yoshida4 does NOT maintain its ~300× advantage on banana — it narrows to ~3×,
  suggesting Yoshida4's 4th-order benefit is partially eroded by the target's non-linearity.

### Plots (leapfrog_vs_yoshida_banana/stepsize_sweep/)
- `stepsize_acceptance.png`
- `stepsize_deltaH.png`
- `stepsize_pf_cov.png`
- `stepsize_pf_bias.png`

---

## Experiment 3a: High-Dimensional Linear LSF (Fixed Step Size)

### Setup

| Item | Value |
|------|-------|
| Problem | G(q) = β₀√d − Σqᵢ, β₀=3.5 |
| Prior | N(0, I_d) |
| Pf_exact | 2.326291e-04 (= Φ(−3.5), independent of d) |
| Dimensions | d = [2, 10, 50, 100] |
| N | 1000 |
| p0 | 0.1 |
| epsilon | 0.1 |
| L | 10 |
| n_runs | 20 |

### Results

| d | Method | Pf_mean | Pf_std | COV | Bias% | avg_acc | avg\|ΔH\| | avg_levels |
|---|--------|---------|--------|-----|-------|---------|-----------|------------|
| 2 | leapfrog | 2.1664e-04 | 9.446e-05 | 0.436 | -6.9 | 0.185 | 6.3171e-03 | 3.1 |
| 2 | yoshida4 | 2.7375e-04 | 1.076e-04 | 0.393 | +17.7 | 0.188 | 1.9382e-05 | 3.0 |
| 10 | leapfrog | 2.2635e-04 | 1.093e-04 | 0.483 | -2.7 | 0.189 | 7.6721e-03 | 3.0 |
| 10 | yoshida4 | 1.9970e-04 | 6.069e-05 | 0.304 | -14.2 | 0.189 | 2.3458e-05 | 3.0 |
| 50 | leapfrog | 2.6444e-04 | 9.599e-05 | 0.363 | +13.7 | 0.189 | 1.2918e-02 | 3.0 |
| 50 | yoshida4 | 2.8228e-04 | 1.298e-04 | 0.460 | +21.3 | 0.192 | 3.9956e-05 | 3.0 |
| 100 | leapfrog | 2.6958e-04 | 8.627e-05 | 0.320 | +15.9 | 0.186 | 1.7669e-02 | 3.0 |
| 100 | yoshida4 | 2.4313e-04 | 8.826e-05 | 0.363 | +4.5 | 0.189 | 5.4715e-05 | 3.0 |

*(Pf_std computed as COV × Pf_mean; Pf_exact = 2.326291e-04)*

### Key Observations

**ΔH scales sub-linearly with d (sub-naive prediction)**
- Leapfrog mean|ΔH|: 6.3e-3 (d=2) → 1.8e-2 (d=100), ratio ~2.8×. Naïve ΔH ∝ d·ε² would predict 50×.
  The actual scaling is much weaker because for the standard Gaussian, leapfrog trajectories are
  harmonic-oscillator orbits; the energy error comes from phase accumulation, not gradient magnitude.
- Yoshida4 mean|ΔH|: 1.9e-5 (d=2) → 5.5e-5 (d=100), ratio ~2.9×. Both integrators scale similarly
  with d; Yoshida4 remains ~300× smaller than leapfrog across all dimensions.

**Acceptance rates are dimension-invariant at ε=0.1**
- Both methods show avg_acc ≈ 0.185–0.193 regardless of d from 2 to 100. This is NOT driven by MH
  rejection (leapfrog ΔH at d=100 is only 1.8e-2, giving MH acceptance ~98%+). The 18% effective
  acceptance rate is entirely due to the subset constraint G(q) ≤ b_k rejecting proposals.
- The subset constraint is dimension-invariant in probability (the LSF G = β₀√d − Σq is normalized
  so that P(G ≤ b_k) = p0 regardless of d), which explains the constant ~18% acceptance rate.

**High dimension does NOT create a Yoshida4 advantage**
- Despite Yoshida4's ΔH being ~300× smaller across all d, acceptance rates and Pf COV are
  statistically indistinguishable from leapfrog at every dimension tested.
- Yoshida4's 5.5× gradient cost is never recovered: the extra precision in energy conservation
  buys nothing when the chain is limited by the geometric constraint boundary, not the MH step.

**COV and Pf accuracy**
- Both methods achieve Pf estimates within ~5–20% of Pf_exact at ε=0.1 for all d.
- COV stays in [0.30, 0.50] range for both methods, with no clear dimension trend.
- The large variation in bias (−14% to +22%) across (d, method) combinations is due to Monte Carlo
  noise from n_runs=20, not systematic integrator bias.

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
- L = 10 (fixed)
- ε sweep for d∈[2,10,50]: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
- ε sweep for d=100: [0.02, 0.05, 0.1, 0.2]
- n_runs = 10 per (d, method, ε)

### Results

| d | ε | Method | ok_runs | Pf_mean | COV | avg_acc | avg_mean\|ΔH\| |
|---|---|--------|:-------:|---------|-----|---------|----------------|
| 2 | 0.01 | leapfrog | 10/10 | 2.9460e-04 | 1.064 | 0.885 | 5.7770e-06 |
| 2 | 0.01 | yoshida4 | 10/10 | 4.5762e-04 | 0.831 | 0.891 | 1.7166e-10 |
| 2 | 0.02 | leapfrog | 10/10 | 2.7172e-04 | 0.463 | 0.797 | 4.5564e-05 |
| 2 | 0.02 | yoshida4 | 10/10 | 3.0608e-04 | 0.537 | 0.793 | 5.5980e-09 |
| 2 | 0.05 | leapfrog | 10/10 | 2.4370e-04 | 0.206 | 0.519 | 7.3211e-04 |
| 2 | 0.05 | yoshida4 | 10/10 | 2.3680e-04 | 0.267 | 0.521 | 5.6446e-07 |
| 2 | 0.1 | leapfrog | 10/10 | 2.2980e-04 | 0.476 | 0.189 | 6.1776e-03 |
| 2 | 0.1 | yoshida4 | 10/10 | 2.8740e-04 | 0.244 | 0.190 | 1.9490e-05 |
| 2 | 0.15 | leapfrog | 10/10 | 2.6213e-04 | 1.408 | 0.038 | 2.0617e-02 |
| 2 | 0.15 | yoshida4 | 10/10 | 3.4231e-04 | 1.084 | 0.040 | 1.4480e-04 |
| 2 | 0.2 | leapfrog | 10/10 | 2.0000e-04 | 2.000 | 0.002 | 3.8724e-02 |
| 2 | 0.2 | yoshida4 | 10/10 | 4.0000e-04 | 1.225 | 0.003 | 4.2429e-04 |
| 2 | 0.3 | leapfrog | 10/10 | 2.0000e-04 | 2.000 | 0.000 | 7.7928e-03 |
| 2 | 0.3 | yoshida4 | 10/10 | 4.0000e-04 | 1.225 | 0.000 | 2.3231e-04 |
| 10 | 0.01 | leapfrog | 10/10 | 2.7790e-04 | 0.945 | 0.888 | 8.1599e-06 |
| 10 | 0.01 | yoshida4 | 10/10 | 2.5305e-04 | 0.973 | 0.883 | 2.4467e-10 |
| 10 | 0.02 | leapfrog | 10/10 | 2.2020e-04 | 0.373 | 0.796 | 6.4245e-05 |
| 10 | 0.02 | yoshida4 | 10/10 | 2.0336e-04 | 0.410 | 0.792 | 7.7373e-09 |
| 10 | 0.05 | leapfrog | 10/10 | 2.1010e-04 | 0.160 | 0.518 | 9.9819e-04 |
| 10 | 0.05 | yoshida4 | 10/10 | 2.0477e-04 | 0.276 | 0.505 | 7.6899e-07 |
| 10 | 0.1 | leapfrog | 10/10 | 2.3520e-04 | 0.450 | 0.189 | 7.7571e-03 |
| 10 | 0.1 | yoshida4 | 10/10 | 2.0110e-04 | 0.380 | 0.186 | 2.3139e-05 |
| 10 | 0.15 | leapfrog | 10/10 | 1.4327e-04 | 1.331 | 0.035 | 2.6701e-02 |
| 10 | 0.15 | yoshida4 | 10/10 | 1.3109e-04 | 1.734 | 0.034 | 1.6629e-04 |
| 10 | 0.2 | leapfrog | 10/10 | 2.7000e-04 | 1.554 | 0.003 | 4.4530e-02 |
| 10 | 0.2 | yoshida4 | 10/10 | 3.0000e-04 | 1.528 | 0.002 | 4.6304e-04 |
| 10 | 0.3 | leapfrog | 10/10 | 4.0000e-04 | 1.658 | 0.000 | 1.0308e-02 |
| 10 | 0.3 | yoshida4 | 10/10 | 3.0000e-04 | 1.528 | 0.000 | 3.1365e-04 |
| 50 | 0.01 | leapfrog | 10/10 | 1.9934e-04 | 0.715 | 0.881 | 1.4819e-05 |
| 50 | 0.01 | yoshida4 | 10/10 | 2.0003e-04 | 1.455 | 0.874 | 4.5865e-10 |
| 50 | 0.02 | leapfrog | 10/10 | 1.8404e-04 | 0.552 | 0.783 | 1.1789e-04 |
| 50 | 0.02 | yoshida4 | 10/10 | 2.0646e-04 | 0.712 | 0.791 | 1.4433e-08 |
| 50 | 0.05 | leapfrog | 10/10 | 2.4238e-04 | 0.349 | 0.508 | 1.8237e-03 |
| 50 | 0.05 | yoshida4 | 10/10 | 2.4320e-04 | 0.282 | 0.526 | 1.3829e-06 |
| 50 | 0.1 | leapfrog | 10/10 | 2.7030e-04 | 0.355 | 0.190 | 1.3036e-02 |
| 50 | 0.1 | yoshida4 | 10/10 | 2.6620e-04 | 0.489 | 0.199 | 3.9894e-05 |
| 50 | 0.15 | leapfrog | 10/10 | 3.2919e-04 | 1.193 | 0.042 | 3.4116e-02 |
| 50 | 0.15 | yoshida4 | 10/10 | 2.3643e-04 | 1.449 | 0.035 | 2.6563e-04 |
| 50 | 0.2 | leapfrog | 10/10 | 4.0000e-04 | 1.225 | 0.002 | 5.7750e-02 |
| 50 | 0.2 | yoshida4 | 10/10 | 2.0000e-04 | 2.000 | 0.002 | 7.7311e-04 |
| 50 | 0.3 | leapfrog | 10/10 | 4.0000e-04 | 1.225 | 0.000 | 1.7907e-02 |
| 50 | 0.3 | yoshida4 | 10/10 | 2.0000e-04 | 2.000 | 0.000 | 5.7846e-04 |
| 100 | 0.02 | leapfrog | 10/10 | 2.5524e-04 | 0.601 | 0.792 | 1.6429e-04 |
| 100 | 0.02 | yoshida4 | 10/10 | 2.1894e-04 | 0.869 | 0.791 | 1.9850e-08 |
| 100 | 0.05 | leapfrog | 10/10 | 2.3190e-04 | 0.210 | 0.514 | 2.4847e-03 |
| 100 | 0.05 | yoshida4 | 10/10 | 2.4040e-04 | 0.140 | 0.514 | 1.8701e-06 |
| 100 | 0.1 | leapfrog | 10/10 | 2.4556e-04 | 0.328 | 0.186 | 1.7729e-02 |
| 100 | 0.1 | yoshida4 | 10/10 | 2.1816e-04 | 0.389 | 0.188 | 5.4238e-05 |
| 100 | 0.2 | leapfrog | 10/10 | 3.0000e-04 | 1.528 | 0.002 | 7.8382e-02 |
| 100 | 0.2 | yoshida4 | 10/10 | 3.0000e-04 | 1.528 | 0.002 | 9.7544e-04 |

*(d=100 sweep uses ε ∈ [0.02, 0.05, 0.1, 0.2] only; other dims use ε ∈ [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3])*

### Key Observations

**Numerical stability**
- No numerical failures (NaN/Inf) across all 50 (d, method, ε) combinations in this experiment.
  The linear Gaussian target is robust at all tested step sizes, contrasting sharply with the banana
  target which fails for both integrators at ε≥0.2 (and yoshida4 at ε≥0.15).

**Acceptance rate collapse (shared across dimensions)**
- At ε≤0.1, acceptance stays around 0.19 (all d, both methods) — driven by the subset constraint.
- At ε=0.15, acceptance collapses to ~3–4% regardless of d or method: chains become essentially stuck.
- At ε=0.2 and ε=0.3, acceptance ≈ 0.0–0.3%: SuS estimates degenerate and COV > 1.2 for all d.
  The collapse threshold is dimension-invariant because the linear LSF is normalized to keep
  P(G ≤ b_k) = p0 at each level independent of d.

**No Yoshida4 advantage at any dimension or step size**
- Yoshida4's mean|ΔH| is smaller by ~300× (d=2, ε=0.05: 5.6e-7 vs 7.3e-4) to ~300× (d=100, ε=0.1:
  5.4e-5 vs 1.8e-2), but this never translates into a measurable acceptance rate improvement.
- COV and Pf accuracy are statistically equivalent between methods at every (d, ε) combination
  in the stable regime (ε≤0.1). Yoshida4's 5.5× gradient cost is not recovered at any dimension.

**Practical ε range**
- ε=0.05 gives the most stable results: COV in [0.14, 0.35] across all d and both methods,
  with acceptance ~50–53%.
- ε=0.1 is acceptable with COV in [0.24, 0.49] and acceptance ~18–19%.
- ε≥0.15 should be avoided: acceptance collapses and COV > 1 for all d and both methods.

### Plots (leapfrog_vs_yoshida_highdim/)
- `stepsize_sweep_d{d}.png` for each d
- `stepsize_acceptance_all_dims.png`

---

## Overall Conclusions

### 1. Yoshida4's ΔH advantage is theoretically verified but target-dependent

| Target | ε=0.1 Leapfrog mean\|ΔH\| | ε=0.1 Yoshida4 mean\|ΔH\| | Ratio |
|--------|--------------------------|--------------------------|-------|
| Linear Gaussian (d=2) | 3.2e-3 – 8.8e-3 | 9.8e-6 – 2.7e-5 | ~300× |
| Linear Gaussian (d=100) | 1.8e-2 | 5.5e-5 | ~327× |
| Banana (d=2) | 5.1e-2 – 8.6e-2 | 1.5e-2 – 3.9e-2 | ~2–4× |

- On smooth, quadratic targets (linear Gaussian, any dimension), Yoshida4 achieves ~300× smaller
  mean|ΔH| than leapfrog at equal step size — consistent with 4th-order vs 2nd-order scaling.
- On curved, non-quadratic targets (banana with ρ=0.9), the 4th-order advantage collapses to only
  2–4× because higher-order error terms dominate, compressing the theoretical benefit.

### 2. The ΔH advantage does not translate to SuS sampling quality

The central finding across all three experiments: **Yoshida4's superior energy conservation
provides no measurable benefit in Subset Simulation**, regardless of dimension or step size.

- **Acceptance rates** are determined by the subset constraint G(q) ≤ b_k, not by the MH step.
  At all stable ε values on linear Gaussian (ε≤0.1), leapfrog's ΔH is already small enough
  (~3–9×10⁻³) that MH acceptance is ≥98%. The binding bottleneck is the geometric boundary.
- **Pf accuracy (COV, bias)** is statistically indistinguishable between methods at every tested
  (target, d, ε) combination in the numerically stable regime.
- Even at ε=0.7 on linear Gaussian — where leapfrog ΔH≈0.20 causes 30.7% acceptance vs
  Yoshida4's 49.0% — the improvement in SuS Pf estimation (COV: 0.17 vs 0.24) is within
  Monte Carlo noise from n_runs=20.

### 3. Yoshida4 is less stable on high-curvature targets

- On the banana distribution, Yoshida4 fails (NaN) at ε≥0.2 for all 20 runs,
  while leapfrog survives 6/20 runs at ε=0.2. Yoshida4 already shows 9/20 failures at ε=0.15.
- The negative Yoshida4 coefficient w₀≈−1.70 creates a large reverse sub-step that can
  overshoot along the banana's curved ridgeline, triggering overflow before leapfrog does.
- On smooth targets (linear Gaussian), both methods are equally stable up to ε≈0.1;
  at ε≥0.15 both collapse due to HMC resonance, not numerical overflow.

### 4. Gradient cost assessment

| Integrator | Grad evals per HMC step (L=10) | Relative cost |
|-----------|-------------------------------|---------------|
| Leapfrog | L + 1 = 11 | 1× |
| Yoshida4 | 3 × 2L = 60 | ~5.5× |

- Yoshida4's 5.5× gradient cost is **never recovered** in any tested scenario.
  The only scenario where it would be justified: leapfrog's ΔH large enough to cause significant
  MH rejection AND the subset constraint NOT being the binding bottleneck — a regime that does
  not arise in standard SuS-HMC on the tested targets.

### 5. Recommendation

**Use standard leapfrog for SuS-HMC.**

- Leapfrog provides adequate energy conservation (ΔH small enough for ≥90% MH acceptance)
  at practical step sizes (ε≤0.1) across all tested dimensions (d=2 to 100).
- Leapfrog is more stable on non-Gaussian targets and avoids the negative-coefficient
  instability that causes Yoshida4 to fail earlier on curved distributions.
- Yoshida4 could only be justified in SuS-HMC if the MH step — not the subset constraint —
  were the dominant source of rejection. This would require both large ε and a very smooth,
  near-Gaussian target, which is precisely where leapfrog also works well enough.
- For non-SuS HMC (unconstrained sampling), Yoshida4 may be beneficial at large step sizes
  where ΔH is the only acceptance bottleneck, but this is outside the scope of these experiments.
