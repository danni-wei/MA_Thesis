# Jacobian Correction Experiment: PINN-HMC vs PINN-HMC-corrected

## 1. Background

Standard HMC uses a leapfrog (symplectic) integrator, which is volume-preserving:
`|det J_leapfrog| = 1`. The Metropolis-Hastings acceptance ratio simplifies to

    α = min(1, exp(−ΔH))

When the leapfrog is replaced by a PINN (Physics-Informed Neural Network), the
map `(q₀, p₀) → (q_T, p_T)` is no longer guaranteed to be volume-preserving.
The correct acceptance ratio is

    α = min(1, exp(−ΔH − log|det J_PINN(q₀, p₀)|))

Prior work (experiment_sus.py) used the *uncorrected* formula, which may bias
the stationary distribution and distort failure-probability estimates.

This experiment quantifies the discrepancy.

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Target | `WangBananaTarget` (Wang 2019): a=1.15, b=0.5, ρ=0.9 |
| Limit-state | `EllipticalLimitState`: c₁=1, c₂=0.5, θ=π/4 |
| Radii r | 10, 12 |
| PINN architecture | 4-layer MLP, 128 hidden, Tanh |
| PINN training | 120 epochs, Adam 1e-3, ProgressBalancer α=1 |
| Dataset | 1 800 trajectories, step size 0.08, 20 leapfrog steps |
| SuS N | 200 samples/level (reduced vs. experiment_sus due to Jacobian cost) |
| SuS p₀ | 0.1 |
| SuS max levels | 20 |
| SuS burn-in | 200 |
| Jacobian | torch.autograd.functional.jacobian (4×4), slogdet for stability |

## 3. Jacobian Statistics

| Statistic | Value |
|-----------|-------|
| Mean log|det J| | -0.2936 |
| Std  log|det J| | 0.4987 |
| Min  log|det J| | -3.3202 |
| Max  log|det J| | 0.4101 |
| Mean |det J|    | 0.8119 |

A mean log|det J| of -0.2936 means the PINN map on average
shrinks phase-space volume by a factor
of ≈ 0.746.  For a truly symplectic map this would be 0.

## 4. Results

| Method | r | P_F | Levels | Train (s) | Prop (ms) | Samp (s) |
|--------|---|-----|--------|-----------|-----------|----------|
| Baseline HMC | 10 | 3.5000e-02 | 2 | 0.00 | 26.488 | 5.62 |
| PINN-HMC | 10 | 7.3500e-03 | 3 | 226.49 | 1.223 | 0.27 |
| PINN-HMC-corrected | 10 | 2.5500e-03 | 3 | 226.49 | 646.281 | 140.90 |
| Baseline HMC | 12 | 1.7000e-02 | 2 | 0.00 | 24.315 | 5.09 |
| PINN-HMC | 12 | 4.5000e-11 | 11 | 226.49 | 1.234 | 0.40 |
| PINN-HMC-corrected | 12 | 2.5000e-08 | 8 | 226.49 | 538.715 | 141.71 |

## 5. Analysis

[TODO: Fill in after running the experiment]

Key questions to address:
- Does the Jacobian correction significantly change P_F estimates?
  If |log|det J|| is small (< 0.01), the correction is negligible.
  If it is large, the uncorrected PINN-HMC may produce a biased chain.
- Does the corrected acceptance rate differ substantially from the uncorrected?
  Higher |log|det J| ⇒ larger discrepancy in acceptance probabilities.
- Is the computational overhead of the Jacobian correction justified given the
  difference in P_F estimates?

## 6. Limitations

* **N = 200**: Smaller than experiment_sus (N=500) because computing N
  Jacobians per SuS step is expensive.  This increases estimator variance
  (CoV ≈ 0.21 per level).
* **Single run**: No confidence intervals; results are indicative.
* **Fixed PINN**: The same trained model is reused across all SuS levels.
  Level-adaptive retraining (RS-HMC-SS) could reduce distribution-shift bias.
* **4D Jacobian only**: We consider the full (q, p) map; Jacobian in q-space
  alone (marginalising p) would require further analysis.
