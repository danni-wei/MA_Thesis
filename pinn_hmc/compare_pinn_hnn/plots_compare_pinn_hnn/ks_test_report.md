# KS Test Report: Baseline HMC vs PINN-HMC vs HNN-HMC

## Background

The samples come from `checkpoint_compare_pinn_hnn.pt` produced by
`compare_pinn_hnn.py`.  Each method ran for 2 000 post-burn-in steps on the
`BananaTarget` (b=0.15, σ₁=σ₂=1).  We test whether the three samplers draw
from the same marginal distributions along `q1` and `q2`.

## Method

**Two-sample Kolmogorov-Smirnov (KS) test** (`scipy.stats.ks_2samp`):

- Null hypothesis H₀: both samples come from the same continuous distribution.
- Test statistic D = max |F₁(x) − F₂(x)| (supremum of ECDF difference).
- **p-value > 0.05** → cannot reject H₀ → samples are consistent with the same distribution.
- **p-value < 0.05** → reject H₀ → samples come from statistically different distributions.

n = 2 000 per method; approximate critical value at α=0.05:
D_crit ≈ 1.358 / √(n/2) ≈ 0.0429.

## Results

| Pair | Dimension | KS statistic | p-value | Conclusion |
|------|-----------|-------------|---------|------------|
| Baseline HMC vs PINN-HMC | q1 | 0.0250 | 5.5968e-01 | ≥ 0.05 — same distribution |
| Baseline HMC vs PINN-HMC | q2 | 0.0430 | 4.9535e-02 | < 0.05 — **different distributions** |
| Baseline HMC vs HNN-HMC | q1 | 0.0350 | 1.7250e-01 | ≥ 0.05 — same distribution |
| Baseline HMC vs HNN-HMC | q2 | 0.0435 | 4.5429e-02 | < 0.05 — **different distributions** |
| PINN-HMC vs HNN-HMC | q1 | 0.0385 | 1.0316e-01 | ≥ 0.05 — same distribution |
| PINN-HMC vs HNN-HMC | q2 | 0.0165 | 9.4837e-01 | ≥ 0.05 — same distribution |

## Interpretation

**Baseline HMC vs PINN-HMC**

- q1: p = 5.5968e-01 > 0.05 → the q1 marginals are statistically indistinguishable (KS = 0.0250).
- q2: p = 4.9535e-02 < 0.05 → the q2 marginals differ significantly (KS = 0.0430); the two samplers do not agree on the q2 distribution.

**Baseline HMC vs HNN-HMC**

- q1: p = 1.7250e-01 > 0.05 → the q1 marginals are statistically indistinguishable (KS = 0.0350).
- q2: p = 4.5429e-02 < 0.05 → the q2 marginals differ significantly (KS = 0.0435); the two samplers do not agree on the q2 distribution.

**PINN-HMC vs HNN-HMC**

- q1: p = 1.0316e-01 > 0.05 → the q1 marginals are statistically indistinguishable (KS = 0.0385).
- q2: p = 9.4837e-01 > 0.05 → the q2 marginals are statistically indistinguishable (KS = 0.0165).

## Plot

`ks_test_results.png` shows the KS statistic for each pair and dimension.
The dashed red line marks the α=0.05 critical value; bars above it indicate
statistically significant differences.
