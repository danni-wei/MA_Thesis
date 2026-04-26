# PINN-HMC Experiments Record

---

## 1. MVP Experiment (Initial PINN-HMC)

### Setup
- Model: Trajectory PINN
- Loss: fixed weights
- λ = (1.0, 1.0, 1.0)
- Target: banana distribution
- Sampler: PINN + MH correction

### Results

#### Training
| Metric | Value |
|--------|------:|
| Final train loss | ~0.50 |
| Final val loss | ~0.45 |

#### Trajectory Quality
| Metric | Value |
|--------|------:|
| mse_q | 8.68e-04 |
| mse_p | 4.24e-02 |
| mean_H_drift_hat | 1.09e-01 |

#### HMC Quality
| Metric | Value |
|--------|------:|
| accept_rate | 1.000 |
| mean_abs_delta_H | 0.7227 |

#### Sample Statistics
- mean ≈ baseline
- std slightly underestimated

### Notes
- PINN can learn trajectories, but:
- **Huge ΔH → poor Hamiltonian preservation**
- Sampling quality unstable
- → Need better loss balancing

---

## 2. Loss Balancing Methods Comparison

### Methods
1. Fixed λ
2. EMA adaptive λ
3. Progress-based λ
4. GradNorm λ

---

## Loss Balancing Strategies

In PINN training, the total loss is defined as a weighted sum of multiple objectives:

\[
L = \lambda_{\text{sup}} L_{\text{sup}} + \lambda_{\text{phys}} L_{\text{phys}} + \lambda_{\text{ic}} L_{\text{ic}}
\]

where:
- \(L_{\text{sup}}\): supervised loss (trajectory fitting)
- \(L_{\text{phys}}\): physics residual (Hamiltonian dynamics)
- \(L_{\text{ic}}\): initial condition constraint

A key challenge is that these loss terms typically have different scales and learning dynamics, which can lead to imbalance during training. To address this, we explored several weighting strategies.

---

### 1. Fixed Weighting

The simplest approach uses constant coefficients:

\[
\lambda_i = \text{constant}
\]

for all training steps.

**Interpretation:**
- Manually assigns importance to each objective.
- Serves as a baseline method.

**Limitation:**
- Requires manual tuning.
- Sensitive to scale differences between loss terms.

---

### 2. Scale-Based Weighting (EMA)

We maintain an exponential moving average (EMA) of each loss:

\[
\tilde{L}_i(t) = \beta \tilde{L}_i(t-1) + (1 - \beta) L_i(t)
\]

The weights are defined as:

\[
\lambda_i = \frac{1}{\tilde{L}_i(t) + \epsilon}
\]

(optionally normalized so that \(\sum_i \lambda_i = 1\)).

**Interpretation:**
- Loss terms with smaller magnitude receive larger weights.
- Attempts to compensate for scale imbalance.

**Limitation:**
- If \(L_i \to 0\), then \(\lambda_i \to \infty\), leading to instability.
- In practice, may cause one loss term to dominate training.

---

### 3. Progress-Based Weighting

We define the relative training progress of each loss:

\[
r_i(t) = \frac{L_i(t)}{L_i(0)}
\]

Weights are assigned inversely proportional to progress:

\[
\lambda_i \propto \frac{1}{r_i(t)}
\]

and normalized:

\[
\lambda_i = \frac{\frac{1}{r_i(t)}}{\sum_j \frac{1}{r_j(t)}}
\]

**Interpretation:**
- Loss terms that decrease slowly (large \(r_i\)) receive higher weights.
- Balances learning speed across objectives rather than scale.

**Advantage:**
- More stable than scale-based methods.
- Automatically focuses training on under-optimized components.

---

### 4. Gradient-Based Weighting (GradNorm)

Let the gradient magnitude of each loss term be:

\[
g_i = \left\| \nabla_\theta \left( \lambda_i L_i \right) \right\|
\]

The goal is to balance gradients across tasks:

\[
g_i \approx \bar{g}
\]

where \(\bar{g}\) is the average gradient norm.

The weights are updated to minimize:

\[
\sum_i \left| g_i - \bar{g} \cdot r_i^\alpha \right|
\]

where:
- \(r_i\) represents relative training progress
- \(\alpha\) controls the strength of balancing

**Interpretation:**
- Ensures each loss contributes equally to parameter updates.
- Operates directly at the optimization level.

**Advantage:**
- More principled than scale-based methods.
- Aligns optimization dynamics across objectives.

**Limitation:**
- Requires additional gradient computations.
- More complex to implement.

---

### Summary

These methods correspond to different balancing principles:

- **Fixed:** manual weighting
- **Scale-based (EMA):** balance by loss magnitude
- **Progress-based:** balance by learning speed
- **GradNorm:** balance by gradient magnitude

Among them, progress-based weighting provides the best trade-off between stability and performance in our experiments.


### 2.1 Fixed (baseline)

| Metric | Value |
|--------|------:|
| Final val loss | 3.86e-03 |
| mean_abs_delta_H | (not evaluated here) |

#### Notes
- Stable training
- Requires manual tuning
- Used as baseline

---

### 2.2 EMA (failed case)

#### Observations
- λ_phys → extremely large (up to 1e6+)
- Training dominated by physics loss
- sup / ic ignored

#### Results
| Metric | Value |
|--------|------:|
| mse_q | 4.59e-01 |
| mse_p | 5.18e-01 |
| mean_abs_delta_H | ~1.00 |

#### Notes
- **Training collapse**
- Severe imbalance between loss terms
- Not usable

---

### 2.3 Progress-based (best PINN)

#### Results
| Metric | Value |
|--------|------:|
| mse_q | 6.72e-04 |
| mse_p | 6.15e-04 |
| mean_abs_delta_H | 4.32e-02 |
| accept_rate | 0.9828 |

#### Sample Stats
- mean ≈ correct
- std ≈ [1.01, 1.01]

#### Notes
- **Best overall PINN performance**
- Stable λ (~0.1, 0.8, 0.07)
- Balanced learning progress

---

### 2.4 GradNorm

#### Results
| Metric | Value |
|--------|------:|
| mse_q | 1.32e-04 |
| mse_p | 4.66e-04 |
| mean_abs_delta_H | 3.63e-02 |
| accept_rate | 0.9836 |

#### Notes
- Good performance
- Slightly worse than progress method in practice
- More complex mechanism

---

### Summary of Loss Methods

| Method | Stability | ΔH | Verdict |
|--------|----------|----|--------|
| Fixed | ✓ | medium | baseline |
| EMA | ✗ | huge | failure |
| Progress | ✓✓ | low | **best** |
| GradNorm | ✓ | low | good |

---

## 3. PINN-HMC vs HNNMC-style Comparison

### Unified Setup
- same target distribution
- same dataset (trajectory-derived)
- same model size (128, depth=4)
- same training budget (120 epochs)
- same sampler config

---

### Results

| Method | accept_rate | mean_abs_delta_H | std |
|--------|------------:|-----------------:|-----|
| Baseline HMC | 0.9992 | 0.00157 | [1.013, 1.030] |
| PINN-HMC | 0.9860 | 0.05711 | [0.972, 1.031] |
| HNNMC-style | 0.9896 | 0.01853 | [0.978, 1.023] |

---

### Diagnostics

#### PINN
- mse_q = 2.90e-04  
- mse_p = 4.47e-04  
- mean_H_drift = 1.47e-02  

#### HNN
- total loss = 2.08e-04  
- strong vector field learning  

---

### Interpretation

- Baseline HMC remains the gold standard
- HNNMC-style significantly reduces ΔH compared to PINN-HMC
- PINN learns trajectories well but:
  - does not preserve Hamiltonian structure as well
- HNN aligns better with HMC dynamics:
  - better proposal quality
  - lower energy error

---

### Key Insight

> Good trajectory approximation ≠ good HMC proposal

- PINN optimizes trajectory fitting
- HNN optimizes dynamics structure
- HMC depends more on **structure preservation than trajectory accuracy**

---

### Conclusion

- PINN-HMC is viable but not optimal in current form
- HNN-based surrogate is better for:
  - energy conservation
  - proposal stability
- PINN advantage:
  - direct trajectory mapping
  - potential amortization benefit

---

## 4. Overall Conclusions

### What worked
- Progress-based λ balancing
- PINN trajectory learning
- HNN surrogate for HMC

### What failed
- EMA weighting (instability)

### Core finding

> Learning Hamiltonian structure is more effective than learning trajectory directly for HMC proposal quality.

---

## Next Steps

- Improve PINN:
  - add structure-preserving loss
  - constrain Hamiltonian drift
- Compare computation cost (PINN vs HNN)
- Extend to higher dimension / harder targets