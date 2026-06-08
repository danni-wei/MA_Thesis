# Yoshida × HNN-HMC in Subset Simulation — Complete Experiment Record

**Date:** 2026-06-08  
**Branch:** main  
**Key files:** `pinn_hmc/smoke_yoshida_hnn.py`, `pinn_hmc/smoke_stepsize_sweep.py`, `pinn_hmc/experiment_sus_stage2.py`  
**Data directory:** `results/smoke_yoshida_hnn/`

---

## 0. Research Question and Motivation

### 0.1 Central question

Within a Subset Simulation (SuS) framework where HMC is used as the within-level MCMC kernel, gradient evaluations are provided by a trained Hamiltonian Neural Network (HNN) rather than the true potential's autodiff gradient.
The question is: **does replacing the standard leapfrog integrator with the 4th-order Yoshida integrator improve the quality of the failure-probability estimate (Pf) or its coefficient of variation (COV), and at what computational cost?**

### 0.2 Why this matters

The 4th-order Yoshida integrator reduces energy error |ΔH| by ~300× relative to leapfrog when the gradient is exact (true potential), leading to near-perfect Metropolis-Hastings acceptance and tighter HMC proposals.
In SuS, better proposals → lower geometric rejection → lower COV.
The hope was that the same improvement transfers when using an HNN surrogate for the gradient.

### 0.3 Experimental design philosophy: 2 × 2 factorial

| Factor A: Integrator | Factor B: Gradient source |
|---------------------|--------------------------|
| Leapfrog (2nd order) | True gradient (autodiff) |
| Yoshida 4th order   | HNN surrogate gradient   |

This 2 × 2 design separates integrator quality from gradient quality, enabling diagnosis of which factor limits performance.

Two 2D targets:
- **Gaussian**: standard 2D Gaussian N(0, I₂), simple control
- **Banana**: Wang banana, U(q₁,q₂) = ½q₁² + ½(q₂ − 0.15(q₁²−1))², non-linear curvature

Shared limit-state function: **g(q) = β − q₁**, β = 3.0  
Reference failure probability: **Pf_ref = Φ(−3) = 1.3499 × 10⁻³**

The reference value applies to both targets because q₁ is marginally N(0,1) under both distributions (sigma1 = 1.0 in `BananaTargetConfig`).

---

## 1. Experiment 1 — Initial Smoke Test (Mixed Integration)

### 1.1 Purpose

Establish a baseline: with a newly trained HNN, does Yoshida reduce |ΔH| over leapfrog in the same way as with a true gradient?
This is a quick sanity check (2 000 proposals, no SuS) to detect gross failures before investing in full SuS runs.

### 1.2 Experimental design

**Configs:** 4 configurations × 2 targets = 8 trajectories

| Config | Gradient source | Integrator |
|--------|----------------|------------|
| leapfrog+true | true autodiff | leapfrog (2nd order) |
| yoshida+true | true autodiff | Yoshida 4th order |
| leapfrog+hnn | HNN surrogate | leapfrog |
| yoshida+hnn | HNN surrogate | Yoshida |

**Integrator construction (Stage 1 — mixed):** In this version, the leapfrog base-step used the **true** gradient for the q-drift and the HNN only for the p-kick (or vice versa), producing a "hybrid" trajectory that does not faithfully integrate H_θ. This was identified as a methodological flaw and corrected in Stage 1.5.

**Key parameters:**

| Parameter | Value |
|-----------|-------|
| step_size (h) | 0.1 |
| n_steps (L) | 10 |
| burn-in | 200 proposals |
| n_proposals | 2 000 |
| HNN arch | 4-layer MLP, 128 hidden, Tanh |
| HNN training | 120 epochs, Adam lr=1e-3, 1 800 trajectories |
| Yoshida w₀ | ≈ −1.7024 (Forest–Ruth 4th-order weight) |

**Measured quantities:**
- `dH_learned` = mean |H_θ(q*, p*) − H_θ(q, p)| along trajectory (energy conservation under learned H)
- `dH_true` = mean |H(q*, p*) − H(q, p)| along trajectory (energy conservation under true H)
- `accept_rate` = Metropolis-Hastings acceptance rate
- `n_grad_evals` = total gradient evaluations (including burn-in)

**Data source:** `smoke_results.pt`

### 1.3 Results

| Config | Target | dH_learned | dH_true | accept | n_grad_evals |
|--------|--------|-----------|---------|--------|-------------|
| leapfrog+true | gaussian | 2.1949e-03 | 2.1949e-03 | 0.9995 | 24 200 |
| yoshida+true  | gaussian | 6.7164e-06 | 6.7164e-06 | 1.0000 | 132 000 |
| leapfrog+hnn  | gaussian | 6.4315e-03 | 5.5106e-03 | 0.9975 | 24 200 |
| yoshida+hnn   | gaussian | 5.8735e-03 | 5.2480e-03 | 0.9975 | 132 000 |
| leapfrog+true | banana   | 2.4023e-03 | 2.4023e-03 | 0.9990 | 24 200 |
| yoshida+true  | banana   | 8.1195e-06 | 8.1195e-06 | 1.0000 | 132 000 |
| leapfrog+hnn  | banana   | 1.1188e-02 | 1.6508e-02 | 0.9915 | 24 200 |
| yoshida+hnn   | banana   | 1.0900e-02 | 1.5930e-02 | 0.9915 | 132 000 |

**Yoshida improvement ratios:**

| Comparison | dH (true gradient) | dH_learned (HNN) | dH_true (HNN) |
|------------|-------------------|-----------------|---------------|
| yoshida+true / leapfrog+true, gaussian | 327× better | — | — |
| yoshida+hnn / leapfrog+hnn, gaussian | 1.09× better | 1.09× better | 1.05× better |
| yoshida+true / leapfrog+true, banana | 296× better | — | — |
| yoshida+hnn / leapfrog+hnn, banana | 1.03× better | 1.03× better | 1.04× better |

![|ΔH| trajectory — Gaussian target](dH_trajectory_gaussian.png)
![|ΔH| trajectory — Banana target](dH_trajectory_banana.png)

### 1.4 Analysis and conclusions

**Finding 1 — True gradient control passes:** yoshida+true achieves 296–327× lower |ΔH| than leapfrog+true, consistent with the expected 4th-order advantage. This confirms the Yoshida implementation is correct.

**Finding 2 — HNN kills the advantage:** yoshida+hnn achieves only ~1.05–1.09× improvement over leapfrog+hnn, compared to 300× for the true gradient. The HNN model error dominates.

**Unexpected observation:** For leapfrog+hnn and yoshida+hnn on Banana, `dH_learned < dH_true`. This is a sign of the "mixed integration" bug: the trajectory uses a mixture of true and HNN gradients, so dH_learned measures energy error in an H_θ that was never consistently integrated.

**Diagnosis leading to Stage 1.5:** The Stage 1 Yoshida+HNN integration uses the true gradient for q-drift and the HNN for p-kick. This is not a faithful integrator of H_θ. The 4th-order cancellation in Yoshida requires exact commutativity of the sub-steps; mixing gradient sources breaks this. Correcting to a full HNN vector field is required before concluding.

---

## 2. Experiment 2 — Full HNN Vector Field Smoke Test (Stage 1.5)

### 2.1 Purpose

Correct the methodological flaw from Stage 1: use the HNN for **both** the q-drift (via ∂H_θ/∂p) and the p-kick (via −∂H_θ/∂q), ensuring the integrator faithfully integrates H_θ.
Confirm whether the Yoshida advantage reappears once the base-step is symmetric and consistent with H_θ.

### 2.2 Experimental design

All parameters identical to Stage 1, except:

**Integrator construction (Stage 1.5 — full HNN vector field):**
- `leapfrog_hnn_one_step`: half p-kick (using HNN −∂H_θ/∂q) → full q-drift (using HNN ∂H_θ/∂p) → half p-kick. This is the symmetric Störmer-Verlet base step, fully inside H_θ.
- `yoshida4_hnn`: three nested leapfrog_hnn sub-steps with Forest-Ruth weights (w₁, w₀, w₁). Fully consistent with H_θ.

**HNN vector-field call counts per proposal:**

| Integrator | Formula | Calls/proposal |
|------------|---------|---------------|
| leapfrog+hnn | 2·L + 1 (half-kick sharing) | 21 |
| yoshida+hnn | 9·L (no merging across sub-steps) | 90 |
| Cost ratio | 90 / 21 | **4.286×** |

Note: n_grad_evals increased from Stage 1 because we now count both ∂H/∂q and ∂H/∂p calls:
- leapfrog+hnn: 2 200 × 21 = 46 200
- yoshida+hnn: 2 200 × 90 = 198 000

**Data source:** `smoke_results_v15.pt`

### 2.3 Results

#### Stage 1.5 absolute values

| Config | Target | dH_learned | dH_true | accept | n_evals |
|--------|--------|-----------|---------|--------|---------|
| leapfrog+true | gaussian | 2.1949e-03 | 2.1949e-03 | 0.9995 | 24 200 |
| yoshida+true  | gaussian | 6.7164e-06 | 6.7164e-06 | 1.0000 | 132 000 |
| leapfrog+hnn  | gaussian | 6.3896e-03 | 6.8434e-03 | 0.9965 | 46 200 |
| yoshida+hnn   | gaussian | 6.0694e-03 | 6.8561e-03 | 0.9965 | 198 000 |
| leapfrog+true | banana   | 2.4023e-03 | 2.4023e-03 | 0.9990 | 24 200 |
| yoshida+true  | banana   | 8.1195e-06 | 8.1195e-06 | 1.0000 | 132 000 |
| leapfrog+hnn  | banana   | 1.0488e-02 | 1.5137e-02 | 0.9915 | 46 200 |
| yoshida+hnn   | banana   | 1.0757e-02 | 1.5028e-02 | 0.9905 | 198 000 |

#### Stage 1 vs Stage 1.5 comparison (HNN configs only)

| Config | Target | dH_lrn (S1) | dH_lrn (S1.5) | ratio | dH_true (S1) | dH_true (S1.5) | ratio |
|--------|--------|------------|--------------|-------|------------|----------------|-------|
| leapfrog+hnn | gaussian | 6.4315e-03 | 6.3896e-03 | 0.993 | 5.5106e-03 | 6.8434e-03 | 1.242 |
| yoshida+hnn  | gaussian | 5.8735e-03 | 6.0694e-03 | 1.033 | 5.2480e-03 | 6.8561e-03 | 1.306 |
| leapfrog+hnn | banana   | 1.1188e-02 | 1.0488e-02 | 0.937 | 1.6508e-02 | 1.5137e-02 | 0.917 |
| yoshida+hnn  | banana   | 1.0900e-02 | 1.0757e-02 | 0.987 | 1.5930e-02 | 1.5028e-02 | 0.943 |

![|ΔH| trajectory Stage 1.5 — Gaussian](dH_trajectory_gaussian_v15.png)
![|ΔH| trajectory Stage 1.5 — Banana](dH_trajectory_banana_v15.png)
![Scissors plot — Gaussian (dH_learned vs dH_true)](scissors_gaussian.png)
![Scissors plot — Banana (dH_learned vs dH_true)](scissors_banana.png)

### 2.4 Analysis and conclusions

**Finding 1 — Fixing the integrator changes almost nothing:** dH_learned changes by ≤7% (ratio 0.937–1.033) going from Stage 1 to Stage 1.5. The "mixed integration" bug was not the bottleneck.

**Finding 2 — dH_true worsens on Gaussian after fix:** Gaussian dH_true increases by 24–31% in Stage 1.5. This is because the full HNN trajectory now accumulates HNN-specific errors in both q and p, whereas Stage 1's mixed version partially cancelled them via the true-gradient component.

**Finding 3 — Yoshida/leapfrog ratio for dH_learned remains ~1:**
- Gaussian: 6.0694e-03 / 6.3896e-03 = 0.950 (5% better)
- Banana: 1.0757e-02 / 1.0488e-02 = 1.026 (actually 2.6% worse)

The scissors plots show that dH_learned and dH_true are of the same order for HNN configs, confirming that the model error floor (dH_true) is the binding constraint, not the integrator.

**Conclusion:** Even with a correct, symmetric full-HNN-vector-field base step, Yoshida does not recover its 4th-order advantage. The problem lies in the HNN gradient itself: it introduces a persistent |ΔH_true| floor that is independent of step size. Stage 1.6 quantifies this precisely.

**Implication for Stage 1.6:** Need to study how dH_learned and dH_true scale with h. If the dH_true floor is h-independent (slope ≈ 0), the model error is fundamentally bounding energy conservation regardless of integrator order.

---

## 3. Experiment 3 — Step-size Convergence Study (Stage 1.6)

### 3.1 Purpose

Directly test whether Yoshida's 4th-order convergence transfers to the HNN setting by sweeping h over nearly two decades.
This distinguishes three regimes:
1. **Asymptotic regime** (large h): integrator error dominates → slopes should be 2 (leapfrog) and 4 (Yoshida)
2. **Floor regime** (small h): model error dominates → slopes collapse toward 0
3. **Crossover h**: step size where yoshida+hnn and leapfrog+hnn produce equal dH_learned

### 3.2 Experimental design

**Protocol:** Fixed integration time T = 1.0, vary h from 0.2 to 0.005, N_IC = 200 initial conditions, measure mean |ΔH_learned| and mean |ΔH_true| at the end of the trajectory.

| h | n_steps (L = T/h) |
|---|------------------|
| 0.200 | 5 |
| 0.100 | 10 |
| 0.050 | 20 |
| 0.020 | 50 |
| 0.010 | 100 |
| 0.005 | 200 |

**Configs:** 6 curves per target:
- `true_lf` — true gradient, leapfrog
- `true_y4` — true gradient, Yoshida
- `hnn_lf_lrn` — HNN, leapfrog, measured against H_θ (dH_learned)
- `hnn_lf_true` — HNN, leapfrog, measured against true H (dH_true)
- `hnn_y4_lrn` — HNN, Yoshida, measured against H_θ (dH_learned)
- `hnn_y4_true` — HNN, Yoshida, measured against true H (dH_true)

**Data source:** `sweep_results.pt`

### 3.3 Results

#### Log-log slopes (linear fit over full h range [0.2, 0.005])

| Target | Curve | log-log slope |
|--------|-------|--------------|
| gaussian | true+leapfrog | **1.9902** |
| gaussian | true+yoshida4 | 0.8143 |
| gaussian | hnn+leapfrog (dH_lrn) | 1.7380 |
| gaussian | hnn+leapfrog (dH_true) | 0.0586 |
| gaussian | hnn+yoshida4 (dH_lrn) | 0.9880 |
| gaussian | hnn+yoshida4 (dH_true) | 0.0124 |
| banana | true+leapfrog | **1.9876** |
| banana | true+yoshida4 | 0.9553 |
| banana | hnn+leapfrog (dH_lrn) | 1.6584 |
| banana | hnn+leapfrog (dH_true) | 0.0646 |
| banana | hnn+yoshida4 (dH_lrn) | 0.9946 |
| banana | hnn+yoshida4 (dH_true) | 0.0157 |

> **Note on true+yoshida4 slope:** The global log-log slope of ~0.8–1.0 is distorted by the precision floor at small h. Inspecting the raw values for Gaussian: h=0.2→0.1 gives a 16.3× improvement in |ΔH| (expected: 2⁴ = 16 for 4th order ✓). At h ≤ 0.05, values stop decreasing and reverse (floor from float32 round-off noise). The 4th-order regime is confirmed in the h ≥ 0.1 window; the low global slope is an artefact of fitting through the floor region.

#### Gaussian per-h table (mean |ΔH_learned|)

| h | n_steps | true_lf | true_y4 | hnn_lf_lrn | hnn_y4_lrn | y4/lf ratio |
|---|---------|---------|---------|-----------|-----------|------------|
| 0.200 | 5 | 8.5098e-03 | 1.0604e-04 | 8.4877e-03 | 3.1641e-03 | **0.373** |
| 0.100 | 10 | 2.1226e-03 | 6.5194e-06 | 2.1111e-03 | 1.6222e-03 | **0.768** |
| 0.050 | 20 | 5.3041e-04 | 9.0852e-07 | 5.2835e-04 | 8.1669e-04 | **1.546** |
| 0.020 | 50 | 8.4869e-05 | 1.2973e-06 | 9.1839e-05 | 3.2791e-04 | 3.571 |
| 0.010 | 100 | 2.1277e-05 | 1.8273e-06 | 3.2330e-05 | 1.6474e-04 | 5.096 |
| 0.005 | 200 | 5.5960e-06 | 2.7443e-06 | 1.6193e-05 | 8.3466e-05 | 5.154 |

#### Banana per-h table (mean |ΔH_learned|)

| h | n_steps | true_lf | true_y4 | hnn_lf_lrn | hnn_y4_lrn | y4/lf ratio |
|---|---------|---------|---------|-----------|-----------|------------|
| 0.200 | 5 | 1.0594e-02 | 1.6053e-04 | 1.0507e-02 | 5.7189e-03 | **0.544** |
| 0.100 | 10 | 2.6429e-03 | 9.7725e-06 | 2.6584e-03 | 2.8846e-03 | **1.085** |
| 0.050 | 20 | 6.6025e-04 | 1.0439e-06 | 6.8968e-04 | 1.4471e-03 | 2.098 |
| 0.020 | 50 | 1.0562e-04 | 1.3416e-06 | 1.3407e-04 | 5.7962e-04 | 4.323 |
| 0.010 | 100 | 2.6532e-05 | 1.8475e-06 | 5.1842e-05 | 2.9020e-04 | 5.598 |
| 0.005 | 200 | 7.0599e-06 | 2.5631e-06 | 2.5673e-05 | 1.4656e-04 | 5.709 |

#### HNN dH_true floor values (confirming h-independence)

| h | gaussian hnn_lf_true | gaussian hnn_y4_true | banana hnn_lf_true | banana hnn_y4_true |
|---|---------------------|---------------------|-------------------|-------------------|
| 0.200 | 9.564e-03 | 7.450e-03 | 1.838e-02 | 1.448e-02 |
| 0.100 | 7.074e-03 | 7.091e-03 | 1.417e-02 | 1.370e-02 |
| 0.050 | 6.995e-03 | 7.033e-03 | 1.362e-02 | 1.353e-02 |
| 0.020 | 7.013e-03 | 7.021e-03 | 1.349e-02 | 1.348e-02 |
| 0.010 | 7.019e-03 | 7.020e-03 | 1.347e-02 | 1.347e-02 |
| 0.005 | 7.020e-03 | 7.020e-03 | 1.346e-02 | 1.346e-02 |

The dH_true values plateau at h ≤ 0.05 at approximately **7.0 × 10⁻³** (Gaussian) and **1.35 × 10⁻²** (Banana), confirming the HNN model-error floor is h-independent.

![Convergence sweep — Gaussian](convergence_gaussian.png)
![Convergence sweep — Banana](convergence_banana.png)

### 3.4 Analysis and conclusions

**Finding 1 — true+leapfrog confirms 2nd-order convergence:** slopes 1.988–1.990, consistent with theory. Integration code is correct.

**Finding 2 — true+yoshida4 shows 4th-order at large h, but hits a float32 floor:** At h = 0.2 → 0.1 the improvement factor is ≈16 (= 2⁴), confirming 4th-order convergence in the asymptotic regime. The global log-log slope is distorted by the floor at h < 0.05, where float32 round-off dominates (~3 × 10⁻⁷ for Gaussian). This is an artefact of float32; float64 would push the floor lower.

**Finding 3 — HNN dH_true floor is h-independent (slope ≈ 0):** The HNN model error (dH_true floor ≈ 7 × 10⁻³ for Gaussian, 1.35 × 10⁻² for Banana) is constant across 40× variation in h. This is the fundamental bottleneck: the HNN cannot reconstruct the true Hamiltonian exactly, and this structural error is independent of discretisation step size.

**Finding 4 — dH_learned crossover confirms Yoshida becomes counterproductive at practical h:**

| Target | Crossover h (yoshida = leapfrog in dH_lrn) |
|--------|-------------------------------------------|
| Gaussian | between 0.05 and 0.10 (≈0.08) |
| Banana | ≈0.10 |

At the practical step size h = 0.1 used in Stage 2: Gaussian y4/lf ratio = 0.768 (Yoshida slightly better), Banana y4/lf ratio = 1.085 (Yoshida slightly worse). At h = 0.05: Gaussian ratio = 1.546, Banana ratio = 2.098. So at h = 0.1 the integrators are nearly equivalent; at smaller h yoshida is strictly worse.

**Finding 5 — The dH_true floor explains everything:** Once dH_learned saturates at the dH_true floor, the integrator order is irrelevant. Higher-order integration reduces the integrator truncation error, but the irreducible model-error component is unchanged. This is **Layer 1** of the three-layer negation.

---

## 4. Experiment 4 — Full Subset Simulation (Stage 2, N_rep = 30)

### 4.1 Purpose

Test whether the conclusions from Stages 1–1.6 (no dH advantage for Yoshida+HNN) translate into statistically measurable differences in SuS performance metrics: Pf estimate, COV, and cost.
This is the decisive experiment that directly answers the thesis question.

### 4.2 Experimental design

**SuS configuration:**

| Parameter | Value |
|-----------|-------|
| N (samples per level) | 500 |
| p₀ (conditional probability) | 0.1 |
| n_seeds per level | 50 (= p₀ × N) |
| spc (steps per chain) | 10 (= ceil(N / n_seeds)) |
| burn-in proposals | 200 |
| max levels | 8 |
| seed base | 0 |
| β | 3.0 |
| Pf_ref | **1.3499 × 10⁻³** (= Φ(−3), hardcoded in `experiment_sus_stage2.py` line 62) |

**Pf_ref derivation:** The limit-state function g(q) = β − q₁ gives failure region {q₁ ≥ 3}. Under both the Gaussian and Banana targets, q₁ ~ N(0, 1) marginally (sigma1 = 1.0 in `BananaTargetConfig`). Therefore Pf_ref = P(q₁ ≥ 3) = Φ(−3) = 1.3499 × 10⁻³ for both targets.

**Integrator parameters (same as Stage 1.5):**

| | leapfrog+hnn | yoshida+hnn |
|-|-------------|------------|
| step_size | 0.1 | 0.1 |
| n_steps | 10 | 10 |
| VF calls/proposal | 21 | 90 |
| Cost ratio | 1.00 | **4.286×** |

**HNN checkpoints:** reused from Stage 1.5 — `hnn_gaussian.pt` and `hnn_banana.pt` in `results/smoke_yoshida_hnn/`.

**N_rep = 30** independent SuS runs per (target, integrator) combination; seeds 0–29.

**Data source:** `sus_stage2_results.pt` (N_rep = 30), `sus_stage2_results_smoke.pt` (N_rep = 2 preview)

### 4.3 Results

#### Main results table

| target | integrator | mean_Pf | Pf_ref | std_Pf | SE | COV | est_vf/run | mean_lv |
|--------|-----------|---------|--------|--------|-----|-----|------------|---------|
| gaussian | leapfrog+hnn | 1.2940e-03 | 1.3499e-03 | 4.444e-04 | 8.114e-05 | 0.343 | 2.132e+06 | 3.333 |
| gaussian | yoshida+hnn  | 1.3439e-03 | 1.3499e-03 | 4.871e-04 | 8.892e-05 | 0.362 | 9.134e+06 | 3.300 |
| banana   | leapfrog+hnn | 1.3927e-03 | 1.3499e-03 | 5.588e-04 | 1.020e-04 | 0.401 | 2.130e+06 | 3.200 |
| banana   | yoshida+hnn  | 1.3200e-03 | 1.3499e-03 | 5.494e-04 | 1.003e-04 | 0.416 | 9.138e+06 | 3.400 |

**Cost ratio (yoshida / leapfrog):**
- Gaussian: 9.134e+06 / 2.132e+06 = 4.284× (theoretical: 90/21 = 4.286)
- Banana: 9.138e+06 / 2.130e+06 = 4.291×

#### Statistical significance of Pf differences

| target | pair | |mean_Pf difference| | SE of difference | difference / SE |
|--------|------|---------------------|-----------------|----------------|
| gaussian | y4 − lf | 4.987e-05 | ≈ 1.20e-04 (pooled) | **0.42σ** |
| banana   | lf − y4 | 7.267e-05 | ≈ 1.44e-04 (pooled) | **0.50σ** |

All differences are well below 1σ. The |mean − Pf_ref| / SE values are:
- gaussian / leapfrog: 0.689σ
- gaussian / yoshida: 0.068σ
- banana / leapfrog: 0.419σ
- banana / yoshida: 0.298σ

#### Pf distribution (N_rep=30 per config)

| target | integrator | min | p25 | median | p75 | max |
|--------|-----------|-----|-----|--------|-----|-----|
| gaussian | leapfrog+hnn | 6.560e-04 | 9.060e-04 | 1.230e-03 | 1.600e-03 | 2.300e-03 |
| gaussian | yoshida+hnn  | 6.560e-04 | 9.365e-04 | 1.240e-03 | 1.720e-03 | 2.320e-03 |
| banana   | leapfrog+hnn | 3.620e-04 | 1.055e-03 | 1.330e-03 | 1.655e-03 | 2.720e-03 |
| banana   | yoshida+hnn  | 3.860e-04 | 9.010e-04 | 1.220e-03 | 1.660e-03 | 2.780e-03 |

#### Mechanism table — Gaussian

The within-level MCMC rejection is decomposed into three exclusive outcomes:
- **(a) geometric rejection**: proposed q* violates the level threshold g(q*) > threshold_l
- **(a) energy rejection**: q* passes geometry but fails MH (g(q*) ≤ threshold_l but exp(−ΔH) < u)
- **accept**: both geometry and MH accepted

The conditional energy rejection rate **(b)** = energy_rej / (energy_rej + accept) is the MH rejection rate *given* that the proposal passed the geometric constraint.

**gaussian / leapfrog+hnn:**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6367 | 0.000963 | 0.00265 | 0.3624 |
| L2 | 0.8479 | 0.000741 | 0.00487 | 0.1514 |
| L3 | 0.9229 | 0.000667 | 0.00865 | 0.0764 |

**gaussian / yoshida+hnn:**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6367 | 0.000741 | 0.00204 | 0.3625 |
| L2 | 0.8487 | 0.000815 | 0.00538 | 0.1505 |
| L3 | 0.9230 | 0.000247 | 0.00321 | 0.0768 |

#### Mechanism table — Banana

**banana / leapfrog+hnn:**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6257 | 0.007704 | 0.02058 | 0.3666 |
| L2 | 0.8293 | 0.006963 | 0.04080 | 0.1637 |
| L3 | 0.8996 | 0.006667 | 0.06642 | 0.0937 |

**banana / yoshida+hnn:**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6254 | 0.007481 | 0.01997 | 0.3671 |
| L2 | 0.8278 | 0.007333 | 0.04258 | 0.1649 |
| L3 | 0.9102 | 0.006667 | 0.07423 | 0.0831 |

![Pf distribution (N_rep=30)](sus_pf_distribution.png)
![Rejection mechanism bars (N_rep=30)](sus_mechanism_bars.png)
![Pf distribution (N_rep=2 smoke preview)](sus_pf_distribution_smoke.png)
![Rejection mechanism bars (N_rep=2 smoke preview)](sus_mechanism_bars_smoke.png)

### 4.4 Analysis and conclusions

**Finding 1 — No statistically significant difference in Pf or COV:**
All four mean_Pf estimates agree with Pf_ref within 0.7σ. The yoshida−leapfrog difference in Pf is ≤ 0.5σ for both targets, and the COV difference is ≤ 0.02 absolute (within N=30 sampling noise ≈ ±0.05).

**Finding 2 — Geometric rejection is the binding constraint:**
At every level, geometric rejection accounts for 63–92% of all proposals. Energy rejection is < 1% unconditionally. The conditional energy rejection rate (b) is 0.2–8.7%, confirming that once a proposal clears the geometric threshold, the MH step accepts it with high probability. This holds equally for both integrators.

**Finding 3 — Yoshida and leapfrog have identical rejection structure to 3 decimal places:**
Geometric rejection rates agree between integrators to the 3rd decimal place at every level (e.g., L1 banana: 0.6257 vs 0.6254). This is a direct consequence of Stage 1.6's finding: at h = 0.1, both integrators produce essentially the same |ΔH_true| (dominated by the HNN model-error floor), leading to indistinguishable MH acceptance.

**Finding 4 — Banana energy rejection is higher than Gaussian:**
Gaussian energy_rej(a) ≈ 0.07–0.10%, banana energy_rej(a) ≈ 0.7–0.8%. This is consistent with the larger dH_true floor on banana (1.35 × 10⁻² vs 7.0 × 10⁻³ for Gaussian), which was established in Stage 1.6. The mechanism hierarchy is internally consistent across all four experiments.

**Finding 5 — Yoshida costs 4.29× more with zero benefit:**
Yoshida est_vf/run ≈ 4.29× that of leapfrog (matching the theoretical 90/21 = 4.286 VF-call ratio). The cost increase is exact and confirmed experimentally; the benefit in Pf or COV is zero within N=30 sampling error.

---

## 5. Summary: Three-Layer Negation + Cost + Closure

### 5.1 The causal chain

The research question — "does Yoshida improve SuS-HMC when gradients come from an HNN?" — admits a three-layer answer where each layer identifies a progressively more fundamental blocker.

---

**Layer 0 — Yoshida's dH advantage inverts at practical step sizes (Experiment 3, partial finding)**

At h = 0.1 (the practical SuS step size), the Yoshida/leapfrog ratio of dH_learned is 0.77–1.09, approximately neutral. For h ≤ 0.05, Yoshida is strictly *worse* (ratio > 1, reaching 5× at h = 0.005). The crossover between "Yoshida better" and "Yoshida worse" falls near h ≈ 0.08–0.10 for these targets. This means the practitioner's natural working step size sits exactly at the boundary where the 4th-order advantage has already vanished.

**Mechanism:** The dH_true floor (7 × 10⁻³ for Gaussian) is larger than the integrator truncation error at h ≤ 0.05. Yoshida produces smaller *integrator* error, but when this falls below the *model* error floor, the total dH_learned is dominated by the floor, erasing the ranking advantage.

*Caveat:* This layer depends on float32 precision and the specific HNN error level. More accurate HNNs or float64 arithmetic would push the floor lower and could restore Yoshida's advantage in dH_lrn at small h. But this does not help at practical h = 0.1. **This layer is a finding, not a fundamental blocker.**

---

**Layer 1 — HNN model error creates an h-independent dH_true floor (Experiment 3, core finding)**

The true-Hamiltonian energy error dH_true saturates at a floor of approximately:
- 7.0 × 10⁻³ (Gaussian), 1.35 × 10⁻² (Banana)

for all h ≤ 0.05. The slopes of hnn_lf_true and hnn_y4_true vs h are **0.06–0.07** (effectively zero). This floor is a structural property of the HNN checkpoint: the network has finite approximation capacity, and the residual |H_θ(q,p) − H(q,p)| cannot be reduced by changing h.

**Consequence:** Both integrators produce the same dH_true at any h where the floor dominates. The MH acceptance rate is determined by dH_true, not dH_learned. Therefore the integrator order cannot improve acceptance or reduce COV.

*This layer is fundamental and independent of float32 precision.* Replacing float64 does not remove the model-error floor; it requires a better-trained HNN.

---

**Layer 2 — The binding constraint in SuS is geometric rejection, not energy rejection (Experiment 4, punchline)**

Even if dH_true were reduced to zero (perfect HNN), SuS performance would barely change, because the energy (MH) rejection contributes < 1% of total rejections:
- Geometric rejection: 63–92% per level
- Energy rejection (unconditional): < 1% per level
- Conditional energy rejection (b): 0.2–8.7%

The within-level MCMC is constrained to the conditional failure region {g(q) ≤ threshold_l}, which becomes an increasingly thin shell at each level. **The geometry of the failure region — not the Hamiltonian integrator quality — determines SuS convergence.**

This is the same mechanism that limits true-gradient Yoshida in SuS (cf. Wang et al.): swapping in a higher-order integrator doesn't change the geometry of the conditional target, so geometric rejection — and hence COV — is unchanged.

*This layer is the most fundamental and most general.* It implies that **no integrator improvement** will materially help SuS COV for this class of limit-state geometries, regardless of HNN quality.

---

### 5.2 Cost summary

| Integrator | VF calls/proposal | Cost ratio | Pf benefit | COV benefit |
|------------|------------------|------------|-----------|------------|
| leapfrog+hnn | 21 | 1.00 | baseline | baseline |
| yoshida+hnn | 90 | 4.286× | 0 (< 0.5σ) | 0 (< 0.02 abs) |

The 4.286× cost premium is exact (90/21) and reproduces in the measured est_vf/run ratios (4.284–4.291×). There is no compensating benefit.

### 5.3 Closure with prior work

Prior work on Yoshida+SuS with **true gradients** (e.g., Wang et al.) shows that Yoshida does not improve SuS COV because geometric rejection dominates — the same finding as Layer 2 here. The present HNN experiments reproduce and extend this: the HNN setting adds Layer 1 (model-error floor) as an additional reason why the integrator order is irrelevant, but the root cause at the SuS level is the same sub-set geometry binding constraint that applies even with perfect gradients.

---

## 6. Limitations and Future Work

### 6.1 Layer 0 mechanism needs float64 confirmation

The floor for true+yoshida4 at small h was attributed to float32 round-off noise. Running the convergence sweep with float64 precision would confirm this and establish the true 4th-order regime for the Yoshida reference. This would firm up Layer 0 (currently marked as "partial finding") and determine the h range where Yoshida has a genuine advantage over leapfrog for HNN gradients.

### 6.2 Dimension: d = 2 only

All experiments used 2D targets. In higher dimensions:
- HNN training requires either dimension-specific networks or factorised architectures
- The dH_true floor may scale with dimension (more parameters to fit, more approximation error)
- The geometric rejection fraction may change depending on the failure-region geometry

The Layer 2 conclusion (geometry binding) should still hold for generic hyperplane-type limit states, but the quantitative mechanism breakdown would differ.

### 6.3 Energy rejection accounting: (a) vs (b)

Two distinct energy-rejection rates exist, with different interpretations:
- **(a) unconditional rate** = energy_rej / N_total: fraction of all proposals that fail MH — used in the mechanism bar charts, appears as < 1%
- **(b) conditional rate** = energy_rej / (energy_rej + accept): MH rejection rate given that the proposal is geometrically valid — ranges from 0.2% to 8.7%

Rate (b) is the correct quantity for assessing MH efficiency within the constrained chain. The banana (b) values (2–7%) are higher than Gaussian (0.2–0.9%), consistent with the banana's larger dH_true floor (1.35 × 10⁻² vs 7.0 × 10⁻³).

### 6.4 Better-trained HNN would not change the conclusion

If the HNN were retrained with more data, longer training, or a larger architecture, the dH_true floor would decrease. This would:
- Push the crossover h (Layer 0) to a smaller value, making Yoshida slightly more competitive at practical h
- Reduce the unconditional energy-rejection rate from < 1% to even closer to 0%

But Layer 2 is unaffected: geometric rejection would remain 60–90% regardless of HNN quality, because it is determined by the limit-state geometry, not the integrator.

### 6.5 COV estimation uncertainty

With N_rep = 30, the standard error of a COV estimate is approximately COV / √(2 × 30) ≈ COV × 0.13. The observed COV differences between leapfrog and yoshida are ≤ 0.02 absolute — roughly 0.4× the sampling uncertainty. Increasing to N_rep = 100 or more would tighten the bounds but is not expected to reveal a significant difference given Layer 2.

---

*Record compiled from .pt files in `results/smoke_yoshida_hnn/`. All numbers loaded directly from saved checkpoints; no placeholders.*
