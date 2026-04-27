# Subset Simulation Experiment: Baseline HMC vs PINN-HMC vs HNN-HMC

## 1. Background

Structural reliability analysis requires estimating failure probabilities
$P_F = P(g(\mathbf{X}) \leq 0)$ that can be extremely small ($10^{-5}$–$10^{-10}$).
Direct Monte Carlo is unaffordable at these scales; **Subset Simulation** (SuS,
Au & Beck 2001; Wang et al. 2019) decomposes the rare event into a product of
more probable conditional events, each estimated by a Markov chain.

The quality of the Markov-chain transitions drives both the accuracy and the
computational cost of SuS. This experiment benchmarks three transition kernels:

* **Baseline HMC** — leapfrog with true gradients $\nabla_q U$. Accurate but
  requires one gradient evaluation per leapfrog step ($L$ per proposal).
* **PINN-HMC** — a Physics-Informed Neural Network (PINN) trained to predict
  the trajectory endpoint $(q_T, p_T)$ given $(q_0, p_0, T)$, replacing the
  entire integrator with a single forward pass.  Upfront training cost is
  amortised over many SuS proposals.
* **HNN-HMC** — a Hamiltonian Neural Network (HNN) that learns the Hamiltonian
  $H_\theta(q, p)$; leapfrog is run with the *learned* vector field, again
  trading training cost for cheaper per-step evaluation.

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Target | `WangBananaTarget` (Wang 2019 eq. 34): $a=1.15$, $b=0.5$, $\rho=0.9$ |
| Limit-state | `EllipticalLimitState` (Wang 2019 eq. 40): $c_1=1$, $c_2=0.5$, $\theta=\pi/4$ |
| Radii $r$ | 10 |
| Model arch. | 4-layer MLP, 128 hidden, Tanh (both PINN and HNN) |
| PINN training | 120 epochs, Adam $10^{-3}$, ProgressBalancer $\alpha=1$ |
| HNN training | 120 epochs, Adam $10^{-3}$ |
| Dataset | 1 800 trajectories, step size 0.08, 20 leapfrog steps |
| SuS $N$ | 500 samples/level |
| SuS $p_0$ | 0.1 |
| SuS max levels | 20 |
| SuS burn-in | 200 |

## 3. Results

| Method | r | P_F | Levels | Train (s) | Prop (ms) | Samp (s) |
|--------|---|-----|--------|-----------|-----------|----------|
| Baseline HMC | 10 | 2.0000e-02 | 2 | 0.00 | 26.205 | 0.00 |
| PINN-HMC | 10 | 0.0000e+00 | 3 | 1.00 | 1.253 | 0.00 |
| HNN-HMC | 10 | 0.0000e+00 | 3 | 1.50 | 37.247 | 0.00 |

## 4. Analysis

[TO BE FILLED] — Discuss whether P_F estimates agree across methods.
Explain the trade-off between training time and per-proposal speed for PINN
and HNN.  Note which method delivers the best total-cost P_F estimate and
why.  Connect to the earlier banana-target HMC experiment: did PINN/HNN
generalise well to the rare-event tails explored by deeper SuS levels?

## 5. Limitations and Future Work

* **Distribution shift in rare-event tails**: PINN and HNN are trained on
  trajectory data from the prior.  Deeper SuS levels condition on ever more
  extreme samples; the learned model may degrade there.
* **Single training run**: models are trained once and reused across all SuS
  levels.  Level-adaptive retraining (RS-HMC-SS, Wang 2019) could improve
  accuracy at low $P_F$.
* **No $P_F$ uncertainty**: only one SuS run per method is reported.  Multiple
  independent runs would yield confidence intervals.
* **Small $N$**: $N = 500$ gives a coefficient of variation of
  $\approx \sqrt{(1-p_0)/(p_0 N)} \approx 0.13$
  per level; larger $N$ would reduce variance.
* **Fixed hyper-parameters**: step size and number of leapfrog steps were not
  tuned for the WangBananaTarget; a tuning phase would likely improve all
  three methods.
