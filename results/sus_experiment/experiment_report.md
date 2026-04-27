# Subset Simulation Experiment: Baseline HMC vs PINN-HMC vs HNN-HMC

## 1. Background

Structural reliability analysis requires estimating failure probabilities
$P_F = P(g(\mathbf{X}) \leq 0)$ that can be extremely small ($10^{-5}$–$10^{-10}$).
Direct Monte Carlo is unaffordable at these scales; **Subset Simulation** (SuS,
Au & Beck 2001; Wang et al. 2019) decomposes the rare event into a product of
more probable conditional events, each estimated by a Markov chain.

The quality of the Markov-chain transitions drives both the accuracy and the
computational cost of SuS. This experiment benchmarks three transition kernels
on a non-Gaussian banana-shaped target with an elliptical failure domain,
extending our earlier plain-sampling comparison to the reliability setting:

- **Baseline HMC** — leapfrog with true gradients $\nabla_q U$. Accurate but
  requires one gradient evaluation per leapfrog step.
- **PINN-HMC** — a Physics-Informed Neural Network trained to predict the
  trajectory endpoint $(q_T, p_T)$ given $(q_0, p_0, T)$, replacing the
  entire leapfrog integrator with a single forward pass.
- **HNN-HMC** — a Hamiltonian Neural Network that learns $H_\theta(q,p)$;
  leapfrog is run with the learned vector field instead of true gradients.

Both surrogates trade upfront training cost for cheaper per-proposal
evaluation. Whether this trade-off is favourable in the SuS setting is the
central question of this experiment.

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| Target | `WangBananaTarget` (Wang 2019 eq. 34): $a=1.15$, $b=0.5$, $\rho=0.9$ |
| Limit-state | `EllipticalLimitState` (Wang 2019 eq. 40): $c_1=1$, $c_2=0.5$, $\theta=\pi/4$ |
| Radii $r$ | 10, 12 |
| Model arch. | 4-layer MLP, 128 hidden units, Tanh (both PINN and HNN) |
| PINN training | 120 epochs, Adam $10^{-3}$, ProgressBalancer $\alpha=1$ |
| HNN training | 120 epochs, Adam $10^{-3}$ |
| Dataset | 1800 trajectories, step size 0.08, 20 leapfrog steps |
| SuS $N$ | 500 samples/level |
| SuS $p_0$ | 0.1 |
| SuS max levels | 20 |
| SuS burn-in | 200 |

## 3. Results

| Method | r | P_F | Levels | Train (s) | Prop (ms) | Samp (s) |
|--------|---|-----|--------|-----------|-----------|----------|
| Baseline HMC | 10 | 3.86e-02 | 2 | 0.00 | 27.5 | 5.93 |
| PINN-HMC | 10 | 3.28e-03 | 3 | 243.80 | 1.4 | 0.31 |
| HNN-HMC | 10 | 3.04e-02 | 2 | 161.75 | 61.3 | 12.82 |
| Baseline HMC | 12 | 2.42e-02 | 2 | 0.00 | 25.5 | 5.34 |
| PINN-HMC | 12 | 8.12e-07 | 7 | 243.80 | 1.3 | 0.36 |
| HNN-HMC | 12 | 1.38e-02 | 2 | 161.75 | 60.7 | 12.70 |

## 4. Analysis

### 4.1 Sampling Quality

Baseline HMC and HNN-HMC produce broadly consistent $P_F$ estimates at both
radii, converging in 2 levels each. PINN-HMC, by contrast, severely
underestimates $P_F$ at $r=12$ (8.12e-07 vs 2.42e-02 for baseline) and
requires 7 levels — a clear diagnostic of sampling collapse. The anomalous
level count indicates that PINN proposals lack spatial diversity: samples
cluster in a narrow band around $x \approx 2$, so each SuS level can only
advance the threshold by a small increment rather than making a large jump
toward the failure boundary. The resulting $\hat{P}_F \approx p_0^7 \times
(N_\text{fail}/N)$ is an artefact of the level count, not a reliable estimate
of the true failure probability.

The root cause is a distribution shift problem. PINN learns trajectories
point-by-point from training data concentrated in the high-probability bulk
of the target. As SuS progresses, the conditional distribution $p_k$ drifts
toward the rare-event tail, far outside the training distribution $p_\text{train}$.
PINN's approximation error grows in these out-of-distribution regions,
producing large $|\Delta H|$, low MH acceptance, and ultimately a collapsed
chain. This can be described by the following degradation chain:

$$\mathrm{TV}(p_k,\, p_\text{train}) \uparrow
\;\Rightarrow\; \varepsilon_\text{PINN} \uparrow
\;\Rightarrow\; |\Delta H| \uparrow
\;\Rightarrow\; \alpha_\text{MH} \downarrow
\;\Rightarrow\; \text{Spectral Gap} \downarrow
\;\Rightarrow\; \tau_\text{mix} \uparrow
\;\Rightarrow\; \mathrm{ESS} \downarrow
\;\Rightarrow\; \mathrm{Var}(\hat{P}_F) \uparrow$$

HNN avoids this collapse because it learns a local vector field
$(\partial H_\theta/\partial p,\, -\partial H_\theta/\partial q)$ rather than
a global trajectory. A local approximation error at one point does not
corrupt the entire proposal, making HNN structurally more robust to the
distribution shift introduced by deeper SuS levels.

### 4.2 Computational Cost

PINN achieves a 20× per-proposal speedup (1.4 ms vs 27.5 ms for baseline),
which is the theoretical motivation for replacing leapfrog with a neural
surrogate. However, this speedup is only useful if proposal quality is
maintained — which it is not in the deeper SuS levels. A fast but incorrect
sampler is strictly worse than a slower correct one.

HNN is paradoxically 2× slower per proposal than Baseline HMC (61.3 ms vs
27.5 ms). Backpropagating through $H_\theta$ at every leapfrog step costs
more than evaluating the true analytical gradient directly. The upfront
training cost (162 s) therefore provides no runtime return in this setting.
HNN's cost advantage would only materialise when the true gradient is
expensive — for example when the likelihood requires solving an ODE, as in
Bayesian parameter inference for the Bouc-Wen oscillator.

In summary, neither surrogate improves upon Baseline HMC here. This connects
directly to our earlier banana-target experiment: both surrogates performed
well under plain HMC sampling, where proposals stay within the training
distribution. SuS breaks this assumption by design — each deeper level
pushes the sampler further into out-of-distribution territory, exposing
PINN's fundamental limitation and erasing HNN's potential speed advantage.

> **Core finding:** good trajectory approximation (PINN) $\neq$ good HMC
> proposal quality; good Hamiltonian structure learning (HNN) $\neq$
> computational efficiency. For SuS, proposal diversity and structural
> correctness matter more than raw proposal speed.

*Note: the TV distance → Spectral Gap argument above combines established
MCMC concepts (mixing time, spectral gap, ESS) into a motivated theoretical
argument supported by empirical evidence. It is not a formally proven theorem.
Supporting literature should be sought in "robustness of MCMC under model
misspecification" and "covariate shift in MCMC".*

## 5. Limitations and Future Work

- **Distribution shift in rare-event tails**: PINN and HNN are trained on
  trajectory data from the prior. Deeper SuS levels condition on increasingly
  extreme samples; the learned surrogate degrades there. Level-adaptive
  retraining (RS-HMC-SS, Wang 2019) is the natural remedy.
- **Single training run**: models are trained once and reused across all SuS
  levels. Online retraining at each level would allow the surrogate to track
  the shifting conditional distribution.
- **No $P_F$ uncertainty quantification**: only one SuS run per method is
  reported. Multiple independent runs would yield confidence intervals and
  allow a fair variance comparison.
- **Small $N$**: $N=500$ gives a coefficient of variation of approximately
  $\sqrt{(1-p_0)/(p_0 N)} \approx 0.13$ per level; larger $N$ would reduce
  variance substantially.
- **Fixed hyperparameters**: step size and leapfrog steps were not tuned for
  `WangBananaTarget`; a tuning phase would likely improve all three methods.
- **2D only**: the experiment is limited to a 2-dimensional target. Scaling
  to higher dimensions is needed to assess whether the findings generalise.


- PINN performs well in regions covered by the training data. However, SuS specifically targets low-probability regions that PINN has rarely or never seen during training. In these out-of-distribution regions, PINN's trajectory predictions are inaccurate, leading to large energy errors ∣ΔH∣|\Delta H|
∣ΔH∣, which causes the Metropolis-Hastings step to reject almost every proposal. As a result, the Markov chain gets stuck — samples lose diversity and collapse into a narrow region. This is why PINN's PFP_F
PF​ estimate is unreliable in deeper SuS levels.

## 6. Proposed Next Step: Adaptive PINN-SuS

### Motivation

The core failure of PINN-HMC in this experiment is distribution shift: the 
surrogate is trained once on the prior and never updated, so its predictions 
degrade as SuS pushes samples into increasingly rare regions far outside the 
training distribution. The natural remedy is to retrain the PINN at each SuS 
level using the current conditional samples, keeping the surrogate 
in-distribution throughout the entire simulation.

### Proposed Method

At each SuS level $k$, after collecting the conditional samples 
$\{q_i : g(q_i) \leq \gamma_k\}$, fine-tune the PINN on trajectories 
generated from these samples as new initial conditions. The updated PINN is 
then used as the proposal kernel for the next level. This can be summarised 
as:

1. **Level 0**: train PINN on prior samples as before
2. **Level k > 0**: fine-tune PINN on trajectories starting from current 
   conditional samples $\{q_i : g(q_i) \leq \gamma_{k-1}\}$
3. Use fine-tuned PINN as proposal for level $k$
4. Repeat until failure domain is reached

This corresponds directly to the RS-HMC-SS framework of Wang (2019), but 
replaces the leapfrog integrator at each level with a level-adapted PINN 
surrogate.

### Expected Trade-off

Adaptive PINN introduces additional training cost at each level, but with 
two important mitigations:

- Fine-tuning from a pre-trained model is significantly cheaper than 
  training from scratch — likely 50–80s per level rather than 244s
- The per-proposal speedup (20×) is preserved within each level, since 
  the PINN remains in-distribution

The central research question becomes:

> Does the per-proposal speedup of Adaptive PINN outweigh the additional 
> fine-tuning cost, compared to Baseline HMC, while maintaining accurate 
> $P_F$ estimates?

This is an empirical question that the next experiment will answer. The 
answer is not obvious: if SuS requires many levels and fine-tuning is 
expensive, Adaptive PINN may be slower overall; if fine-tuning is fast and 
few levels are needed, it could outperform Baseline HMC in total cost while 
recovering correct $P_F$ estimates.

### Connection to Existing Work

This approach directly addresses the limitation identified in Section 4: 
PINN's in-distribution assumption is violated by SuS by design. By 
continuously adapting the surrogate to the current conditional distribution, 
Adaptive PINN-SuS maintains the structural advantage of trajectory learning 
(fast forward pass, no gradient evaluation at sampling time) while resolving 
the distribution shift problem that caused sampling collapse in the present 
experiment.
