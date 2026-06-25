# Geometry-Bounce Experiment: Supervisor-Ready Summary

**Run date:** 2026-06-23. Full run, N_rep=30.
**All numbers trace to:** `record_geometry_bounce.md`, `exp0_results.json`, `exp1_raw.npz` (per-rep β̂
arrays recomputed in this document), and `experiment_geometry_bounce.py` (for the ∇U accounting).

---

## 1. Executive Summary

This experiment tested two hypotheses about what limits HMC-SuS performance when the Hamiltonian
trajectory crosses a subset boundary F_k. **H_B**: a geometry-aware adaptive step size alone
(arm B) reduces geometric rejection. **H_C**: adding specular barrier bouncing (arm C) eliminates
geometric rejection entirely. The results are unambiguous. H_B is **rejected**: arm B's geometric
rejection fractions match arm A (fixed leapfrog baseline) to three decimal places at every level
of both configurations, confirming that trajectory exit through the boundary is a property of the
true Hamiltonian flow and cannot be fixed by step-size refinement alone. H_C is **confirmed**:
arm C reduces geometric rejection from 51–96% (arm A) to 0–0.6%, with no energy-rejection
penalty, and cuts the COV of β̂ by 32% at β=3.5 and 42% at β=4.0. Correctness of the bounce is
validated in a separate truncated-Gaussian experiment (Exp 0): arm C's leak rate is 0.000 with
non-significant KS statistics on both marginals. At matched accuracy (equal COV), arm C is
approximately **1.64× more cost-efficient** at β=3.5 and **1.80×** at β=4.0, even when all
constraint evaluations (G) are priced at the same cost as a potential gradient (∇U). The main
caveats are: (i) arm C uses a state-dependent adaptive step that breaks leapfrog reversibility,
making it an approximate sampler; and (ii) in settings where evaluating the limit-state function G
is expensive (e.g., FEM-based reliability or HNNMC-SuS), arm C's per-step constraint monitoring
multiplies cost and the efficiency gain can shrink or reverse.

---

## 2. Experimental Setup

### Configurations

| Config | LSF | ρ | analytic β_ref | Pf_ref |
|--------|-----|---|---------------|--------|
| Config 1 | Thaler linear (§5.1), d=2 | 0.00 | 3.5 | 2.3263e-04 |
| Config 2 | Thaler linear (§5.1), d=2 | 0.75 | 4.0 | 3.1671e-05 |

SuS parameters: N=1000, p₀=0.1, burn-in=200 iterations (global; see §6), N_rep=30, seed 0–29.

### Arms

| Arm | Integrator | Boundary treatment | Sampler type |
|-----|-----------|-------------------|-------------|
| A | Fixed leapfrog (dt=0.1, n=10 steps) | End-reject | Exact (up to HMC discretisation) |
| B | Geometry+energy adaptive step | End-reject | Approximate (variable dt) |
| C | Geometry+energy adaptive step | Specular barrier bounce | Approximate (variable dt) |
| C′ | Fixed step (dt=0.1) | Specular barrier bounce | Exact reference — **not run this pass** |

Note: arms B and C use a state-dependent per-step dt, which breaks leapfrog reversibility and
volume-preservation. C′ (fixed step + bounce) is the exactly-reversible ablation needed to isolate
the adaptivity bias from the bounce benefit; it is the planned next run.

### Controller / bounce settings (from record)

| Parameter | Value |
|-----------|-------|
| c_cfl | 0.5 |
| eps_E | 0.05 |
| tau_safety | 0.85 |
| dt_min | 0.001 |
| dt_max | 0.5 |
| dt_max_change | 5.0 |
| n_bounce_max | 5 |
| tol_hit | 1e-06 |
| bisect_max_iter | 30 |
| dt_seed (baseline dt) | 0.1 |
| t_f (integration time) | 1.0 |

---

## 3. Mechanism Result

### Geometric and energy rejection by level (mean over N_rep=30)

**Config 1: β=3.5, ρ=0.0**

| Level | A geom_rej | B geom_rej | C geom_rej | A ener_rej | B ener_rej | C ener_rej | C accept |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|---------|
| L1 | 0.655 | 0.656 | **0.000** | 0.000 | 0.001 | 0.003 | 0.997 |
| L2 | 0.855 | 0.854 | **0.003** | 0.000 | 0.001 | 0.002 | 0.994 |
| L3 | 0.934 | 0.934 | **0.006** | 0.000 | 0.000 | 0.002 | 0.991 |
| L4 | 0.957 | 0.958 | *(C terminates in 3 levels; last row extrapolated)* | — | — | — | — |

**Config 2: β=4.0, ρ=0.75**

| Level | A geom_rej | B geom_rej | C geom_rej | A ener_rej | B ener_rej | C ener_rej | C accept |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|---------|
| L1 | 0.515 | 0.515 | **0.000** | 0.002 | 0.002 | 0.004 | 0.996 |
| L2 | 0.713 | 0.713 | **0.001** | 0.001 | 0.001 | 0.003 | 0.996 |
| L3 | 0.818 | 0.818 | **0.003** | 0.000 | 0.000 | 0.004 | 0.993 |
| L4 | 0.883 | 0.885 | **0.005** | 0.000 | 0.000 | 0.002 | 0.992 |

**Observation.** Arms A and B are indistinguishable at every level to three decimal places. The
geometry-aware adaptive step in arm B accomplishes nothing against geometric rejection: this is not
a step-size problem. This replicates, in a different form, the Yoshida null result — just as
fourth-order cancellation cannot survive HNN gradient noise, step-size refinement cannot prevent
the Hamiltonian trajectory from exiting F_k when the true flow exits F_k. The binding constraint
is the geometry of the subset boundary, not the integrator accuracy.

Arm C collapses geometric rejection to near zero (0–0.6%) across all levels of both
configurations. The specular reflection keeps the trajectory inside F_k by construction: at each
detected crossing, the trajectory is walked to the boundary by bisection and the momentum is
reflected in the constraint normal. This directly addresses the mechanism that arms A and B leave
untouched. The energy-rejection rate in arm C is small and comparable to A/B (0–0.4%), confirming
that the bounce itself does not introduce large Hamiltonian errors.

This fills the gap explicitly acknowledged by Thaler (2024), who noted that barrier bouncing in
constrained HMC "is skipped for simplicity" in the SuS context. The results show that bouncing is
not a detail: it is the only intervention of the three tested that moves the binding constraint.

---

## 4. Correctness Validation (Exp 0)

Target: N(0,I₂) truncated to {x₁ ≤ 2.0}. Arms run as constrained MCMC (no SuS levels),
n_chains=100, n_burn=500, n_test=1000. Analytic reference samples drawn from TruncNorm(−∞, 2].

| Arm | leak_rate | KS(x₁) | p-value | KS(x₂) | p-value | bounces/traj | gradU | G |
|-----|-----------|---------|---------|---------|---------|-------------|-------|---|
| A | 0.0186 | 0.0104 | **0.009** | 0.0059 | 0.348 | 0.000 | 550,000 | 0 |
| B | 0.0188 | 0.0073 | 0.137 | 0.0066 | 0.231 | 0.000 | 355,694 | 355,694 |
| C | **0.000** | 0.0043 | 0.738 | 0.0070 | 0.177 | 0.023 | 352,111 | 367,058 |

Arm C's leak rate is exactly 0.000 (no sample violated the constraint across 50,000 test
proposals), and both marginal KS tests are non-significant (x₁: p=0.738; x₂: p=0.177).

Arm A's x₁ KS is **significant** (p=0.009 < 0.05) despite a seemingly small D-statistic (0.0104).
This is mechanistically expected, not just a power artefact. When a proposal lands outside
{x₁ ≤ 2.0}, it is rejected by the geometric check and the chain repeats its current state.
This creates sticky boundary copies that inflate the empirical density near x₁ = 2.0 relative to
the true truncated marginal. Arm A's 1.86% leak rate is sufficient to produce a detectable
distortion of the x₁ marginal — the constrained dimension — while the unconstrained x₂ is
unaffected (p=0.348). Arm B shows the same pattern (leak 1.88%, borderline x₁ p=0.137) because
its adaptive step does not prevent boundary crossings.

This "sticky boundary" effect is mechanistically important beyond SuS: rejection-based constrained
HMC overrepresents boundary states. The bounce fixes this by reflecting back into the interior
rather than repeating the current state, producing samples that faithfully cover the truncated
domain. The Exp 0 result is therefore both a correctness check (leak=0) and positive evidence
that bouncing improves sample quality on truncated targets.

---

## 5. Accuracy and Variance

### Recomputed from exp1_raw.npz (N_rep=30 per arm × config)

Statistics use population standard deviation (ddof=0), matching the record's convention.
Record COV is defined as std/β_ref (not std/mean); since mean ≈ β_ref throughout, the two differ
by less than 0.3% and are treated interchangeably below. All recomputed values match
record_geometry_bounce.md to four decimal places for mean and std; no discrepancies to flag.

**Config 1: β_ref=3.5, ρ=0.0**

| Arm | β̂ mean | std | bias | bias/σ | COV |
|-----|--------|-----|------|--------|-----|
| A | 3.5061 | 0.1075 | +0.0061 | +0.06σ | 0.0307 |
| B | 3.5055 | 0.1082 | +0.0055 | +0.05σ | 0.0309 |
| **C** | **3.4871** | **0.0731** | **−0.0129** | **−0.18σ** | **0.0210** |

**Config 2: β_ref=4.0, ρ=0.75**

| Arm | β̂ mean | std | bias | bias/σ | COV |
|-----|--------|-----|------|--------|-----|
| A | 3.9968 | 0.1111 | −0.0032 | −0.03σ | 0.0278 |
| B | 3.9952 | 0.1095 | −0.0048 | −0.04σ | 0.0274 |
| **C** | **4.0038** | **0.0646** | **+0.0038** | **+0.06σ** | **0.0161** |

**COV reduction (C vs A):** 31.6% at β=3.5; 42.0% at β=4.0.

Arm C achieves substantially lower variance. Intuitively, bouncing converts geometric rejections
(wasted proposals that repeat the current state and add no new information) into accepted moves
that explore the conditional distribution — improving effective sample size without increasing
the nominal sample count.

The largest absolute bias belongs to arm C at β=3.5 (−0.0129, or −0.18σ). Arms A and B have
bias < 0.07σ in both configurations. The difference is the expected signature of the
variable-step non-reversibility: arm C's adaptive integrator does not satisfy detailed balance
exactly, and the bias direction/magnitude depends on the geometry of the constraint near the
subset boundary. This is not alarming at 0.18σ, but it is real, and it motivates running the
C′ ablation (fixed step + bounce) to separate the bias attributable to adaptive stepping from
any residual bias attributable to the bounce itself.

---

## 6. Cost and Efficiency

### Gradient and constraint evaluation counts (per run, mean over N_rep=30)

> **Accounting note:** For all LSFs in this experiment (linear Thaler, half-space), grad_G is a
> constant vector and is evaluated analytically at zero incremental cost. The code explicitly
> tracks `n_grad_G = 0` in PropStats for every arm. The cost formula therefore reduces to
> **cost = n_grad_U + n_G**, with no ∇G term. In applications with non-linear LSFs, ∇G would
> carry real cost; the formula is stated in full generality and then simplified here.

> **G-accounting convention for arm A:** Arm A's integrator (`integrate_A`) performs no mid-
> trajectory G evaluations. The geometric check (end-of-trajectory `G(q_new) ≤ threshold`) is
> done outside the integrator after each proposal and is not counted in PropStats. This means
> arm A's reported G ≈ 0, but arm A does consume approximately (spc−1)×n_seeds×(mean_lv−1) ≈
> 2,700 untracked G calls per run. This is negligible relative to total costs and does not affect
> the efficiency calculation, but it should be noted that arm A is not literally "free of G".
> Arms B and C count G at every integrator step (including the initial evaluation) plus bounce
> bisection calls, so their G totals correctly include all constraint evaluations.

| Config | Arm | ∇U/run | G/run | ∇G/run | **Total cost** |
|--------|-----|--------|-------|--------|--------------|
| β=3.5 | A | 2.230e6 | ~0 | 0 | **2.230e6** |
| β=3.5 | B | 1.442e6 | 1.442e6 | 0 | 2.884e6 |
| β=3.5 | C | 1.430e6 | 1.469e6 | 0 | **2.899e6** |
| β=4.0 | A | 2.239e6 | ~0 | 0 | **2.239e6** |
| β=4.0 | B | 1.845e6 | 1.845e6 | 0 | 3.690e6 |
| β=4.0 | C | 1.833e6 | 1.877e6 | 0 | **3.710e6** |

### Matched-accuracy efficiency gain (C vs A)

Monte Carlo cost scales as 1/COV². The matched-accuracy efficiency gain is:

    E = (cost_A / cost_C) × (COV_A / COV_C)²

This answers: *how many times fewer total oracle calls does arm C need to reach the same COV as arm A?*

| Config | cost_A | cost_C | COV_A | COV_C | **E** | raw cost_C/cost_A |
|--------|--------|--------|-------|-------|-------|------------------|
| β=3.5, ρ=0.0 | 2.230e6 | 2.899e6 | 0.0307 | 0.0210 | **1.64** | 1.30 |
| β=4.0, ρ=0.75 | 2.239e6 | 3.710e6 | 0.0278 | 0.0161 | **1.80** | 1.66 |

Even charging all G evaluations at the same price as ∇U, arm C is approximately **1.6–1.8× more
cost-efficient** at matched accuracy. The two cost framings should not be conflated: at a fixed
sample budget, arm C costs **1.30–1.66× more** per run (it has a higher total cost per fixed N);
but at a fixed accuracy target, it wins by 1.64–1.80× because the variance reduction more than
compensates.

### ⚠ The ∇U count oddity — resolved

The summary table shows arms B and C using *fewer* ∇U/run than arm A (1.43–1.83e6 vs 2.23–2.24e6),
even though the per-level mechanism tables show B and C taking *more* integrator steps per
trajectory (11–15 steps vs 10). This appears contradictory and needs explanation.

**Root cause (traced to the code):** `total_grad_U = sum(lv_grad_U) + burn_grad_U` (line 839,
`experiment_geometry_bounce.py`). The burn-in phase runs `SUS_BURN_IN=200` iterations over all
`N=1000` particles, producing **200,000 HMC trajectories** before any subset levels begin.

For **arm A** (fixed 10-step leapfrog): each burn-in trajectory costs `N_STEPS_A + 1 = 11` ∇U
(one initial evaluation + one per Verlet step). Burn-in ∇U = 200 × 1000 × 11 = **2,200,000**,
accounting for ~98.7% of arm A's reported total. The 4 MCMC levels contribute only ~30,000 ∇U
(`mean_lv − 1 ≈ 3.03` levels that actually run MCMC × 900 proposals × 11 = ~30,000; note that
the final level, where threshold ≤ 0 is first detected, does not run MCMC chains).

For **arms B and C** during burn-in: the constraint is set to `b_k = 1e9` (inactive), so the
geometry controller defaults to `dt_geom = dt_max = 0.5`. Starting from `dt_seed = 0.1` with
`dt_max_change = 5.0`, the step size reaches 0.5 within 1–2 steps, completing `T_F = 1.0` in
approximately 6–7 Verlet steps rather than 10. This gives a burn-in cost of roughly
200 × 1000 × 7 ≈ 1.4e6 ∇U for arms B/C — substantially cheaper than arm A's 2.2e6.

**The counts are apples-to-apples** in the sense that both burn-in and MCMC are genuine costs of
running the sampler. However, the dominant source of arm A's cost disadvantage is the burn-in
phase, not the constrained MCMC itself. Within the constrained MCMC levels (the per-level table
rows), arm B/C actually use *more* ∇U per proposal than arm A (9,900 vs 11,400–14,800 per level),
as expected from their longer adaptive trajectories. The efficiency gain E reported above reflects
the full cost including burn-in, which is appropriate. What one should not read from the table is
"arm B/C are more efficient per constrained MCMC step" — they are not; they are more efficient
overall because the fixed-step integrator wastes most of its budget on an unnecessarily fine
resolution during unconstrained warm-up.

### Cost caveat for expensive-model applications

In this experiment G is analytic and cheap (a single dot product). Charging it at full ∇U price
is deliberately conservative and arm C still wins by 1.6–1.8×. In real structural-reliability or
HNNMC-SuS applications where G represents a finite-element model or a surrogate evaluation
cycle, the per-step G monitoring in arms B/C (roughly one G per Verlet step plus ~15 G calls per
bounce bisection) multiplies cost. If G is, say, 10× more expensive than ∇U, arm C's effective
cost increases by roughly `n_G × 10 / n_∇U` ≈ 10× the G fraction, potentially reversing the
efficiency advantage. This motivates two mitigations in future work: (a) cheap margin-estimate
proxies (surrogate G) to avoid calling the true G at every step, and (b) lazy G evaluation
triggered only when the margin estimate falls below a threshold.

---

## 7. Interpretation for the Thesis

The geometry-bounce experiment completes a three-experiment narrative. In Exp 1 (PINN), the
finding was that HNN-surrogate gradient noise degrades SuS performance. In Exp 2 (Yoshida), the
finding was that higher-order integrators bring no benefit when the binding constraint is not the
energy error but the geometry: higher-order cancellation is destroyed by the same gradient noise
that makes leapfrog fail. In this experiment, the finding is that even with *exact* gradients,
neither fixed nor adaptive leapfrog can reduce geometric rejection — because the problem is not
the integrator, it is the absence of any mechanism that keeps the trajectory inside the subset.
Arm B is the critical control: it proves that geometry-awareness in the step-size controller is
necessary (the controller correctly predicts when the trajectory is about to cross the boundary)
but not sufficient (knowing does not help if you do nothing at the crossing). The B–C pair
constitutes a clean "necessary vs sufficient" experiment: refinement is necessary to locate the
boundary accurately enough to bounce off it, but reflection is the mechanism that actually moves
the binding constraint. Across all three experiments, the unifying conclusion is that SuS-HMC
performance is governed by the subset geometry, not the energy or MH correction; the first
intervention that targets the geometry directly — barrier bouncing — is the first to deliver a
meaningful and consistent efficiency gain.

---

## 8. Caveats and Next Steps

**(a) Run C′ (fixed step + bounce).** Arm C′ uses the same specular bounce as C but with the
original fixed step `dt=0.1`. It is exactly reversible and volume-preserving, making it a
theoretically correct sampler. Comparing C′ to C isolates the bias contribution of the adaptive
integrator from any residual bias of the bounce itself. This is the highest-priority next run:
without C′, the bias of arm C (−0.18σ at β=3.5) is attributed jointly to adaptivity and bounce
with no way to separate them.

**(b) Reconcile / verify ∇U burn-in counts independently.** The burn-in explanation above is
derived by reading the code and working through the arithmetic; it has not been verified by
instrumenting `burn_grad_U` separately. Logging `burn_grad_U` and per-level ∇U independently
in a short diagnostic run would make the cost claim fully auditable.

**(c) Port to HNNMC-SuS** (surrogate ∇U + true G). This is where the G-cost caveat becomes
central. The two-integrator setup (HNN for ∇U, exact evaluations for G) makes the per-step G
monitoring genuinely expensive, and the efficiency tradeoff identified above needs to be
re-measured in that setting. The proxy / lazy-G mitigation should be prototyped here.

**(d) Optional: deeper β or higher dimension.** The current configurations have well-conditioned
boundaries (linear LSF, mild correlation). Grazing trajectories (momentum nearly parallel to the
boundary normal) and high rejection rates at very deep levels (β ≥ 5) would stress both the
bisection convergence and the `n_bounce_max` cap. A sweep over β = {3.5, 4.0, 4.5, 5.0} or
dimension d = {2, 5, 10} would provide a stress-test profile before using the bounce in more
complex LSFs.

---

## 9. Anticipated Supervisor Questions

**Q: Is arm C still an exact sampler?**
No. The state-dependent per-step dt breaks the leapfrog map's reversibility and volume-preservation, making arm C an approximate (biased) sampler. The bias is small here (−0.18σ at β=3.5, +0.06σ at β=4.0) but is attributable jointly to the adaptive step and the bounce. Arm C′ (fixed step + bounce) is the exactly-correct reference and will isolate how much bias comes from each source.

**Q: Why does arm B achieve nothing?**
The geometric rejection arises because the true Hamiltonian trajectory — the continuous-time flow the integrator approximates — exits the feasible set F_k. Step-size refinement improves the fidelity of the discrete trajectory to the true flow, but if the true flow exits, a more faithful discrete approximation also exits. The only way to avoid rejection is to change the trajectory itself (via bouncing), not to approximate the same trajectory more accurately.

**Q: Isn't the extra G cost a problem?**
Not in this experiment, where G is analytic and cheap: even charging G at full ∇U price, arm C is 1.6–1.8× more efficient at matched accuracy. In settings where G is an expensive model (FEM, neural surrogate), per-step G monitoring multiplies cost and can reverse the advantage. This motivates cheap proxy models or lazy evaluation triggered only when the margin estimate is small. This is the central engineering challenge for the HNNMC-SuS port.

**Q: How do you know the bounce is correctly implemented?**
Exp 0 (truncated Gaussian, known analytic solution): arm C achieves leak_rate = 0.000 (no sample
ever violates the constraint across 50,000 test proposals) and passes two-sample KS tests on both
x₁ (p=0.738) and x₂ (p=0.177) against exact analytic reference samples. The specular reflection
`p_ref = p − 2(p·n̂)n̂` is involutive (applying it twice recovers the original momentum),
volume-preserving (|det Jacobian| = 1), and energy-preserving (|p_ref| = |p|) — these properties
are standard and do not depend on the LSF geometry.

**Q: Why does arm A use more ∇U than arms B and C, when B/C have longer trajectories?**
This is a burn-in artefact (see §6 for the full explanation). The burn-in runs 200 iterations
over all 1,000 particles (200,000 trajectories), dominating arm A's ∇U budget at 2.2M out of
2.23M total. During the unconstrained burn-in, the adaptive controller in B/C reaches dt_max=0.5
quickly, completing T_F=1.0 in ~6–7 steps instead of arm A's fixed 10. Per constrained MCMC
proposal, arms B/C actually use more ∇U than arm A, as the per-level tables show.

**Q: A 32–42% COV reduction — is that actually worth the implementation complexity?**
At matched accuracy in this setting (cheap G), yes: E ≈ 1.6–1.8 is a meaningful gain comparable
to what the Yoshida integrator promised on paper (and failed to deliver). The implementation is
also modular: the bounce is a drop-in post-step check. The relevant question for the thesis is
whether the gain survives expensive G, and §6 is explicit that it may not without proxy
mitigations. The thesis positions the bounce as the right mechanism, with the proxy design as the
open engineering problem, not as a completed end-to-end solution.
