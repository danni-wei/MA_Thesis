# Cost Accounting: Geometry-Bounce Experiment

**Purpose:** Re-account the eval costs reported in EXPERIMENT_SUMMARY.md, separating
burn-in from the constrained SuS phase and reporting efficiency under both G-free and
G=nabla_U pricing. The total-cost efficiency (E approx 1.6-1.8) in EXPERIMENT_SUMMARY.md
is contaminated by a burn-in step-size artefact; this document separates the
mechanism-relevant numbers from the artefact.

**Sources:** record_geometry_bounce.md (per-level tables + summary tables),
exp1_raw.npz (raw beta_hat arrays, N_rep=30), experiment_geometry_bounce.py (burn-in logic).

---

## 1. Phase Decomposition

### Structure of `total_grad_U` (from code line 839)

    total_grad_U = sum(per_level_grad_U) + burn_grad_U

Two phases:

- **Burn-in phase** (`burn_grad_U + burn_G`): 200 iterations x N=1000 particles = 200,000
  HMC proposals run with `b_k = 1e9` (constraint inactive). This is pure unconstrained
  warm-up; it does not test the A/B/C mechanism difference.

- **Constrained-MCMC phase** (`sum(per_level_grad_U) + sum(per_level_G)`): the actual
  SuS levels where proposals are generated under the active subset constraint. This is
  the mechanism-relevant phase.

### Derivation of per-phase costs

**Arm A burn-in (EXACT — fixed integrator):**
`integrate_A` runs `N_STEPS_A=10` leapfrog steps with 1 initial grad_U:
cost per trajectory = N_STEPS_A + 1 = 11 grad_U, 0 G.

    burn_grad_U_A = 200 x 1000 x 11 = 2,200,000  (exact)
    burn_G_A      = 0                              (integrate_A has no G calls;
                                                    burn-in does not check constraint)

`constrained_grad_U_A = total_grad_U_A - 2,200,000` (derived exactly below).

**Arms B and C burn-in (DERIVED BY SUBTRACTION):**
During burn-in, `b_k = 1e9` makes the geometry controller default to `dt_geom = dt_max = 0.5`.
Starting from `dt_seed = 0.1` with `dt_max_change = 5.0`, the step reaches 0.5 within 1-2
steps, completing T_F=1.0 in ~6-9 verlet steps rather than arm A's fixed 10. For arm C,
`b_k = 1e9` means no boundary crossing is detected (margin = 1e9 - G(q) >> 0 always), so
arm C's burn-in behaves identically to arm B's (no bisection G calls). Consistency check:
`burn_grad_U_C approx burn_G_C` (G approx grad_U during unconstrained stepping for B/C)
-- confirmed below.

Burn costs derived as: `total - constrained_estimate` (shown below).

**Constrained-MCMC phase costs:**
Per-level `grad_U` and `G` are read from the mechanism tables in record_geometry_bounce.md
(mean over reps that had that level). The number of MCMC levels per run is `mean_lv - 1`
(the final SuS iteration detects `threshold <= 0` and returns immediately without running
MCMC chains). Method label for each:

- EXACT: `mean_lv` is an integer => all 30 reps had identical MCMC level count.
- DERIVED: arm A constrained = total - known burn (no assumption needed).
- ESTIMATED: `mean_lv` non-integer => mix of level counts across reps; computed as
  weighted sum from per-level table. Marked with [est].

---

## 2. Per-Phase Cost Table

All values are per-run means (mean over N_rep=30). Totals in the last two columns match
the record to within rounding of the 3-significant-figure summary values.

### Config 1: beta=3.5, rho=0.0

| Arm | burn_nablaU | burn_G | cons_nablaU | cons_G | total_nablaU | total_G | derivation |
|-----|------------|--------|------------|--------|-------------|---------|------------|
| A | 2,200,000 | 0 | 30,000 | 0 | 2,230,000 | 0 | burn exact; cons = total - burn |
| B | 1,402,213 [est] | 1,402,213 [est] | 39,787 [est] | 39,787 [est] | 1,442,000 | 1,442,000 | see note (i) |
| C | 1,401,911 | 1,402,052 | 28,089 | 66,948 | 1,430,000 | 1,469,000 | exact: all 30 reps = 3 MCMC levels |

Note (i) — arm B beta=3.5 estimation: `mean_lv=4.03` implies ~29 reps with 3 MCMC
levels and ~1 rep with 4 MCMC levels (0.03*30=0.9 rounds to 1). Constrained grad_U
estimated as `(29*(11402+13402+14490) + 1*(11402+13402+14490+14804))/30 = 39,787`.
Since `n_G = n_grad_U` for arm B throughout (one G call per verlet step plus one initial,
matching grad_U exactly -- visible from total_G = total_grad_U for B in the record),
`cons_G = cons_grad_U = 39,787`. Burn derived by subtraction.

Cross-check -- burn_grad_U per burn-in trajectory:
  A: 2,200,000 / 200,000 = 11.0 grad_U/traj  (= N_STEPS_A+1, exact)
  B: 1,402,213 / 200,000 = 7.01 grad_U/traj  (adaptive dt, ~6 verlet steps, unconstrained)
  C: 1,401,911 / 200,000 = 7.01 grad_U/traj  (same adaptive dt; no bounces with b_k=1e9)
  C (G): 1,402,052 / 200,000 = 7.01 G/traj   (G approx grad_U during unconstrained => consistent)

### Config 2: beta=4.0, rho=0.75

| Arm | burn_nablaU | burn_G | cons_nablaU | cons_G | total_nablaU | total_G | derivation |
|-----|------------|--------|------------|--------|-------------|---------|------------|
| A | 2,200,000 | 0 | 39,000 | 0 | 2,239,000 | 0 | burn exact; cons = total - burn |
| B | 1,792,990 | 1,792,990 | 52,010 | 52,010 | 1,845,000 | 1,845,000 | exact: all 30 reps = 4 MCMC levels |
| C | 1,792,747 | 1,792,395 | 40,253 | 84,605 | 1,833,000 | 1,877,000 | exact: all 30 reps = 4 MCMC levels |

Cross-check -- burn_grad_U per trajectory:
  A: 2,200,000 / 200,000 = 11.0 grad_U/traj  (exact, same as beta=3.5)
  B: 1,792,990 / 200,000 = 8.96 grad_U/traj  (more steps than beta=3.5: rho=0.75 increases
                                                Hessian max eigenvalue to 1/(1-0.75)=4,
                                                energy safety cap more restrictive => smaller dt)
  C: 1,792,747 / 200,000 = 8.96 grad_U/traj  (same; no bounces during unconstrained burn-in)
  C (G): 1,792,395 / 200,000 = 8.96 G/traj   (consistent)

Observation: burn costs for B and C are nearly identical across both configs (B and C
behave the same during unconstrained burn-in). The higher burn cost at beta=4.0 vs
beta=3.5 for B/C (~8.96 vs ~7.01 steps/traj) reflects the correlated target's higher
curvature limiting the energy safety step size.

---

## 3. Constrained-Phase Cost Breakdown (per-level granularity)

For reference, the per-level grad_U and G are reproduced here so the constrained sums
above can be verified directly.

### beta=3.5, rho=0.0  (3 MCMC levels for A, B, C; A/B have occasional L4)

| Arm | L1 nablaU | L2 nablaU | L3 nablaU | L4 nablaU | L1 G | L2 G | L3 G | sum_nablaU | sum_G |
|-----|-----------|-----------|-----------|-----------|------|------|------|-----------|-------|
| A | 9,900 | 9,900 | 9,900 | (9,900*) | 0 | 0 | 0 | 30,000** | 0 |
| B | 11,402 | 13,402 | 14,490 | (14,804*) | 11,402 | 13,402 | 14,490 | 39,787** [est] | 39,787** [est] |
| C | 8,324 | 9,484 | 10,281 | — | 17,098 | 22,798 | 27,052 | 28,089 | 66,948 |

*L4 appears in per-level table as the average over the ~1 rep (out of 30) that ran L4 MCMC.
**weighted sum accounting for ~1 rep having 4 MCMC levels.

### beta=4.0, rho=0.75  (4 MCMC levels for all arms, all 30 reps)

| Arm | L1 nablaU | L2 nablaU | L3 nablaU | L4 nablaU | L1 G | L2 G | L3 G | L4 G | sum_nablaU | sum_G |
|-----|-----------|-----------|-----------|-----------|------|------|------|------|-----------|-------|
| A | 9,900 | 9,900 | 9,900 | 9,900 | 0 | 0 | 0 | 0 | 39,000* | 0 |
| B | 11,273 | 12,770 | 13,620 | 14,347 | 11,273 | 12,770 | 13,620 | 14,347 | 52,010 | 52,010 |
| C | 9,203 | 9,877 | 10,374 | 10,799 | 15,870 | 19,999 | 23,009 | 25,727 | 40,253 | 84,605 |

*39,600 from 4*9,900; slight discrepancy (39,000 from subtraction) reflects ~0.9 rep with only
 3 MCMC levels (mean_lv=4.97, not exactly 5.00).

### G cost anatomy for arm C (constrained phase)

For arm C, G >> grad_U in the constrained phase. The G evals come from two sources:
  - Per-step monitoring: 1 G per verlet step + 1 G initial = (n_steps_mean + 1) per proposal
  - Bisection per bounce: up to bisect_max_iter=30 G calls; empirically ~15 G per bounce
    (derived: at L1 beta=3.5, extra_G = 17,098 - 900*(7.5+1) = 17,098 - 7,650 = 9,448
     from 900*0.703 = 633 bounces => 9,448/633 approx 14.9 G per bounce)

At beta=3.5 L1: 0.703 bounces per proposal x ~15 G per bounce = ~10.5 extra G per
  proposal on top of ~8.5 step-monitoring G => total ~19 G per proposal vs 0 for arm A.
  
The constrained-phase G/grad_U ratio for C is ~2.4 (beta=3.5) to ~2.1 (beta=4.0). When
G is priced at full nabla_U cost, this ~2.4x G overhead per constrained grad_U is the
dominant cost driver for arm C.

---

## 4. COV Values (from exp1_raw.npz, ddof=0)

| Config | Arm | COV (std/mean) |
|--------|-----|---------------|
| beta=3.5 | A | 0.030664 |
| beta=3.5 | B | 0.030868 |
| beta=3.5 | C | 0.020970 |
| beta=4.0 | A | 0.027806 |
| beta=4.0 | B | 0.027409 |
| beta=4.0 | C | 0.016123 |

COV reduction C vs A: 31.6% at beta=3.5; 42.0% at beta=4.0.
B vs A: -0.7% at beta=3.5 (B is marginally WORSE); +1.4% at beta=4.0 (negligible).

---

## 5. Constrained-Phase Efficiency

E = (cost_A / cost_arm) * (COV_A / COV_arm)^2.
E > 1 means the arm is more efficient than A at matched accuracy.
cost = nabla_U only (G-free) or nabla_U + G (G = nabla_U pricing).
nabla_G = 0 throughout (grad_G is analytic/constant for all linear LSFs tested).

### Config 1: beta=3.5, rho=0.0

| Arm | cons_nablaU | cons_G | G-free cost | G=nablaU cost | raw (Gfree) | raw (G=nablaU) | E (Gfree) | E (G=nablaU) |
|-----|------------|--------|-------------|--------------|------------|---------------|-----------|-------------|
| A | 30,000 | 0 | 30,000 | 30,000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B | 39,787 [est] | 39,787 [est] | 39,787 | 79,574 | 1.326 | 2.652 | 0.744 | 0.372 |
| C | 28,089 | 66,948 | 28,089 | 95,037 | 0.936 | 3.168 | 2.284 | 0.675 |

### Config 2: beta=4.0, rho=0.75

| Arm | cons_nablaU | cons_G | G-free cost | G=nablaU cost | raw (Gfree) | raw (G=nablaU) | E (Gfree) | E (G=nablaU) |
|-----|------------|--------|-------------|--------------|------------|---------------|-----------|-------------|
| A | 39,000 | 0 | 39,000 | 39,000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B | 52,010 | 52,010 | 52,010 | 104,020 | 1.334 | 2.667 | 0.772 | 0.386 |
| C | 40,253 | 84,605 | 40,253 | 124,858 | 1.032 | 3.202 | 2.882 | 0.929 |

**Reading the table:**

Arm B: raw cost is 1.33-1.33x (G-free) and 2.65-2.67x (G=nablaU) compared to A, with
  effectively zero COV improvement. E < 1 under all pricings. Arm B is strictly less
  efficient than A in the constrained phase. This is the clearest quantification of H_B
  rejection: refinement adds cost without benefit.

Arm C (G-free): cost is 0.94x A at beta=3.5 (slightly cheaper on grad_U alone!) and
  1.03x at beta=4.0, combined with 32-42% COV reduction => E = 2.28 and 2.88.
  The grad_U saving for arm C is real: arm C uses fewer verlet steps per constrained
  proposal (n_steps_mean 7.5-9.8 vs 10.0 for A) because bouncing keeps trajectories in
  F_k, reducing the need for very small CFL steps near the boundary.

Arm C (G=nablaU): bisection adds ~15 G per bounce; total constrained G is 2.4x constrained
  grad_U. Raw cost = 3.2x A. The COV reduction is not large enough to compensate:
  E = 0.68 at beta=3.5 (C is LESS efficient than A), E = 0.93 at beta=4.0 (roughly
  break-even). At these configurations, if G costs as much as nabla_U, the bounce does
  not pay for itself in the constrained phase.

---

## 6. Total-Cost Efficiency (burn-in included, for reference only)

**THIS IS NOT THE HEADLINE.** These numbers are dominated by the burn-in artefact and
do not reflect the mechanism's benefit. Reproduced here for traceability to
EXPERIMENT_SUMMARY.md.

| Config | Arm | E (G-free total) | E (G=nablaU total) | note |
|--------|-----|-----------------|-------------------|------|
| beta=3.5 | B | 1.527 | 0.763 | burn artefact inflates B's apparent G-free gain |
| beta=3.5 | C | 3.336 | 1.645 | **burn artefact + ignoring G inflates this number** |
| beta=4.0 | B | 1.248 | 0.624 | |
| beta=4.0 | C | 3.636 | 1.795 | **burn artefact + ignoring G inflates this number** |

The G=nablaU total-cost E values (1.64 and 1.80) cited in EXPERIMENT_SUMMARY.md are
produced by the burn-in cost of arm A (~98.7% of arm A's total) inflating the denominator.
If the burn-in were run with matching step sizes across arms (e.g., all arms using the
adaptive controller during burn-in), arm A's burn cost would drop from 2.2M to ~1.4M,
and the total-cost advantage would largely disappear. The mechanism benefit lives
entirely in the constrained phase.

---

## 7. Honest Headline and Key Caveats

**What the data DO support:**

Within the constrained SuS levels:
- Under **G-free pricing**: arm C reduces COV by 32-42% while using slightly fewer or
  comparable nabla_U per run (0.94x at beta=3.5, 1.03x at beta=4.0). The matched-accuracy
  efficiency gain is **E = 2.28 (beta=3.5) and 2.88 (beta=4.0)**. This is the honest
  mechanism-relevant number when G is cheap or free.

- Under **G = nablaU pricing**: arm C's bisection adds ~15 G calls per bounce event,
  inflating constrained-phase cost to 3.2x arm A's. The COV reduction is insufficient
  to compensate: **E = 0.68 (beta=3.5, arm C is LESS efficient) and 0.93 (beta=4.0,
  roughly break-even)**. In the expensive-G regime, barrier bouncing as implemented
  here does not pay for itself.

- The total-cost E approx 1.6-1.8 (EXPERIMENT_SUMMARY.md) is **a burn-in artefact** and
  should not be cited as the mechanism's benefit. It would vanish if all arms used the
  same step-size strategy during warm-up.

- Arm B is strictly worse than A under all pricings in the constrained phase (E < 1),
  providing the sharpest possible evidence that step-size refinement alone is not the solution.

**Critical untested regime:**

Both configurations use **linear LSFs** (Thaler rho=0.0 and rho=0.75). For all such
LSFs, grad_G is a constant vector of O(1) cost. The "G = nablaU" pricing model is
already artificial for these cases (it penalizes G as if it were as expensive as a
gradient, even though it is actually a dot product). The **true expensive-G regime** --
where G involves a finite-element solve, neural surrogate evaluation, or a full forward
model call -- is **completely untested** in this experiment. This is the regime most
relevant to structural reliability (FEM-based LSF) and to HNNMC-SuS (where the true
LSF is expensive and nabla_U comes from a surrogate). The G=nablaU sensitivity analysis
above shows that once G carries real cost, the bounce's efficiency advantage erodes
quickly. Quantifying the breakeven G/nablaU cost ratio, and whether proxy strategies
(cheap margin estimate, lazy G evaluation) can extend the G-free advantage into the
expensive-G regime, is the central open question for the HNNMC-SuS port. No claim should
be made about arm C's efficiency in real reliability applications based on these results
alone.

**Single honest thesis-facing sentence:**

"Barrier bouncing eliminates geometric rejection (51-96% -> 0-0.6%) and, under G-free
pricing within the constrained SuS phase, achieves 2.3-2.9x better cost-efficiency at
matched accuracy than standard leapfrog; however, the bounce's bisection G cost
(~15 G evaluations per bounce event) reduces this advantage to 0.68-0.93x when G is
priced at nabla_U cost, so the mechanism's practical benefit depends critically on the
relative cost of the limit-state function evaluation -- the regime motivating cheap
surrogate proxies in the HNNMC-SuS context."
