# Geometry-aware adaptive step + Barrier Bouncing in HMC-SuS

## Controller settings

| param | value |
|-------|-------|
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
| SUS_N | 1000 |  SUS_P0 | 0.1 |  SUS_BURN_IN | 200 |

**Correctness note**: arms B and C use state-dependent per-step dt,
breaking leapfrog reversibility/volume-preservation → approximate samplers.
The dt_geom regulariser (dt_seed floor) ensures the step never stalls at
m→0 but increases bias vs the pure CFL limit. C' (fixed+bounce) is the
exactly-correct reference for isolating adaptivity bias.

## Exp 0 — Bounce sanity on truncated Gaussian

Target: N(0,I₂) truncated to {x₁ ≤ 2.0}.

| arm | leak_rate | KS_x1 | KS_x1_pval | KS_x2 | KS_x2_pval | bounces_mean | frac_bounced | grad_U | G |
|-----|-----------|-------|------------|-------|------------|-------------|--------------|--------|---|
| A | 0.0186 | 0.0104 | 0.009 | 0.0059 | 0.348 | 0.000 | 0.000 | 550000 | 0 |
| B | 0.0188 | 0.0073 | 0.137 | 0.0066 | 0.231 | 0.000 | 0.000 | 355694 | 355694 |
| C | 0.0000 | 0.0043 | 0.738 | 0.0070 | 0.177 | 0.023 | 0.023 | 352111 | 367058 |

## Exp 1 — 3-arm SuS on Thaler linear LSF  (N_rep=30)

### beta=3.5, rho=0.0  (Pf_ref=2.3263e-04, beta_ref=3.5)

#### Summary table

| arm | beta_hat | std | bias | COV_beta | grad_U/run | G/run | mean_lv |
|-----|----------|-----|------|----------|------------|-------|---------|
| A | 3.5061 | 0.1075 | +0.0061 | 0.031 | 2.230e+06 | 0.000e+00 | 4.03 |
| B | 3.5055 | 0.1082 | +0.0055 | 0.031 | 1.442e+06 | 1.442e+06 | 4.03 |
| C | 3.4871 | 0.0731 | -0.0129 | 0.021 | 1.430e+06 | 1.469e+06 | 4.00 |

#### Per-level mechanism

**Arm A**

| lv | geom_rej | energy_rej | accept | cond_ener_rej | bounces_mean | bounce_traj | grad_U | G | n_steps_mean |
|----|----------|------------|--------|---------------|-------------|-------------|--------|---|--------------|
| L1 | 0.655 | 0.000 | 0.344 | 0.001 | 0.000 | 0.000 | 9900 | 0 | 10.0 |
| L2 | 0.855 | 0.000 | 0.145 | 0.002 | 0.000 | 0.000 | 9900 | 0 | 10.0 |
| L3 | 0.934 | 0.000 | 0.066 | 0.003 | 0.000 | 0.000 | 9900 | 0 | 10.0 |
| L4 | 0.957 | 0.000 | 0.043 | 0.000 | 0.000 | 0.000 | 9900 | 0 | 10.0 |

**Arm B**

| lv | geom_rej | energy_rej | accept | cond_ener_rej | bounces_mean | bounce_traj | grad_U | G | n_steps_mean |
|----|----------|------------|--------|---------------|-------------|-------------|--------|---|--------------|
| L1 | 0.656 | 0.001 | 0.343 | 0.003 | 0.000 | 0.000 | 11402 | 11402 | 11.7 |
| L2 | 0.854 | 0.001 | 0.145 | 0.004 | 0.000 | 0.000 | 13402 | 13402 | 13.9 |
| L3 | 0.934 | 0.000 | 0.066 | 0.004 | 0.000 | 0.000 | 14490 | 14490 | 15.1 |
| L4 | 0.958 | 0.000 | 0.042 | 0.000 | 0.000 | 0.000 | 14804 | 14804 | 15.4 |

**Arm C**

| lv | geom_rej | energy_rej | accept | cond_ener_rej | bounces_mean | bounce_traj | grad_U | G | n_steps_mean |
|----|----------|------------|--------|---------------|-------------|-------------|--------|---|--------------|
| L1 | 0.000 | 0.003 | 0.997 | 0.003 | 0.703 | 0.655 | 8324 | 17098 | 7.5 |
| L2 | 0.003 | 0.002 | 0.994 | 0.002 | 1.057 | 0.853 | 9484 | 22798 | 8.5 |
| L3 | 0.006 | 0.002 | 0.991 | 0.002 | 1.325 | 0.934 | 10281 | 27052 | 9.1 |

#### What this tells us (H_B and H_C)

H_B (refinement alone): ARM B vs ARM A — if geom_rej rates are similar,
refinement did NOT reduce geometric rejection (consistent with the hypothesis
that the true Hamiltonian trajectory exits F_k regardless of step size).

H_C (bouncing): ARM C vs ARM A/B — if ARM C's geom_rej ≈ 0 and A/B's are
substantial, bouncing successfully prevents geometric rejection.
Cost comparison: C pays extra G/grad_G evals for bisection; net efficiency
= (COV improvement) / (extra eval cost).

### beta=4.0, rho=0.75  (Pf_ref=3.1671e-05, beta_ref=4.0)

#### Summary table

| arm | beta_hat | std | bias | COV_beta | grad_U/run | G/run | mean_lv |
|-----|----------|-----|------|----------|------------|-------|---------|
| A | 3.9968 | 0.1111 | -0.0032 | 0.028 | 2.239e+06 | 0.000e+00 | 4.97 |
| B | 3.9952 | 0.1095 | -0.0048 | 0.027 | 1.845e+06 | 1.845e+06 | 5.00 |
| C | 4.0038 | 0.0646 | +0.0038 | 0.016 | 1.833e+06 | 1.877e+06 | 5.00 |

#### Per-level mechanism

**Arm A**

| lv | geom_rej | energy_rej | accept | cond_ener_rej | bounces_mean | bounce_traj | grad_U | G | n_steps_mean |
|----|----------|------------|--------|---------------|-------------|-------------|--------|---|--------------|
| L1 | 0.515 | 0.002 | 0.483 | 0.003 | 0.000 | 0.000 | 9900 | 0 | 10.0 |
| L2 | 0.713 | 0.001 | 0.286 | 0.004 | 0.000 | 0.000 | 9900 | 0 | 10.0 |
| L3 | 0.818 | 0.000 | 0.181 | 0.002 | 0.000 | 0.000 | 9900 | 0 | 10.0 |
| L4 | 0.883 | 0.000 | 0.117 | 0.002 | 0.000 | 0.000 | 9900 | 0 | 10.0 |

**Arm B**

| lv | geom_rej | energy_rej | accept | cond_ener_rej | bounces_mean | bounce_traj | grad_U | G | n_steps_mean |
|----|----------|------------|--------|---------------|-------------|-------------|--------|---|--------------|
| L1 | 0.515 | 0.002 | 0.483 | 0.004 | 0.000 | 0.000 | 11273 | 11273 | 11.5 |
| L2 | 0.713 | 0.001 | 0.285 | 0.005 | 0.000 | 0.000 | 12770 | 12770 | 13.2 |
| L3 | 0.818 | 0.000 | 0.181 | 0.003 | 0.000 | 0.000 | 13620 | 13620 | 14.1 |
| L4 | 0.885 | 0.000 | 0.115 | 0.002 | 0.000 | 0.000 | 14347 | 14347 | 14.9 |

**Arm C**

| lv | geom_rej | energy_rej | accept | cond_ener_rej | bounces_mean | bounce_traj | grad_U | G | n_steps_mean |
|----|----------|------------|--------|---------------|-------------|-------------|--------|---|--------------|
| L1 | 0.000 | 0.004 | 0.996 | 0.004 | 0.535 | 0.515 | 9203 | 15870 | 8.7 |
| L2 | 0.001 | 0.003 | 0.996 | 0.003 | 0.809 | 0.712 | 9877 | 19999 | 9.2 |
| L3 | 0.003 | 0.004 | 0.993 | 0.004 | 1.007 | 0.817 | 10374 | 23009 | 9.5 |
| L4 | 0.005 | 0.002 | 0.992 | 0.002 | 1.187 | 0.883 | 10799 | 25727 | 9.8 |

#### What this tells us (H_B and H_C)

H_B (refinement alone): ARM B vs ARM A — if geom_rej rates are similar,
refinement did NOT reduce geometric rejection (consistent with the hypothesis
that the true Hamiltonian trajectory exits F_k regardless of step size).

H_C (bouncing): ARM C vs ARM A/B — if ARM C's geom_rej ≈ 0 and A/B's are
substantial, bouncing successfully prevents geometric rejection.
Cost comparison: C pays extra G/grad_G evals for bisection; net efficiency
= (COV improvement) / (extra eval cost).

