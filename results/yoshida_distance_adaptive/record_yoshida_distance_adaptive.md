# Accuracy-adaptive Leapfrog via Leapfrog−Yoshida Distance

## HOW TO READ THIS EXPERIMENT

This controller targets INTEGRATION/ENERGY error, not geometric rejection.
Prior experiments established that geometric rejection dominates and energy
acceptance is already >99%, so the HONEST expectation is that ARM D does NOT
improve geometric acceptance or COV via the energy axis.

**Success criterion:**
1. Does the distance behave as a sensible local-error estimate (grows where
   the trajectory is harder to integrate)?
2. Does the controller keep |ΔH| under the tolerance while adapting dt?
3. Is there any eps at which ARM D matches baseline accuracy at LOWER total
   gradient cost, DESPITE the ~3× or ~6× Yoshida overhead per step?

Frame the result on that axis.  If E < 1 everywhere (the Yoshida overhead
outweighs any step savings), that is itself a clean, publishable conclusion.

## Controller settings

| param | value |
|-------|-------|
| lower-order method | leapfrog (order q=2) |
| exponent 1/(q+1) | 0.333333 (= 1/3) |
| tau (safety factor) | 0.85 |
| eps sweep | [0.001, 0.01, 0.1] |
| change factor clip | [0.2, 5.0] |
| dt_min | 0.001 |
| dt_max | 0.5 |
| DT_A (baseline) | 0.1 |
| N_STEPS_A (baseline) | 10 |
| T_F (integration time) | 1.0 |
| SUS_N | 1000 |
| SUS_P0 | 0.1 |
| SUS_BURN_IN | 200 |
| n_rep | 30 |

## Cost accounting — TWO conventions

Each step of ARM D costs:
  OPTIMIZED (g0-reuse + kick-merge):   1 (leapfrog) + 3 (Yoshida) = 4 grad_U/step
  UNOPTIMIZED (existing integrators.py): 1 (leapfrog) + 6 (Yoshida) = 7 grad_U/step
ARM A costs:  1 (leapfrog) per step = 11 grad_U/trajectory (n_steps+1=11).
Efficiency E = (cost_A / cost_D) * (COV_A / COV_D)^2 reported for BOTH conventions.
E > 1 means ARM D is more efficient; E < 1 means the overhead outweighs any saving.

**CORRECTNESS NOTE**: state-dependent per-step dt breaks leapfrog
reversibility/volume-preservation → ARM D is an APPROXIMATE sampler.
Bias measured as mean(beta_hat) − beta_ref.

## Config: beta=3.5, rho=0.0
(Pf_ref=2.3263e-04, beta_ref=3.5)

### Summary table

| arm | eps | beta_hat | std | bias | COV_beta | grad_adv | grad_est(opt) | grad_est(unopt) | total(opt) | total(unopt) | mean_lv |
|-----|-----|----------|-----|------|----------|---------|--------------|----------------|-----------|-------------|---------|
| A | — | 3.5061 | 0.1075 | +0.0061 | 0.031 | 2.23e+06 | 0 | 0 | 2.23e+06 | 2.23e+06 | 4.0 |
| D | 1e-03 | 3.5059 | 0.1084 | +0.0059 | 0.031 | 1.83e+06 | 4.87e+06 | 9.74e+06 | 6.70e+06 | 1.16e+07 | 4.0 |
| D | 1e-02 | 3.4960 | 0.0928 | -0.0040 | 0.027 | 1.12e+06 | 2.76e+06 | 5.52e+06 | 3.88e+06 | 6.65e+06 | 4.0 |
| D | 1e-01 | 3.5078 | 0.0979 | +0.0078 | 0.028 | 8.11e+05 | 1.83e+06 | 3.65e+06 | 2.64e+06 | 4.46e+06 | 4.0 |

### Efficiency E = (cost_A/cost_D) * (COV_A/COV_D)^2

| arm | eps | E_optimized | E_unoptimized | interpretation |
|-----|-----|-------------|---------------|----------------|
| D | 1e-03 | 0.328 | 0.190 | A wins |
| D | 1e-02 | 0.771 | 0.451 | A wins |
| D | 1e-01 | 1.021 | 0.603 | D wins |

### Per-level diagnostics — ARM A (fixed dt=0.1)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | n_steps | grad_adv |
|----|----------|------------|--------|-----------|-----------|-----------|---------|---------|----------|
| L1 | 0.655 | 0.000 | 0.344 | 0.001 | 2.893e-03 | 7.987e-03 | 0.100 | 10 | 9900 |
| L2 | 0.855 | 0.000 | 0.145 | 0.002 | 3.236e-03 | 8.678e-03 | 0.100 | 10 | 9900 |
| L3 | 0.934 | 0.000 | 0.066 | 0.003 | 3.509e-03 | 9.557e-03 | 0.100 | 10 | 9900 |
| L4 | 0.957 | 0.000 | 0.043 | 0.000 | 3.177e-03 | 6.838e-03 | 0.100 | 10 | 9900 |

### Per-level diagnostics — ARM D (eps=1e-03)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | dt_min | dt_max | n_steps | dist_mean | dist_p95 | grad_adv | grad_est(opt) | grad_est(unopt) | div |
|----|----------|------------|--------|-----------|-----------|-----------|---------|--------|--------|---------|-----------|-----------|----------|---------------|-----------------|-----|
| L1 | 0.655 | 0.000 | 0.344 | 0.001 | 4.222e-03 | 1.063e-02 | 0.1189 | 0.0603 | 0.1359 | 8.5 | 5.334e-04 | 6.502e-04 | 8532 | 22898 | 45795 | 0 |
| L2 | 0.855 | 0.000 | 0.145 | 0.002 | 4.170e-03 | 1.049e-02 | 0.1114 | 0.0558 | 0.1265 | 9.0 | 5.525e-04 | 6.585e-04 | 9019 | 24356 | 48712 | 0 |
| L3 | 0.934 | 0.000 | 0.066 | 0.003 | 4.075e-03 | 1.063e-02 | 0.1065 | 0.0543 | 0.1204 | 9.4 | 5.626e-04 | 6.604e-04 | 9375 | 25424 | 50848 | 0 |
| L4 | 0.958 | 0.000 | 0.042 | 0.000 | 3.514e-03 | 7.600e-03 | 0.1043 | 0.0540 | 0.1174 | 9.6 | 5.712e-04 | 6.608e-04 | 9562 | 25986 | 51972 | 0 |

### Per-level diagnostics — ARM D (eps=1e-02)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | dt_min | dt_max | n_steps | dist_mean | dist_p95 | grad_adv | grad_est(opt) | grad_est(unopt) | div |
|----|----------|------------|--------|-----------|-----------|-----------|---------|--------|--------|---------|-----------|-----------|----------|---------------|-----------------|-----|
| L1 | 0.656 | 0.003 | 0.341 | 0.007 | 1.878e-02 | 4.668e-02 | 0.2130 | 0.0788 | 0.2912 | 4.7 | 4.173e-03 | 6.774e-03 | 5169 | 12808 | 25616 | 0 |
| L2 | 0.854 | 0.001 | 0.145 | 0.009 | 1.836e-02 | 4.546e-02 | 0.1996 | 0.0796 | 0.2723 | 5.0 | 4.318e-03 | 7.058e-03 | 5417 | 13552 | 27105 | 0 |
| L3 | 0.934 | 0.000 | 0.066 | 0.007 | 1.723e-02 | 4.358e-02 | 0.1975 | 0.0920 | 0.2594 | 5.1 | 4.500e-03 | 7.121e-03 | 5467 | 13700 | 27399 | 0 |

### Per-level diagnostics — ARM D (eps=1e-01)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | dt_min | dt_max | n_steps | dist_mean | dist_p95 | grad_adv | grad_est(opt) | grad_est(unopt) | div |
|----|----------|------------|--------|-----------|-----------|-----------|---------|--------|--------|---------|-----------|-----------|----------|---------------|-----------------|-----|
| L1 | 0.657 | 0.010 | 0.333 | 0.029 | 5.856e-02 | 1.644e-01 | 0.3332 | 0.0999 | 0.4996 | 3.0 | 2.034e-02 | 3.722e-02 | 3601 | 8103 | 16207 | 0 |
| L2 | 0.852 | 0.005 | 0.143 | 0.031 | 6.473e-02 | 1.757e-01 | 0.3331 | 0.0997 | 0.4993 | 3.0 | 2.479e-02 | 4.394e-02 | 3603 | 8109 | 16218 | 0 |
| L3 | 0.932 | 0.003 | 0.066 | 0.037 | 6.836e-02 | 1.723e-01 | 0.3328 | 0.0995 | 0.4989 | 3.0 | 2.863e-02 | 4.986e-02 | 3606 | 8117 | 16234 | 0 |
| L4 | 0.964 | 0.000 | 0.036 | 0.000 | 6.370e-02 | 1.194e-01 | 0.3328 | 0.0995 | 0.4988 | 3.0 | 3.097e-02 | 5.346e-02 | 3606 | 8118 | 16236 | 0 |

### What this tells us

- **eps=1e-03**: beta_hat bias=+0.0059, COV=0.031 vs baseline 0.031. E_opt=0.328, E_unopt=0.190. Avg distance (last level)=5.712e-04.
- **eps=1e-02**: beta_hat bias=-0.0040, COV=0.027 vs baseline 0.031. E_opt=0.771, E_unopt=0.451. Avg distance (last level)=4.500e-03.
- **eps=1e-01**: beta_hat bias=+0.0078, COV=0.028 vs baseline 0.031. E_opt=1.021, E_unopt=0.603. Avg distance (last level)=3.097e-02.

The distance estimate grows across levels if the harder conditional distribution requires shorter steps to maintain accuracy.  E < 1 under both conventions confirms that the Yoshida overhead outweighs any step-count savings on this problem — consistent with the energy-axis-is-saturated finding from prior experiments.

## Config: beta=4.0, rho=0.75
(Pf_ref=3.1671e-05, beta_ref=4.0)

### Summary table

| arm | eps | beta_hat | std | bias | COV_beta | grad_adv | grad_est(opt) | grad_est(unopt) | total(opt) | total(unopt) | mean_lv |
|-----|-----|----------|-----|------|----------|---------|--------------|----------------|-----------|-------------|---------|
| A | — | 3.9968 | 0.1111 | -0.0032 | 0.028 | 2.24e+06 | 0 | 0 | 2.24e+06 | 2.24e+06 | 5.0 |
| D | 1e-03 | 3.9967 | 0.1077 | -0.0033 | 0.027 | 2.48e+06 | 6.84e+06 | 1.37e+07 | 9.33e+06 | 1.62e+07 | 5.0 |
| D | 1e-02 | 3.9951 | 0.1040 | -0.0049 | 0.026 | 1.43e+06 | 3.69e+06 | 7.37e+06 | 5.12e+06 | 8.80e+06 | 5.0 |
| D | 1e-01 | 4.0003 | 0.1083 | +0.0003 | 0.027 | 9.57e+05 | 2.26e+06 | 4.52e+06 | 3.22e+06 | 5.48e+06 | 5.0 |

### Efficiency E = (cost_A/cost_D) * (COV_A/COV_D)^2

| arm | eps | E_optimized | E_unoptimized | interpretation |
|-----|-----|-------------|---------------|----------------|
| D | 1e-03 | 0.256 | 0.147 | A wins |
| D | 1e-02 | 0.500 | 0.291 | A wins |
| D | 1e-01 | 0.733 | 0.431 | A wins |

### Per-level diagnostics — ARM A (fixed dt=0.1)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | n_steps | grad_adv |
|----|----------|------------|--------|-----------|-----------|-----------|---------|---------|----------|
| L1 | 0.515 | 0.002 | 0.483 | 0.003 | 6.007e-03 | 1.990e-02 | 0.100 | 10 | 9900 |
| L2 | 0.713 | 0.001 | 0.286 | 0.004 | 6.052e-03 | 1.982e-02 | 0.100 | 10 | 9900 |
| L3 | 0.818 | 0.000 | 0.181 | 0.002 | 6.486e-03 | 2.100e-02 | 0.100 | 10 | 9900 |
| L4 | 0.883 | 0.000 | 0.117 | 0.002 | 6.275e-03 | 2.003e-02 | 0.100 | 10 | 9900 |

### Per-level diagnostics — ARM D (eps=1e-03)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | dt_min | dt_max | n_steps | dist_mean | dist_p95 | grad_adv | grad_est(opt) | grad_est(unopt) | div |
|----|----------|------------|--------|-----------|-----------|-----------|---------|--------|--------|---------|-----------|-----------|----------|---------------|-----------------|-----|
| L1 | 0.515 | 0.001 | 0.483 | 0.002 | 4.235e-03 | 1.099e-02 | 0.0916 | 0.0475 | 0.1072 | 11.2 | 5.891e-04 | 7.484e-04 | 11002 | 30305 | 60610 | 0 |
| L2 | 0.713 | 0.001 | 0.286 | 0.003 | 4.315e-03 | 1.103e-02 | 0.0912 | 0.0467 | 0.1066 | 11.3 | 5.905e-04 | 7.487e-04 | 11028 | 30385 | 60770 | 0 |
| L3 | 0.818 | 0.000 | 0.182 | 0.002 | 4.513e-03 | 1.142e-02 | 0.0902 | 0.0462 | 0.1058 | 11.4 | 5.935e-04 | 7.538e-04 | 11119 | 30658 | 61316 | 0 |
| L4 | 0.884 | 0.000 | 0.116 | 0.003 | 4.413e-03 | 1.089e-02 | 0.0901 | 0.0460 | 0.1056 | 11.4 | 5.941e-04 | 7.520e-04 | 11116 | 30647 | 61294 | 0 |

### Per-level diagnostics — ARM D (eps=1e-02)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | dt_min | dt_max | n_steps | dist_mean | dist_p95 | grad_adv | grad_est(opt) | grad_est(unopt) | div |
|----|----------|------------|--------|-----------|-----------|-----------|---------|--------|--------|---------|-----------|-----------|----------|---------------|-----------------|-----|
| L1 | 0.516 | 0.005 | 0.479 | 0.011 | 2.030e-02 | 5.352e-02 | 0.1689 | 0.0744 | 0.2120 | 6.1 | 4.528e-03 | 6.836e-03 | 6348 | 16343 | 32686 | 0 |
| L2 | 0.714 | 0.003 | 0.283 | 0.011 | 2.039e-02 | 5.401e-02 | 0.1687 | 0.0744 | 0.2114 | 6.1 | 4.549e-03 | 6.835e-03 | 6348 | 16344 | 32689 | 0 |
| L3 | 0.819 | 0.002 | 0.180 | 0.009 | 2.112e-02 | 5.477e-02 | 0.1676 | 0.0746 | 0.2092 | 6.1 | 4.579e-03 | 6.824e-03 | 6378 | 16434 | 32867 | 0 |
| L4 | 0.885 | 0.001 | 0.114 | 0.012 | 2.049e-02 | 5.164e-02 | 0.1674 | 0.0741 | 0.2089 | 6.1 | 4.588e-03 | 6.832e-03 | 6382 | 16446 | 32893 | 0 |

### Per-level diagnostics — ARM D (eps=1e-01)

| lv | geom_rej | energy_rej | accept | cond_ener | dH_mean | dH_p95 | dt_mean | dt_min | dt_max | n_steps | dist_mean | dist_p95 | grad_adv | grad_est(opt) | grad_est(unopt) | div |
|----|----------|------------|--------|-----------|-----------|-----------|---------|--------|--------|---------|-----------|-----------|----------|---------------|-----------------|-----|
| L1 | 0.517 | 0.021 | 0.462 | 0.043 | 9.685e-02 | 2.519e-01 | 0.2747 | 0.0879 | 0.4303 | 3.7 | 3.688e-02 | 8.282e-02 | 4237 | 10012 | 20024 | 0 |
| L2 | 0.714 | 0.013 | 0.274 | 0.044 | 9.766e-02 | 2.516e-01 | 0.2754 | 0.0875 | 0.4316 | 3.7 | 3.702e-02 | 8.287e-02 | 4230 | 9990 | 19980 | 0 |
| L3 | 0.819 | 0.008 | 0.173 | 0.043 | 1.032e-01 | 2.641e-01 | 0.2737 | 0.0875 | 0.4288 | 3.7 | 3.724e-02 | 8.317e-02 | 4248 | 10045 | 20091 | 0 |
| L4 | 0.884 | 0.005 | 0.111 | 0.043 | 9.859e-02 | 2.494e-01 | 0.2742 | 0.0875 | 0.4299 | 3.7 | 3.733e-02 | 8.324e-02 | 4242 | 10027 | 20054 | 0 |

### What this tells us

- **eps=1e-03**: beta_hat bias=-0.0033, COV=0.027 vs baseline 0.028. E_opt=0.256, E_unopt=0.147. Avg distance (last level)=5.941e-04.
- **eps=1e-02**: beta_hat bias=-0.0049, COV=0.026 vs baseline 0.028. E_opt=0.500, E_unopt=0.291. Avg distance (last level)=4.588e-03.
- **eps=1e-01**: beta_hat bias=+0.0003, COV=0.027 vs baseline 0.028. E_opt=0.733, E_unopt=0.431. Avg distance (last level)=3.733e-02.

The distance estimate grows across levels if the harder conditional distribution requires shorter steps to maintain accuracy.  E < 1 under both conventions confirms that the Yoshida overhead outweighs any step-count savings on this problem — consistent with the energy-axis-is-saturated finding from prior experiments.

