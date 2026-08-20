# MA_Thesis

Code for Danni Wei's Master's thesis on accelerating Hamiltonian Monte Carlo
(HMC) sampling for rare-event / structural reliability problems using
Physics-Informed Neural Networks (PINNs) and Hamiltonian Neural Networks
(HNNs), evaluated inside a Subset Simulation (SuS) framework.

## Repository layout

```
pinn_hmc/       Core PINN/HNN-HMC framework and experiments (main body of work)
hmc_hnnmc/      Earlier-stage HMC / HNN-MC prototypes; HNNMC/ includes
                adapted third-party code (see License and Attribution)
adim/           Polynomial chaos expansion (PCE) tooling, separate from
                the HMC experiments
_run_*.py       Root-level entry points that launch individual experiments
results/        Experiment outputs (git-ignored; regenerate by running the
                scripts below)
```

### `pinn_hmc/` — PINN/HNN-HMC framework

| File | What it does |
|---|---|
| `target.py` | Target distributions (banana, correlated Gaussian) and limit-state functions |
| `model.py` | `TrajectoryPINN` and `HamiltonianNN` (HNN) model definitions |
| `integrators.py` | Leapfrog and 4th-order Yoshida symplectic integrators (torch) |
| `losses.py` | Supervised / physics / initial-condition loss terms for PINN and HNN training |
| `balancers.py` | Loss-weighting strategies: Fixed, EMA-normalized, Progress, GradNorm |
| `train.py` | Training loops for PINN and HNN models |
| `run_experiment.py` | Dataset generation and HMC baseline / PINN sampler runners |
| `compare_pinn_hnn.py` | End-to-end comparison harness: baseline HMC vs PINN-HMC vs HNN-HMC |
| `sus.py` | Core Subset Simulation algorithm (level-by-level rare-event sampling) |
| `experiment_sus.py` | SuS experiment: Baseline HMC vs PINN-HMC vs HNN-HMC on the banana target |
| `experiment_sus_stage2.py` | Stage 2 SuS integrator comparison: leapfrog+HNN vs Yoshida+HNN |
| `experiment_jacobian.py` | Jacobian-correction experiment across three MCMC kernels in SuS |
| `experiment_geometry_bounce.py` | 3-arm HMC comparison using a barrier-bounce mechanism in SuS |
| `experiment_highdim_bounce.py` | Extends the geometry-bounce experiment to higher dimensions |
| `experiment_bb_hnn.py` | Barrier-bouncing driven by an HNN gradient (Arm C_HNN vs Arm C) |
| `experiment_yoshida_distance_adaptive.py` | Adaptive leapfrog step size driven by a leapfrog-Yoshida local error estimate |
| `smoke_yoshida_hnn.py` | Single-chain HMC smoke test with the full HNN vector field |
| `smoke_stepsize_sweep.py` | Step-size convergence smoke test |
| `plot_history.py` | Plots training loss history from a saved checkpoint |
| `plot_sampling.py` | Plots sampling results from a saved checkpoint |
| `plot_compare_pinn_hnn.py` | Plots baseline / PINN / HNN sample comparison from a checkpoint |

### `hmc_hnnmc/` — earlier prototypes

| File | What it does |
|---|---|
| `HMC/hmc.py` | HMC sampler for Subset Simulation |
| `HMC/hmc_annulus.py` | HMC demo on an annulus (donut) target |
| `HMC/hmc_banana_shaped.py` | Banana-shaped target distribution |
| `HMC/sus_banana.py` | SuS on the banana target: leapfrog vs Yoshida4 |
| `HMC/sus_high_dim.py` | SuS on a high-dimensional linear LSF: leapfrog vs Yoshida4 |
| `HMC/sus_leapfrog_vs_yoshida.py` | SuS: leapfrog vs 4th-order Yoshida integrator comparison |
| `HMC/sus_stepsize_sweep.py` | Step-size sensitivity sweep, leapfrog vs Yoshida4 |
| `MH/mh.py`, `MH/mh_banana_shaped.py` | Plain Metropolis-Hastings sampler and banana-target variant |
| `HNNMC/hnnmc/*` | HNN model, training and utility code (third-party, see below) |
| `HNNMC/hnn_hmc_*.py` | HNN-HMC sampling on banana / donut / linear-LSF / series-system targets |
| `HNNMC/SuS_HNN_*.py` | Subset Simulation driven by HNN-HMC, per target |
| `HNNMC/SuS_series_system.py/` | ERA-group Subset Simulation toolbox (Nataf transform, adaptive conditional sampling) |

### `adim/` — polynomial chaos expansion tooling

| File | What it does |
|---|---|
| `tests/pcetools/distribution.py` | Abstract `Distribution` base class |
| `tests/pcetools/normal.py`, `unifrom.py` | Normal and uniform distribution implementations |
| `tests/pcetools/abstract_pce.py` | Abstract PCE base class and `Derivative` helper |
| `tests/pcetools/pce.py` | Concrete PCE implementation |
| `tests/test_pce.py` | Unit tests |

### Root-level entry points

Thin CLI wrappers that call one experiment's `main()`:

| Script | Runs |
|---|---|
| `_run_v15.py` | `pinn_hmc/smoke_yoshida_hnn.py` |
| `_run_v16.py` | `pinn_hmc/smoke_stepsize_sweep.py` |
| `_run_v2stage2.py` | `pinn_hmc/experiment_sus_stage2.py` (`--full` for n_rep=30) |
| `_run_v3_geometry_bounce.py` | `pinn_hmc/experiment_geometry_bounce.py` (`--full`, `--skip-exp0`) |
| `_run_v4_yoshida_distance.py` | `pinn_hmc/experiment_yoshida_distance_adaptive.py` (`--full` for n_rep=30) |

`experiment_bb_hnn.py`, `experiment_highdim_bounce.py`, `experiment_sus.py`
and `experiment_jacobian.py` are run directly (e.g.
`python -u pinn_hmc/experiment_bb_hnn.py --stage1`).

## Setup

```bash
pip install -r requirements.txt
```

`requirements.txt` covers the main `pinn_hmc/` experiment code. See
`adim/environment.yml` if you also need the PCE tooling's (macOS-specific)
conda environment.

## License and Attribution

The original code in this repository — `pinn_hmc/`, the root-level
`_run_*.py` scripts, `adim/`, and `hmc_hnnmc/HMC/` and `hmc_hnnmc/MH/` — is
authored by Danni Wei and released under the MIT License (see `LICENSE`).

Only the `hmc_hnnmc/HNNMC/` subdirectory contains code adapted from two
third-party sources, and their original copyright headers have been kept
intact in each file:

- [BIhNNs](https://github.com/IdahoLabResearch/BIhNNs) by Idaho National
  Laboratory / Battelle Energy Alliance, LLC — MIT License.
- [hamiltonian-nn](https://github.com/greydanus/hamiltonian-nn) by Sam
  Greydanus, Misko Dzamba, and Jason Yosinski — Apache License 2.0.

I do not claim authorship of this adapted code; it is included, with
attribution, because it was used and modified as part of this thesis's
experiments. See `hmc_hnnmc/HNNMC/NOTICE` for the consolidated attribution
notice, and the header of each individual file for the specific license it
falls under.
