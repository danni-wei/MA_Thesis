from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional

import torch

try:
    from .target import EllipticalLimitState, get_device
except ImportError:
    from target import EllipticalLimitState, get_device  # type: ignore[no-redef]


def run_subset_simulation(
    proposal_fn: Callable[[torch.Tensor], torch.Tensor],
    limit_state: EllipticalLimitState,
    r: float,
    n_samples_per_level: int = 1000,
    p0: float = 0.1,
    max_levels: int = 10,
    burn_in: int = 200,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Subset Simulation for failure-probability estimation (Wang et al. 2019
    RS-HMC-SS).

    Parameters
    ----------
    proposal_fn : fn(q [N,2]) -> q_new [N,2]
        One MCMC step applied to all N parallel chains simultaneously.
        Called without any surrounding gradient context so that HMC-based
        proposals that need autograd internally work correctly.
    limit_state : EllipticalLimitState
        Provides evaluate(q, r) -> g tensor and is_failure(q, r).
    r : float
        Radius parameter forwarded to the limit-state function.
    n_samples_per_level : int
        N — number of samples per intermediate level.
    p0 : float
        Target conditional failure probability per level (e.g. 0.1).
    max_levels : int
        Hard cap on conditional levels; returns early if failure domain is
        reached first.
    burn_in : int
        Number of proposal_fn calls used to warm up the N chains before
        collecting level-0 samples.
    device : torch.device, optional
        Defaults to CUDA when available.

    Returns
    -------
    dict with keys
        P_F              – estimated failure probability
        levels           – number of conditional levels used (including the
                           terminating one)
        all_samples      – list of CPU tensors [N, 2], one per level
        thresholds       – list of intermediate thresholds c_1, c_2, …
        mean_proposal_time – mean wall-clock time per proposal_fn call (s)

    Algorithm
    ---------
    P_F = p0^k × (n_fail / N)  where k is the number of levels with
    threshold > 0 that preceded the terminating level.
    """
    if device is None:
        device = get_device()

    N = n_samples_per_level
    n_seeds = max(1, int(round(p0 * N)))
    samples_per_chain = math.ceil(N / n_seeds)

    proposal_times: List[float] = []
    all_samples: List[torch.Tensor] = []
    thresholds: List[float] = []

    # ── Level 0: unconditional samples via burn-in from the origin ──────────
    q = torch.zeros(N, 2, device=device)
    for _ in range(burn_in):
        t0 = time.perf_counter()
        q = proposal_fn(q).detach()
        proposal_times.append(time.perf_counter() - t0)

    all_samples.append(q.cpu())

    # ── Conditional levels ───────────────────────────────────────────────────
    for level in range(max_levels):
        with torch.no_grad():
            g_vals = limit_state.evaluate(q, r)           # [N]

        threshold = float(torch.quantile(g_vals, p0).item())
        thresholds.append(threshold)

        if threshold <= 0.0:
            # The p0-quantile already lies in the failure domain; estimate
            # failure probability directly.  No additional p0 factor is needed
            # for this level because we haven't done a conditional resampling
            # step yet — q still represents the distribution from the previous
            # (or unconditional) level.
            n_fail = int((g_vals <= 0.0).sum().item())
            P_F = (p0 ** level) * (n_fail / N)
            return _build_result(P_F, level + 1, all_samples, thresholds, proposal_times)

        # ── MCMC resampling within {g(q) ≤ threshold} ───────────────────────
        # Seeds: the n_seeds samples with the smallest (closest-to-failure) g.
        seed_idx = torch.argsort(g_vals)[:n_seeds]        # ascending order
        chains = q[seed_idx].detach().clone()              # [n_seeds, 2]

        chain_snapshots: List[torch.Tensor] = [chains.clone()]
        for _ in range(samples_per_chain - 1):
            t0 = time.perf_counter()
            q_prop = proposal_fn(chains).detach()
            proposal_times.append(time.perf_counter() - t0)

            with torch.no_grad():
                g_prop = limit_state.evaluate(q_prop, r)  # [n_seeds]
                accept = (g_prop <= threshold).unsqueeze(-1)  # [n_seeds, 1]
                chains = torch.where(accept, q_prop, chains)

            chain_snapshots.append(chains.clone())

        # chain_snapshots is a list of (samples_per_chain) tensors [n_seeds, 2].
        # Stack so each chain's samples are contiguous, then truncate to N.
        new_q = (
            torch.stack(chain_snapshots, dim=1)   # [n_seeds, samples_per_chain, 2]
            .reshape(-1, 2)[:N]                   # [N, 2]
        )
        q = new_q
        all_samples.append(q.cpu())

    # max_levels reached without entering the failure domain
    with torch.no_grad():
        g_final = limit_state.evaluate(q, r)
    n_fail = int((g_final <= 0.0).sum().item())
    P_F = (p0 ** max_levels) * (n_fail / N)

    return _build_result(P_F, max_levels, all_samples, thresholds, proposal_times)


def _build_result(
    P_F: float,
    levels: int,
    all_samples: List[torch.Tensor],
    thresholds: List[float],
    proposal_times: List[float],
) -> Dict:
    mean_t = sum(proposal_times) / len(proposal_times) if proposal_times else 0.0
    return {
        "P_F": P_F,
        "levels": levels,
        "all_samples": all_samples,
        "thresholds": thresholds,
        "mean_proposal_time": mean_t,
    }


# ── Standalone smoke-test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from pinn_hmc.target import EllipticalLimitState, WangBananaTarget, get_device

    torch.manual_seed(0)
    device = get_device()

    target = WangBananaTarget()
    ls = EllipticalLimitState()

    def make_hmc_proposal(
        target: WangBananaTarget,
        step_size: float = 0.1,
        n_steps: int = 5,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Baseline leapfrog HMC with Metropolis accept/reject."""

        def proposal_fn(q: torch.Tensor) -> torch.Tensor:
            q = q.detach()
            batch = q.shape[0]
            p_cur = target.sample_momentum(batch, q.device)

            # Leapfrog integration (detach at each step to avoid graph build-up)
            q_p = q.clone()
            p_p = (p_cur - 0.5 * step_size * target.grad_U(q_p)).detach()
            for i in range(n_steps):
                q_p = (q_p + step_size * p_p).detach()
                if i < n_steps - 1:
                    p_p = (p_p - step_size * target.grad_U(q_p)).detach()
            p_p = (p_p - 0.5 * step_size * target.grad_U(q_p)).detach()

            # Metropolis acceptance
            with torch.no_grad():
                H0 = target.H(q, p_cur)
                H1 = target.H(q_p, p_p)
                log_alpha = torch.clamp(H0 - H1, max=80.0)
                accept = (
                    torch.rand(batch, device=q.device)
                    < torch.exp(log_alpha).clamp(max=1.0)
                ).unsqueeze(-1)  # [batch, 1]
            return torch.where(accept, q_p, q)

        return proposal_fn

    proposal = make_hmc_proposal(target)

    print("Running Subset Simulation (baseline HMC, N=500, p0=0.1, r=1.5) ...")
    result = run_subset_simulation(
        proposal_fn=proposal,
        limit_state=ls,
        r=1.5,
        n_samples_per_level=500,
        p0=0.1,
        max_levels=5,
        burn_in=100,
        device=device,
    )

    print(f"  P_F             = {result['P_F']:.4e}")
    print(f"  levels          = {result['levels']}")
    print(f"  thresholds      = {[f'{t:.4f}' for t in result['thresholds']]}")
    print(f"  mean_proposal_t = {result['mean_proposal_time'] * 1e3:.3f} ms")
    print(f"  level sets      = {len(result['all_samples'])}")
    for i, s in enumerate(result["all_samples"]):
        print(f"    level {i}: {tuple(s.shape)}")
    print("Done.")
