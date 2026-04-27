from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class BananaTargetConfig:
    b: float = 0.15
    sigma1: float = 1.0
    sigma2: float = 1.0


class BananaTarget:
    """
    2D banana-shaped target.

    Potential:
        U(q1, q2) = 0.5 * (q1 / sigma1)^2
                    + 0.5 * ((q2 - b * (q1^2 - sigma1^2)) / sigma2)^2

    Kinetic:
        K(p) = 0.5 * p^T p

    Hamiltonian:
        H(q, p) = U(q) + K(p)
    """

    def __init__(self, config: Optional[BananaTargetConfig] = None):
        cfg = config or BananaTargetConfig()
        self.b = float(cfg.b)
        self.sigma1 = float(cfg.sigma1)
        self.sigma2 = float(cfg.sigma2)
        self.dim = 2

    def U(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] != 2:
            raise ValueError(f"Expected q[..., 2], got {tuple(q.shape)}")

        q1 = q[..., 0]
        q2 = q[..., 1]
        z = q2 - self.b * (q1**2 - self.sigma1**2)

        return 0.5 * (q1 / self.sigma1) ** 2 + 0.5 * (z / self.sigma2) ** 2

    def grad_U(self, q: torch.Tensor) -> torch.Tensor:
        if q.shape[-1] != 2:
            raise ValueError(f"Expected q[..., 2], got {tuple(q.shape)}")

        q1 = q[..., 0]
        q2 = q[..., 1]
        z = q2 - self.b * (q1**2 - self.sigma1**2)

        dU_dq1 = q1 / (self.sigma1**2) - (2.0 * self.b * q1 * z) / (self.sigma2**2)
        dU_dq2 = z / (self.sigma2**2)

        return torch.stack([dU_dq1, dU_dq2], dim=-1)

    def K(self, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(p**2, dim=-1)

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.U(q) + self.K(p)

    def sample_momentum(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.dim, device=device)


@dataclass
class WangBananaConfig:
    a: float = 1.15
    b: float = 0.5
    rho: float = 0.9


class WangBananaTarget:
    """
    2D banana-shaped target from Wang et al. 2019 (Structural Safety) eq.34.

    Transformation from correlated Gaussians (u1, u2) with correlation rho:
        x = u1 * a
        y = u2 * a + b * (u1^2 - a^2) + rho * u2 * a

    Potential U(q) = -log f_XY(x, y) derived via change of variables from the
    bivariate normal for (u1, u2) (eq.35).  grad_U is computed by autograd.
    """

    def __init__(self, config: Optional[WangBananaConfig] = None):
        cfg = config or WangBananaConfig()
        self.a = float(cfg.a)
        self.b = float(cfg.b)
        self.rho = float(cfg.rho)
        self.dim = 2
        # Normalisation constant: log(2π√(1-ρ²)) + log(a²(1+ρ))
        self._log_const = (
            math.log(2 * math.pi)
            + 0.5 * math.log(1 - self.rho**2)
            + math.log(self.a**2 * (1 + self.rho))
        )

    def U(self, q: torch.Tensor) -> torch.Tensor:
        """Negative log-density -log f_XY(x, y) via the inverse transform."""
        if q.shape[-1] != 2:
            raise ValueError(f"Expected q[..., 2], got {tuple(q.shape)}")
        x, y = q[..., 0], q[..., 1]
        # Inverse: u1 = x/a, u2 = (y - b*(u1²-a²)) / (a*(1+ρ))
        u1 = x / self.a
        u2 = (y - self.b * (u1**2 - self.a**2)) / (self.a * (1 + self.rho))
        # Bivariate-normal quadratic form for corr=ρ
        quad = (u1**2 - 2 * self.rho * u1 * u2 + u2**2) / (2 * (1 - self.rho**2))
        return quad + self._log_const

    def grad_U(self, q: torch.Tensor) -> torch.Tensor:
        # torch.enable_grad() lets this work even inside @torch.no_grad() contexts
        # (e.g. generate_teacher_dataset / generate_hnn_dataset).
        # When q already carries a grad (training time), use autograd.grad with
        # create_graph=True so the physics-loss residual stays differentiable
        # w.r.t. model parameters.
        with torch.enable_grad():
            if q.requires_grad:
                return torch.autograd.grad(
                    self.U(q).sum(), q, create_graph=True
                )[0]
            q_ = q.detach().requires_grad_(True)
            self.U(q_).sum().backward()
            return q_.grad.clone()

    def K(self, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum(p**2, dim=-1)

    def H(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.U(q) + self.K(p)

    def sample_momentum(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch_size, self.dim, device=device)


class EllipticalLimitState:
    """
    Elliptical limit-state function from Wang et al. 2019 eq.40.

        g(x, y; r) = r² - ((x·cos θ - y·sin θ)/c1)² - ((x·sin θ + y·cos θ)/c2)²

    Failure domain: g ≤ 0.
    """

    def __init__(
        self,
        c1: float = 1.0,
        c2: float = 0.5,
        theta: float = math.pi / 4,
    ):
        self.c1 = c1
        self.c2 = c2
        self.theta = theta
        self._cos_t = math.cos(theta)
        self._sin_t = math.sin(theta)

    def evaluate(self, q: torch.Tensor, r: float) -> torch.Tensor:
        if q.shape[-1] != 2:
            raise ValueError(f"Expected q[..., 2], got {tuple(q.shape)}")
        x, y = q[..., 0], q[..., 1]
        cos_t, sin_t = self._cos_t, self._sin_t
        term1 = (x * cos_t - y * sin_t) / self.c1
        term2 = (x * sin_t + y * cos_t) / self.c2
        return r**2 - term1**2 - term2**2

    def is_failure(self, q: torch.Tensor, r: float) -> torch.Tensor:
        """True where g(q, r) ≤ 0 (failure domain)."""
        return self.evaluate(q, r) <= 0


if __name__ == "__main__":
    device = get_device()

    wang = WangBananaTarget()
    q = torch.randn(5, 2, device=device)
    p = wang.sample_momentum(5, device)
    print(f"WangBananaTarget  U: {wang.U(q).shape}  grad_U: {wang.grad_U(q).shape}  H: {wang.H(q, p).shape}")
    print(f"  U values: {wang.U(q)}")

    els = EllipticalLimitState()
    r = 1.5
    g = els.evaluate(q, r)
    fail = els.is_failure(q, r)
    print(f"EllipticalLimitState  g: {g.shape}  is_failure: {fail.shape}")
    print(f"  g values: {g}")
    print(f"  failure mask: {fail}")
    print("All checks passed.")
