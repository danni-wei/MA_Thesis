from __future__ import annotations

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
    