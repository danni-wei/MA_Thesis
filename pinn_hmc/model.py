from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    dim: int = 2
    hidden_dim: int = 128
    depth: int = 4


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, depth: int = 4):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2")

        layers: List[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(depth - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrajectoryPINN(nn.Module):
    """
    Input:
        [q0, p0, t]   shape [batch, 2*dim + 1]
    Output:
        [q_hat, p_hat] shape [batch, 2*dim]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dim = config.dim
        self.model = MLP(
            in_dim=2 * config.dim + 1,
            out_dim=2 * config.dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )

    def forward(
        self,
        q0: torch.Tensor,
        p0: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if t.ndim == 1:
            t = t.unsqueeze(-1)

        x = torch.cat([q0, p0, t], dim=-1)
        out = self.model(x)

        q_hat = out[:, : self.dim]
        p_hat = out[:, self.dim :]
        return q_hat, p_hat

    def forward_flat(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q0 = x[:, : self.dim]
        p0 = x[:, self.dim : 2 * self.dim]
        t = x[:, 2 * self.dim : 2 * self.dim + 1]
        return self.forward(q0, p0, t)

    def representative_params(self):
        """
        Parameters used by gradient-balancing methods.
        Here we use the last linear layer if possible.
        """
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            return list(self.parameters())
        return list(linear_layers[-1].parameters())

from dataclasses import dataclass


@dataclass
class HNNConfig:
    dim: int = 2
    hidden_dim: int = 128
    depth: int = 4


class HamiltonianNN(nn.Module):
    """
    Input:
        [q, p] shape [batch, 2*dim]
    Output:
        scalar H_theta(q, p) shape [batch, 1]
    """

    def __init__(self, config: HNNConfig):
        super().__init__()
        self.dim = config.dim
        self.model = MLP(
            in_dim=2 * config.dim,
            out_dim=1,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
        )

    def forward(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = torch.cat([q, p], dim=-1)
        return self.model(x)

    def forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        q = x[:, : self.dim]
        p = x[:, self.dim :]
        return self.forward(q, p)

    def representative_params(self):
        linear_layers = [m for m in self.modules() if isinstance(m, nn.Linear)]
        if not linear_layers:
            return list(self.parameters())
        return list(linear_layers[-1].parameters())