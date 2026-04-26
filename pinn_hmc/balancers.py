from __future__ import annotations

from typing import Dict, Optional

import torch


class LossBalancer:
    def __call__(self, raw_losses: Dict[str, torch.Tensor], model=None):
        raise NotImplementedError


class FixedBalancer(LossBalancer):
    def __init__(self, lambda_sup: float = 1.0, lambda_phys: float = 0.1, lambda_ic: float = 1.0):
        self.lambda_sup = float(lambda_sup)
        self.lambda_phys = float(lambda_phys)
        self.lambda_ic = float(lambda_ic)

    def __call__(self, raw_losses: Dict[str, torch.Tensor], model=None):
        total = (
            self.lambda_sup * raw_losses["sup"]
            + self.lambda_phys * raw_losses["phys"]
            + self.lambda_ic * raw_losses["ic"]
        )

        stats = {
            "total_loss": total.detach(),
            "lambda_sup": torch.tensor(self.lambda_sup, device=total.device),
            "lambda_phys": torch.tensor(self.lambda_phys, device=total.device),
            "lambda_ic": torch.tensor(self.lambda_ic, device=total.device),
        }
        return total, stats


class EMANormalizer(LossBalancer):
    """
    Method 1:
    Normalize each loss by its EMA scale.
    """

    def __init__(self, beta: float = 0.98, eps: float = 1e-8):
        self.beta = float(beta)
        self.eps = float(eps)
        self.ema: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, raw_losses: Dict[str, torch.Tensor], model=None):
        if self.ema is None:
            self.ema = {
                "sup": raw_losses["sup"].detach(),
                "phys": raw_losses["phys"].detach(),
                "ic": raw_losses["ic"].detach(),
            }
        else:
            for k in ["sup", "phys", "ic"]:
                self.ema[k] = self.beta * self.ema[k] + (1.0 - self.beta) * raw_losses[k].detach()

        lambda_sup = 1.0 / (self.ema["sup"] + self.eps)
        lambda_phys = 1.0 / (self.ema["phys"] + self.eps)
        lambda_ic = 1.0 / (self.ema["ic"] + self.eps)

        total = (
            lambda_sup * raw_losses["sup"]
            + lambda_phys * raw_losses["phys"]
            + lambda_ic * raw_losses["ic"]
        )

        stats = {
            "total_loss": total.detach(),
            "lambda_sup": lambda_sup.detach(),
            "lambda_phys": lambda_phys.detach(),
            "lambda_ic": lambda_ic.detach(),
            "ema_sup": self.ema["sup"].detach(),
            "ema_phys": self.ema["phys"].detach(),
            "ema_ic": self.ema["ic"].detach(),
        }
        return total, stats


class ProgressBalancer(LossBalancer):
    """
    Method 2:
    Increase weight for the objective that is learning more slowly.
    """

    def __init__(self, alpha: float = 1.0, eps: float = 1e-8):
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.initial: Optional[Dict[str, torch.Tensor]] = None

    def __call__(self, raw_losses: Dict[str, torch.Tensor], model=None):
        if self.initial is None:
            self.initial = {
                "sup": raw_losses["sup"].detach(),
                "phys": raw_losses["phys"].detach(),
                "ic": raw_losses["ic"].detach(),
            }

        r_sup = raw_losses["sup"].detach() / (self.initial["sup"] + self.eps)
        r_phys = raw_losses["phys"].detach() / (self.initial["phys"] + self.eps)
        r_ic = raw_losses["ic"].detach() / (self.initial["ic"] + self.eps)

        raw = torch.stack([
            r_sup.pow(self.alpha),
            r_phys.pow(self.alpha),
            r_ic.pow(self.alpha),
        ])
        lambdas = raw / (raw.sum() + self.eps)

        lambda_sup, lambda_phys, lambda_ic = lambdas[0], lambdas[1], lambdas[2]

        total = (
            lambda_sup * raw_losses["sup"]
            + lambda_phys * raw_losses["phys"]
            + lambda_ic * raw_losses["ic"]
        )

        stats = {
            "total_loss": total.detach(),
            "lambda_sup": lambda_sup.detach(),
            "lambda_phys": lambda_phys.detach(),
            "lambda_ic": lambda_ic.detach(),
            "progress_sup": r_sup.detach(),
            "progress_phys": r_phys.detach(),
            "progress_ic": r_ic.detach(),
        }
        return total, stats


class GradNormBalancer(LossBalancer):
    """
    Method 3:
    Adjust weights so gradient norms are closer to each other.

    This is a simplified online balancing rule.
    """

    def __init__(
        self,
        eta: float = 0.1,
        eps: float = 1e-8,
        init_lambdas: Optional[Dict[str, float]] = None,
    ):
        self.eta = float(eta)
        self.eps = float(eps)
        init_lambdas = init_lambdas or {"sup": 1.0, "phys": 1.0, "ic": 1.0}
        self.lambdas = {
            "sup": float(init_lambdas["sup"]),
            "phys": float(init_lambdas["phys"]),
            "ic": float(init_lambdas["ic"]),
        }

    @staticmethod
    def _grad_norm(loss: torch.Tensor, params) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        pieces = []
        for g in grads:
            if g is not None:
                pieces.append(g.reshape(-1))
        if not pieces:
            return torch.tensor(0.0, device=loss.device)
        return torch.norm(torch.cat(pieces), p=2)

    def __call__(self, raw_losses: Dict[str, torch.Tensor], model=None):
        if model is None:
            raise ValueError("GradNormBalancer requires model")

        params = model.representative_params()

        g_sup = self._grad_norm(self.lambdas["sup"] * raw_losses["sup"], params)
        g_phys = self._grad_norm(self.lambdas["phys"] * raw_losses["phys"], params)
        g_ic = self._grad_norm(self.lambdas["ic"] * raw_losses["ic"], params)

        g_avg = (g_sup + g_phys + g_ic) / 3.0

        self.lambdas["sup"] *= float((g_avg / (g_sup + self.eps)).item() ** self.eta)
        self.lambdas["phys"] *= float((g_avg / (g_phys + self.eps)).item() ** self.eta)
        self.lambdas["ic"] *= float((g_avg / (g_ic + self.eps)).item() ** self.eta)

        s = self.lambdas["sup"] + self.lambdas["phys"] + self.lambdas["ic"]
        self.lambdas["sup"] /= s
        self.lambdas["phys"] /= s
        self.lambdas["ic"] /= s

        lambda_sup = torch.tensor(self.lambdas["sup"], device=raw_losses["sup"].device)
        lambda_phys = torch.tensor(self.lambdas["phys"], device=raw_losses["sup"].device)
        lambda_ic = torch.tensor(self.lambdas["ic"], device=raw_losses["sup"].device)

        total = (
            lambda_sup * raw_losses["sup"]
            + lambda_phys * raw_losses["phys"]
            + lambda_ic * raw_losses["ic"]
        )

        stats = {
            "total_loss": total.detach(),
            "lambda_sup": lambda_sup.detach(),
            "lambda_phys": lambda_phys.detach(),
            "lambda_ic": lambda_ic.detach(),
            "grad_sup": g_sup.detach(),
            "grad_phys": g_phys.detach(),
            "grad_ic": g_ic.detach(),
        }
        return total, stats