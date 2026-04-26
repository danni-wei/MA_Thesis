# file: mh_simple.py
import numpy as np
from typing import Callable, Optional, Tuple

Array = np.ndarray

class MH:
  """
  general Metropolis–Hastings 。
  - energy(X): [1, batch], E(x) = -log π(x)
  - propose(X): (X_prop, log_q_prop_given_curr, log_q_curr_given_prop), [d,B], [1,B], [1,B]
  - feasible: optional: feasible(X_col[dim,1])->bool (for SuS, g(x)≤Lk)
  """
  def __init__(
    self,
    Xinit: Array,
    energy: Callable[[Array], Array],
    propose: Callable[[Array], Tuple[Array, Array, Array]],
    feasible: Optional[Callable[[Array], bool]] = None,
    display: int = 1,
  ):
    self.X = Xinit.copy()           # [d, B]
    self.energy_fn = energy
    self.propose_fn = propose
    self.feasible = feasible
    self.display = display

    self.d, self.B = self.X.shape

  # energy function
  def energy(self, X: Array) -> Array:
    return self.energy_fn(X).reshape(1, -1)

  # single step of MH
  def step(self) -> float:
    X0 = self.X
    E0 = self.energy(X0)            # [1,B]

    Xp, log_q_p_c, log_q_c_p = self.propose_fn(X0)  # [d,B], [1,B], [1,B]

    # SuS 硬约束（逐列检查，不可行→强制拒绝）
    if self.feasible is not None:
      ok = np.ones(self.B, dtype=bool)
      for j in range(self.B):
        if not self.feasible(Xp[:, [j]]):
          ok[j] = False
      # 将不可行提议替换回原状态（并使接受概率=0）
      Xp[:, ~ok] = X0[:, ~ok]
      log_q_p_c[:, ~ok] = 0.0
      log_q_c_p[:, ~ok] = 0.0

    E1 = self.energy(Xp)            # [1,B]

    # MH acceptance :min(1, exp( -E1 + E0 + logq(c|p) - logq(p|c) ))
    log_alpha = -(E1 - E0) + (log_q_c_p - log_q_p_c)      # [1,B]
    logu = np.log(np.random.rand(1, self.B))
    accept = (logu < log_alpha).ravel()

    # accept / rejected
    self.X[:, accept] = Xp[:, accept]

    return float(np.mean(accept))

  # main loop
  def sample(self, num_steps: int = 1000, return_trace: bool = False) -> Array:
    acc_hist, trace = [], []
    for _ in range(num_steps):
      acc_hist.append(self.step())
      if return_trace:
        trace.append(self.X.copy())
    if self.display:
      print(f"[MH] steps={num_steps}  acc={np.mean(acc_hist):.2f}")
    return (np.stack(trace, axis=0) if return_trace else self.X.copy())


class RWMH(MH):
  """
  Random Walk MH for bivariate non-Gaussian: x' = x + N(0, Σ)
  """
  def __init__(
    self,
    Xinit: Array,
    energy: Callable[[Array], Array],
    cov: Optional[Array] = None,     # None -> Identity
    feasible: Optional[Callable[[Array], bool]] = None,
    display: int = 1,
    rng: Optional[np.random.Generator] = None,
  ):
    self.rng = rng if rng is not None else np.random.default_rng()

    d, B = Xinit.shape
    if cov is None:
      self.chol = None  # Identity
    else:
      cov = np.asarray(cov, dtype=float)
      if cov.ndim == 1:
          cov = np.diag(cov)
      self.chol = np.linalg.cholesky(cov)  # positive defined

    def _propose(X: Array) -> Tuple[Array, Array, Array]:
      if self.chol is None:
        Z = self.rng.normal(size=X.shape)              # [d,B]
        Xp = X + Z
      else:
        Z = self.rng.normal(size=X.shape)              # [d,B]
        Xp = X + (self.chol @ Z)                       # [d,B]
      # log q(p|c) = log q(c|p)
      zeros = np.zeros((1, X.shape[1]))
      return Xp, zeros, zeros

    super().__init__(Xinit=Xinit, energy=energy, propose=_propose,
                      feasible=feasible, display=display)