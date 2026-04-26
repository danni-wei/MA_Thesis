'''
10.10.2025 Danni Wei
HMC for Subset Simulation Implementation
'''

import numpy as np

class HMC:
  """
  Generic Hamiltonian Monte Carlo sampler.
  - target_log_prob: log π(q)
  - target_grad_log:  ∇ log π(q) 
  - step_size:        leapfrog step size ε
  - n_steps:          leapfrog steps L (trajectory length ~ ε*L)
  - mass_matrix:      positive-definite matrix M (shape d*d). If None, use I.
  - feasible:         optional constraint checker: feasible(q) -> bool
                      (e.g., for Subset Simulation enforce g(q) <= L_k)
  """
  def __init__(self, Xinit, E, dEdX, epsilon=0.05, num_leapfrog_steps=21,
                 mass_matrix=None, feasible=None, display=1, args=(), kwargs={}):
    """
    Standard HMC method.
    Xinit: [dim, batch] The initial particle positions for sampling. 
    E(X): [1, batch] Function that returns the energy. 
    dEdX(X): [dim, batch] Function that returns the gradient.
    feasible: optional, Function(X_col[dim,1])->bool
    """

    self.X = Xinit.copy()   # [d, B]
    self.E_fn = E
    self.grad_fn = dEdX
    self.args, self.kwargs = args, kwargs
    self.eps = float(epsilon)
    self.M = int(num_leapfrog_steps)
    self.feasible = feasible
    self.display = display
    self.d, self.B = self.X.shape
    
    # initialize mass matrix
    if mass_matrix is None:
      self.Mmat = None
      self.Minv = None
      self.chol = None
    else:
      M = np.array(mass_matrix, dtype=float)
      if M.ndim == 1:
        M = np.diag(M)
      self.chol = np.linalg.cholesky(M)
      self.Mmat = M
      self.Minv = np.linalg.inv(M)

    '''
    # initialize 缓存
    self.V = np.random.randn(self.d, self.B)
    self.EX = self.E_fn(self.X)
    self.grad = self.grad_fn(self.X)
    self.EV = 0.5*np.sum(self.V**2, axis=0, keepdims=True) if self.Mmat is None \
                  else 0.5*np.sum(self.V * (self.Minv @ self.V), axis=0, keepdims=True)
    '''

  def E(self, X):
    ''' compute the energy function at X '''
    return self.E_fn(X,*self.args, **self.kwargs).reshape((1,-1))
    
  def grad(self, X):
    ''' compute the energy function gradint at X '''
    return self.grad_fn(X, *self.args, **self.kwargs)
    
  def _leapfrog_once(self, X, V):
    ''' singale leapfrog step for X and V '''
    g = self.grad(X)
    V = V - 0.5 * self.eps * g
    if self.Mmat is None:
      X = X + self.eps * V
    else:
      X = X + self.eps * (self.Minv @ V)
    g = self.grad(X)
    V = V - 0.5 * self.eps * g
    return X, V
    
  def step(self, record_path = False):
    ''' Run the leapfrog operator for M leapfrog steps '''
    d, B = self.d, self.B
    if self.Mmat is None:
      V0 = np.random.randn(d, B)
      kin = lambda V: 0.5 * np.sum(V ** 2, axis = 0, keepdims = True)
    else:
      Z = np.ramdom.randn(d,B)
      V0 = self.chol @ Z
      kin = lambda V: 0.5 * np.sum(V * (self.Minv @ V), axis = 0, keepdims = True)

    X0 = self.X.copy()
    V = V0.copy()
    ok = np.ones(B, dtype=bool)
    
    # M_th leapfrog
    Xp, Vp = X0.copy(), V.copy()
    path = [Xp.copy()] if record_path else None

    # the first half setp already did in _leapfrog_once, here noly need to loop M times
    for _ in range(self.M):
      Xp, Vp = self._leapfrog_once(Xp, Vp)
      if record_path:
        path.append(Xp.copy())
    '''
    for _ in range(self.M):
      Xp, Vp = self._leapfrog_once(Xp, Vp)
      if record_path:
          # 只记录可行点
          ok_vis = True
          if self.feasible is not None:
              # 仅检查第一列用于可视化
              if not self.feasible(Xp[:, [0]]):
                  ok_vis = False
          if ok_vis:
              path.append(Xp.copy())
          else:
              break  # 越界后停止追加，避免“穿心”的可视化
    '''
    Vp = -Vp

    # optional for feasibility check
    if self.feasible is not None:
      ok = np.ones(B, dtype=bool)
      for j in range(B):
        if not self.feasible(Xp[:, [j]]):
          ok[j] = False
      # if not appliable then reject and return back to X0
      Xp[:, ~ok] = X0[:, ~ok]
      Vp[:, ~ok] = -V0[:, ~ok]
      
    # Metropolis Hastings
    H0 = self.E(X0) + kin(V0) #[1,B]
    H1 = self.E(Xp) + kin(Vp) #[1,B]
    logu = np.log(np.random.rand(1,B))
    accept = (logu < (H0 - H1)).ravel() #[B]
    accept = accept & ok

    # apply accept/reject
    self.X[:, accept] = Xp[:, accept]
    self.V = Vp

    acc_rate = float(np.mean(accept))
    if record_path:
      return acc_rate, np.stack(path, axis = 0), bool(accept[0])
    return acc_rate
    
  def sample(self, num_steps = 100, return_trace = True):
    acc_hist = []
    trace = []
    for t in range(num_steps):
      a = self.step()
      acc_hist.append(a)
      if return_trace: trace.append(self.X.copy())
    if self.display:
      print(f"[HMC]  steps = {num_steps}  acc = {np.mean(acc_hist):.2f}")
    return (np.stack(trace, axis = 0) if return_trace else self.X.copy())

    