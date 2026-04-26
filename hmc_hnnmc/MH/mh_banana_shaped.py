import numpy as np
import matplotlib.pyplot as plt
from mh import RWMH

# E(X) = -log π(X), [1,B]
def make_banana_energy(a=1.15, b=0.5, rho=0.9):
    Sigma_inv = (1.0/(1.0 - rho**2)) * np.array([[1.0, -rho], [-rho, 1.0]])
    def _u(X):
        x, y = X[0,:], X[1,:]
        return np.vstack([a*x, a*(y - b*x*x)])
    def E(X):
        U = _u(X)
        quad = np.einsum('ib,ij,jb->b', U, Sigma_inv, U)
        return 0.5 * quad.reshape(1, -1)
    return E

E = make_banana_energy()
Xinit = np.array([[4.0], [5.0]])   # [2,1]

# step size
cov = np.array([0.10, 0.10])       
sampler = RWMH(Xinit, energy=E, cov=cov, display=1)

Xs = sampler.sample(num_steps=100, return_trace=True)   # [T,2,1]
traj = Xs[:, :, 0]
x, y = traj[:,0], traj[:,1]

# visualize, scatter and trace
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 8, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])
EE = E(grid).reshape(-1)
ZZ = np.exp(-EE).reshape(xx.shape)

plt.figure(figsize=(7,6))
plt.contour(xx, yy, ZZ, levels=25, cmap="Blues")
plt.plot(x, y, '-', color='green', lw=0.8, alpha=0.6, label='MH trace')
plt.scatter(x, y, s=2, c='k', alpha=0.6, label='MH samples')
plt.scatter([x[0]], [y[0]], c='red', s=25, label='start')
plt.xlabel("x"); plt.ylabel("y")
plt.title("Random-Walk MH on banana-shaped target")
plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()