from hmc import HMC
import numpy as np
import matplotlib.pyplot as plt

# define the pdf of the banana shaped distribution
def make_banana(a = 7, b = 0.75, rho = 0.3):
  #def make_banana(a = 1.15, b = 0.5, rho = 0.9):
  Sigma_inv = (1.0/(1.0-rho**2)) * np.array([[1.0, -rho], [-rho, 1.0]])

  def _u(X): #X:[1, batch]
    x, y = X[0,:], X[1,:]
    u1 = a * x 
    u2 = a * (y - b * x * x)
    return np.vstack([u1, u2]) #[2, batch]
  
  def E(X):
    U = _u(X)
    # energy = -logpi = 0.5 * u^T Sigma_inv u
    quad = np.einsum('ib,ij,jb->b', U, Sigma_inv, U)
    return 0.5 * quad.reshape(1, -1)
  
  def dEdX(X):
    x, y = X[0, :], X[1, :]
    U = _u(X)
    g_u = (Sigma_inv @ U) #d(energy)/du = Sigma_inv u
    du_dx = np.vstack([np.full_like(x, a), -2*a*b*x]) # shape [2, batch]
    du_dy = np.vstack([np.zeros_like(y), np.full_like(y, a)])
    dE_dx = np.einsum('ib,ib->b', g_u, du_dx)
    dE_dy = np.einsum('ib,ib->b', g_u, du_dy)
    return np.vstack([dE_dx, dE_dy])

  return E, dEdX

# implement HMC on banana_shaped
E, dEdX = make_banana(a = 7, b = 0.75, rho = 0.3)
Xinit = np.array([[4.0],[5.0]]) #[2,1]
sampler = HMC(Xinit, E, dEdX, epsilon = 0.05, num_leapfrog_steps=21, display=1)

# visualize the original distribution with meshgrid linspace
xx, yy = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
grid = np.vstack([xx.ravel(), yy.ravel()])
EE = E(grid).reshape(-1)
ZZ = np.exp(-EE).reshape(xx.shape)
plt.figure(figsize=(7,6))
plt.contour(xx, yy, ZZ, levels=25, cmap="Blues")

N = 1000  # number of steps
accepted_cnt = 0
for t in range(N):
    acc, path, accepted = sampler.step(record_path=True)
    curve = path[:, :, 0]            # [L+1, 2]
    color = "tab:green" if accepted else "0.6"  # green = accepted；gray = rejected
    lw = 1.4 if accepted else 0.8
    alpha = 0.9 if accepted else 0.6
    plt.plot(curve[:,0], curve[:,1], '-', lw=lw, alpha=alpha, color=color)
    plt.scatter(curve[-1,0], curve[-1,1], c="black", s=1, alpha=0.8, zorder=3)
    accepted_cnt += int(accepted)

# mark the current sample point as red
curr = sampler.X[:, 0]
plt.scatter([curr[0]], [curr[1]], c="red", s=35, label=f"current (acc {accepted_cnt}/{N})")

plt.xlabel("x"); plt.ylabel("y")
plt.title("Leapfrog paths per HMC step on banana-shaped target")
plt.legend()
plt.grid(True); plt.tight_layout(); plt.show()


'''
Xs = sampler.sample(num_steps = 1000, return_trace = True) # return the final position
print("final:", Xs.ravel())
print("Xs shape =", Xs.shape)
print("unique positions =", np.unique(Xs.reshape(Xs.shape[0], -1), axis=0).shape[0])

# visualization
traj = Xs[:, :, 0]   # shape [steps, 2]
x, y = traj[:, 0], traj[:, 1]
plt.figure(figsize=(6, 5))
plt.plot(x, y, 'o-', lw=1, ms=3, alpha=0.7)
plt.title("HMC trajectory on Banana-shaped distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show() 
'''