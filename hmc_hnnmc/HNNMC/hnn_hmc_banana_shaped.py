# Copyright (c) 2022 Battelle Energy Alliance, LLC
# Licensed under MIT License, please see LICENSE for details
# https://github.com/IdahoLabResearch/BIhNNs/blob/main/LICENSE

# Coded by Som Dhulipala at Idaho National Laboratory
# Hamiltonian Monte Carlo with HNNs

import torch, sys
import autograd.numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
#import tensorflow as tf
#import tensorflow_probability as tfp
from hnnmc.nn_models import MLP
from hnnmc.hnn import HNN
from scipy.stats import norm
from scipy.stats import uniform
from hnnmc.get_args import get_args
from hnnmc.utils import leapfrog
from hnnmc.functions import functions
args = get_args()

##### User-defined sampling parameters #####

chains = 1 # number of Markov chains
N = 100 # number of samples   
L = 10 # length of each hamiltonian trajectory
burn = 0 # burn-in samples
epsilon = 0.025 # step for time integration

##### Sampling code below #####

y0 = np.zeros(args.input_dim)
def get_model(args, baseline):
    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
              field_type=args.field_type, baseline=baseline)
    path = args.dist_name + ".tar"
    model.load_state_dict(torch.load(path))
    return model

def integrate_model(model, t_span, y0, n, **kwargs):
    def fun(t, np_x):
        x = torch.tensor( np_x, requires_grad=True, dtype=torch.float32).view(1,args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

hnn_model = get_model(args, baseline=False)

steps = L*int(1/epsilon)
t_span = [0,L]
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}
hnn_fin = np.zeros((chains,N,int(args.input_dim/2)))
hnn_accept = np.zeros((chains,N))
for ss in np.arange(0,chains,1):
    x_req = np.zeros((N,int(args.input_dim/2)))
    x_req[0,:] = y0[0:int(args.input_dim/2)]
    accept = np.zeros(N)
    
    for ii in np.arange(0,int(args.input_dim/2),1):
        y0[ii] = 0.0
    for ii in np.arange(int(args.input_dim/2),int(args.input_dim),1):
        y0[ii] = norm(loc=0,scale=1).rvs()
    HNN_sto = np.zeros((args.input_dim,steps,N))
    for ii in np.arange(0,N,1):
        hnn_ivp = integrate_model(hnn_model, t_span, y0, steps-1, **kwargs)
        for sss in range(0,args.input_dim):
            HNN_sto[sss,:,ii] = hnn_ivp[sss,:]
        yhamil = np.zeros(args.input_dim)
        for jj in np.arange(0,args.input_dim,1):
            yhamil[jj] = hnn_ivp[jj,steps-1]
        H_star = functions(yhamil)
        H_prev = functions(y0)
        alpha = np.minimum(1,np.exp(H_prev - H_star))
        if alpha > uniform().rvs():
            y0[0:int(args.input_dim/2)] = hnn_ivp[0:int(args.input_dim/2),steps-1]
            x_req[ii,:] = hnn_ivp[0:int(args.input_dim/2),steps-1]
            accept[ii] = 1
        else:
            x_req[ii,:] = y0[0:int(args.input_dim/2)]
        for jj in np.arange(int(args.input_dim/2),args.input_dim,1):
            y0[jj] = norm(loc=0,scale=1).rvs()
        print("Sample: "+str(ii)+" Chain: "+str(ss))
    hnn_accept[ss,:] = accept
    hnn_fin[ss,:,:] = x_req

import numpy as np

def ess_1d(x):
    """x: [T]  单变量 ESS 估计（自相关截尾法）"""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 1e-20:
        return 1.0
    acf = np.correlate(x, x, mode='full')[T-1:] / denom
    s = 0.0
    for k in range(1, T-1, 2):
        pair = acf[k] + acf[k+1]
        if pair <= 0:
            break
        s += pair
    return max(1.0, T / (1 + 2*s))

def ess_matrix(X):
    """X: [T, D]，逐维 ESS"""
    X = np.asarray(X)
    return np.array([ess_1d(X[:, d]) for d in range(X.shape[1])])

# 计算 ESS and visualize
chain_idx = 0
traj = hnn_fin[chain_idx, burn:, :]   # [T,2]
x, y = traj[:, 0], traj[:, 1]
post = hnn_fin[chain_idx, burn:, :]   # [T, 2]
ess_est = ess_matrix(post)
print("ESS (numpy):", ess_est)

# 2) banana 等高线（exp(-U)）
def make_banana(a=1.15, b=0.5, rho=0.9):
    Sigma_inv = (1.0/(1.0 - rho**2)) * np.array([[1.0, -rho], [-rho, 1.0]])
    def _u(q): xq,yq = q[...,0], q[...,1]; return np.stack([a*xq, a*(yq - b*xq*xq)], axis=-1)
    def U(q):  u=_u(q); return 0.5*np.einsum('...i,ij,...j->...', u, Sigma_inv, u)
    return U
U = make_banana()

xx, yy = np.meshgrid(np.linspace(-5, 5, 240), np.linspace(-5, 8, 280))
grid = np.vstack([xx.ravel(), yy.ravel()]).T
ZZ = np.exp(-U(grid)).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contour(xx, yy, ZZ, levels=25, cmap="Blues", linewidths=0.8)

# 3) 叠加轨迹与终点
plt.plot(x, y, '-', color='0.6', lw=0.8, alpha=0.6, label='trace')      # 轨迹（灰细线）
plt.scatter(x, y, s=3, c='k', alpha=0.8, label='endpoints')              # 每步终点（黑小点）
plt.scatter([x[0]],  [y[0]],  c='red',   s=28, label='start')            # 起点
plt.scatter([x[-1]], [y[-1]], c='green', s=28, label='end')              # 终点

# 4) 可选：仅高亮“被接受”的步（绿点更亮）
try:
    acc_mask = (hnn_accept[chain_idx, burn:] > 0.5)
    plt.scatter(x[acc_mask], y[acc_mask], s=6, facecolors='none', edgecolors='lime', linewidths=0.8, label='accepted')
except Exception:
    pass

# 5) 装饰 & 导出
try:
    acc_rate = float(np.mean(hnn_accept[chain_idx, burn:]))
    plt.title(f"HNN-HMC on Banana  |  acc={acc_rate:.2f}  |  N={traj.shape[0]}")
except Exception:
    plt.title(f"HNN-HMC on Banana  |  N={traj.shape[0]}")

plt.xlabel("x"); plt.ylabel("y"); plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("banana_hnnhmc_trace.png", dpi=200)
plt.show()
# ==== end viz ====
'''
ess_hnn = np.zeros((chains,int(args.input_dim/2)))

for ss in np.arange(0,chains,1):
    hnn_tf = tf.convert_to_tensor(hnn_fin[ss,burn:N,:])
    ess_hnn[ss,:] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
'''