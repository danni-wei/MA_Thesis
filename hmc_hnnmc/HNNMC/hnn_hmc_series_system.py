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
from nn_models import MLP
from hnn import HNN
from scipy.stats import norm
from scipy.stats import uniform
from HNNMC.hnnmc.get_args import get_args
from HNNMC.hnnmc.utils import leapfrog
from HNNMC.hnnmc.functions import functions
args = get_args()

##### User-defined sampling parameters #####

chains = 1 # number of Markov chains
N = 1000 # number of samples   
L = 10 # length of each hamiltonian trajectory
burn = 100 # burn-in samples
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

n_dim = int(args.input_dim // 2)
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

'''
ess_hnn = np.zeros((chains,int(args.input_dim/2)))
for ss in np.arange(0,chains,1):
    hnn_tf = tf.convert_to_tensor(hnn_fin[ss,burn:N,:])
    ess_hnn[ss,:] = np.array(tfp.mcmc.effective_sample_size(hnn_tf))
'''

def ess_1d(x):
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

ess = np.array([ess_1d(hnn_fin[0, burn:N, d]) for d in range(n_dim)])
acc_rate = float(np.mean(hnn_accept[0, burn:]))

print(f"[diagnostics] ESS per dim: {np.round(ess, 1)}")
print(f"[diagnostics] post-burn acceptance: {acc_rate:.2f}")

# Series system evaluation
samples = hnn_fin[0, burn:N, :2]     # 只用前两维 x=(x1,x2) —— 本例为2D系统
x1, x2 = samples[:, 0], samples[:, 1]
u = x1 - x2
v = (x1 + x2)/np.sqrt(2.0)

g1 = 0.1*(u**2) - v + 3.0
g2 = 0.1*(u**2) + v + 3.0
g3 = u + 7.0/np.sqrt(2.0)
g4 = -u + 7.0/np.sqrt(2.0)

G_all = np.vstack([g1, g2, g3, g4]).T
G_sys = np.min(G_all, axis=1)             # 串联系统：min
fail  = (G_sys <= 0)
Pf_hat   = float(np.mean(fail))
beta_hat = -N.ppf(Pf_hat) if Pf_hat > 0 else np.inf

print(f"[series system] P_F^ = {Pf_hat:.6e} , beta^ = {beta_hat:.3f}")

# Persist & visualize
# -------------------------
np.save("hnn_fin.npy", hnn_fin)
np.save("hnn_accept.npy", hnn_accept)

# 背景等高线（标准高斯）
xx, yy = np.meshgrid(np.linspace(-6,6,240), np.linspace(-6,6,240))
ZZ = np.exp(-0.5*(xx**2 + yy**2))

plt.figure(figsize=(7,6))
plt.contour(xx, yy, ZZ, levels=18, cmap="Blues", linewidths=0.7)
plt.scatter(x1, x2, s=5, c='k', alpha=0.65, label='samples')

# 画 g1=0, g2=0（用 u 参数化后映回 x1,x2）
u_grid = np.linspace(-10, 10, 800)
v1 = 0.1*(u_grid**2) + 3.0
v2 = -0.1*(u_grid**2) - 3.0
x1_g1 = 0.5*(u_grid + np.sqrt(2)*v1); x2_g1 = 0.5*(np.sqrt(2)*v1 - u_grid)
x1_g2 = 0.5*(u_grid + np.sqrt(2)*v2); x2_g2 = 0.5*(np.sqrt(2)*v2 - u_grid)
plt.plot(x1_g1, x2_g1, 'r-',  lw=2.0, label='g1(x)=0')
plt.plot(x1_g2, x2_g2, 'r--', lw=2.0, label='g2(x)=0')

# 画 g3=0, g4=0（直线：u=const -> x2=x1 - u）
u3 = -7.0/np.sqrt(2.0)
u4 = +7.0/np.sqrt(2.0)
x_line = np.linspace(-6, 6, 240)
plt.plot(x_line, x_line - u3, 'g-',  lw=2.0, label='g3(x)=0')
plt.plot(x_line, x_line - u4, 'g--', lw=2.0, label='g4(x)=0')

plt.title(f"Series system over HNN-HMC samples | P_F^={Pf_hat:.2e}, beta^={beta_hat:.2f}, acc={acc_rate:.2f}")
plt.xlabel("x1"); plt.ylabel("x2"); plt.grid(True, alpha=0.3)
plt.legend(fontsize=9); plt.tight_layout()
plt.savefig("series_system_linear.png", dpi=200)
plt.show()

# --- End of file ---