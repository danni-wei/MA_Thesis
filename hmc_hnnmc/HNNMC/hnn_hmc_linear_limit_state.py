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
from HNNMC.hnnmc.nn_models import MLP
from HNNMC.hnnmc.hnn import HNN
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
#beta = 5.0 #!!!!

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
        #alpha = np.minimum(1, np.exp(-beta * (H_star - H_prev)))
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

# ==== NumPy-based ESS (no TFP needed) ====
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

n_dim = 2

ess_hnn = np.zeros((chains, n_dim))
for ss in np.arange(0, chains, 1):
    post = hnn_fin[ss, burn:N, :]   # [T, n_dim]
    ess_hnn[ss, :] = np.array([ess_1d(post[:, d]) for d in range(n_dim)])
print("ESS (numpy):", ess_hnn[0])


chain_idx = 0
burn = int(burn)  # reuse your existing burn
samples = hnn_fin[chain_idx, burn:N, :]   # [T, d] —— 只画每步终点
d = samples.shape[1]


# 1️⃣ 基本散点图（你已经有的）
x, y = samples[:, 0], samples[:, 1]
#plt.figure(figsize=(7,6))
plt.figure(figsize=(6,6))
plt.scatter(x, y, s=5, c='k', alpha=0.6, label='HNN-HMC samples')

# 2️⃣ 画出背景等高线（标准高斯）
xx, yy = np.meshgrid(np.linspace(-4, 4, 200), np.linspace(-4, 4, 200))
ZZ = np.exp(-0.5*(xx**2 + yy**2))
plt.contour(xx, yy, ZZ, levels=15, cmap='Blues', linewidths=0.8)

# 3️⃣ 定义并绘制线性极限状态线 g(x) = aᵀx - b = 0
a = np.array([1.0, 1.0])   # 方向向量（论文里常用 [1,1]）
b = 5.0                    # 截距，根据你想画的位置调节

# g(x,y) = 0 => y = (b - a1*x)/a2
x_line = np.linspace(-4, 4, 100)
y_line = (b - a[0]*x_line) / a[1]

plt.plot(x_line, y_line, 'r--', lw=2.0, label='g(x)=0 limit state')

# 4️⃣ 样式优化
plt.xlabel("x₁"); plt.ylabel("x₂")
plt.title("HNN-HMC samples (ρ=0) with Linear Limit State")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




'''# 2D 可视（前两维）
if d >= 2:
    x, y = samples[:, 0], samples[:, 1]

    # 背景等高线（标准高斯 ρ=0）
    xx, yy = np.meshgrid(np.linspace(-4, 4, 220), np.linspace(-4, 4, 220))
    ZZ = np.exp(-0.5 * (xx**2 + yy**2))  # 未归一化即可用于可视

    plt.figure(figsize=(7, 6))
    plt.contour(xx, yy, ZZ, levels=20, cmap="Blues", linewidths=0.8)
    # 轨迹: 细灰线（样本之间连线）
    plt.plot(x, y, '-', color='0.7', lw=0.8, alpha=0.6, label='trace')
    # 终点云：黑色小点
    plt.scatter(x, y, s=5, c='k', alpha=0.75, label='samples')
    # 起止高亮
    plt.scatter([x[0]], [y[0]], c='red', s=28, label='start')
    plt.scatter([x[-1]], [y[-1]], c='green', s=28, label='end')

    # KPI 抬头（可读性拉满）
    try:
        acc_rate = float(np.mean(hnn_accept[chain_idx, burn:]))
        title_kpi = f"acc={acc_rate:.2f}, N={samples.shape[0]}"
    except Exception:
        title_kpi = f"N={samples.shape[0]}"
    plt.title(f"HNN-HMC samples (ρ=0)  |  {title_kpi}")
    plt.xlabel("q₁"); plt.ylabel("q₂"); plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("hnn_hmc_rho0_scatter.png", dpi=200)
    plt.show()

# 1D 直方图（若 d>2，也先看前两维的边际分布）
plt.figure(figsize=(7, 3))
plt.subplot(1,2,1); plt.hist(samples[:,0], bins=40, density=True, alpha=0.8)
plt.title("q₁ marginal"); plt.grid(True, alpha=0.3)
plt.subplot(1,2,2); plt.hist(samples[:,1], bins=40, density=True, alpha=0.8)
plt.title("q₂ marginal"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig("hnn_hmc_rho0_hist.png", dpi=200); plt.show()'''

# 持久化样本，便于复用
try:
    np.save("hnn_fin.npy", hnn_fin)
except Exception:
    pass
# ==== end viz ====