# SuS_HNN_banana.py — Subset Simulation with HNN-HMC on a banana-shaped failure band
import numpy as np
import torch, os
import matplotlib.pyplot as plt
from scipy.stats import norm as Nrm, uniform

from hnnmc.nn_models import MLP
from hnnmc.hnn import HNN
from hnnmc.get_args import get_args
from hnnmc.utils import leapfrog
from hnnmc.functions import functions  # H(y) = U(q) + 0.5||p||^2  (你的HNN-HMC哈密顿量)

# ===== Ops & viz toggles =====
LOG_EVERY   = 100
VIS_LEVEL   = True
OUTDIR      = "sus_hnn_banana_figs"
os.makedirs(OUTDIR, exist_ok=True)

# ===== Banana transform & quadratic metric =====
# u = [a x, a (y - b x^2)],  metric = u^T Σ^{-1} u
def make_banana(a=1.15, b=0.5, rho=0.9):
    # Σ^{-1} for the 2D Gaussian in u-space (correlation rho)
    SInv = (1.0 / (1.0 - rho**2)) * np.array([[1.0,   -rho],
                                              [-rho,  1.0]])
    def Uquad(q):
        x, y = q[..., 0], q[..., 1]
        u1 = a * x
        u2 = a * (y - b * x * x)
        # return u^T Σ^{-1} u  （不乘0.5，方便当作“半径平方”使用）
        return (u1 * (SInv[0,0]*u1 + SInv[0,1]*u2) +
                u2 * (SInv[1,0]*u1 + SInv[1,1]*u2))
    return Uquad

# ===== Failure: banana-band  t_in <= Uquad(x) <= t_out  =====
# 类比圆环：把“半径平方”换成 Uquad(x)
A, B, RHO = 1.15, 0.5, 0.9
Uquad = make_banana(A, B, RHO)

# 你可以按需调这两个阈值；默认给个中等“稀有度”的带宽
T_IN, T_OUT = 6.0, 14.0   # 失效域: T_IN <= Uquad <= T_OUT

def G_banana(q):
    val = Uquad(q)
    f = val - T_OUT       # <=0 => Uquad <= T_OUT
    g = T_IN - val        # <=0 => Uquad >= T_IN
    return max(f, g)      # <=0 => T_IN <= Uquad <= T_OUT

# ===== HNN model (与 hnn_hmc.py 一致) =====
args = get_args()
args.input_dim = 4           # 2D位置 + 2D动量
d = args.input_dim // 2

def get_model(args, baseline=False):
    nn_model = MLP(args.input_dim, args.hidden_dim, args.input_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=baseline)
    path = args.dist_name + ".tar"   # e.g., 'nD_standard_Gaussian.tar'
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def integrate_model(model, t_span, y0, n):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

hnn_model = get_model(args, baseline=False)

def hnn_hmc_step(y, t_span, steps):
    y_traj = integrate_model(hnn_model, t_span, y, steps-1)  # [4, steps]
    y_prop = y_traj[:, -1]
    H_prev = functions(y)
    H_star = functions(y_prop)
    alpha  = np.minimum(1.0, np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        return y_prop, True
    else:
        y_rej = y.copy()
        y_rej[d:] = np.random.randn(d)   # 刷新动量
        return y_rej, False

# ===== SuS hyper-params =====
N0   = 1000
p0   = 0.10
Jmax = 12
L = 10
epsilon = 0.025
steps  = int(L/epsilon)
t_span = [0, L]
assert (N0*p0).is_integer() and (1/p0).is_integer(), "N0*p0 与 1/p0 必须是整数"

# ===== Level 0: unconstrained HNN-HMC =====
samples0 = np.zeros((N0, d))
acc_hist0 = []
y = np.zeros(args.input_dim)
y[:d] = 0.0
y[d:] = np.random.randn(d)

for i in range(N0):
    y, acc = hnn_hmc_step(y, t_span, steps)
    samples0[i, :] = y[:d]
    acc_hist0.append(1.0 if acc else 0.0)
    if (i+1) % LOG_EVERY == 0:
        print(f"[level 0] progress: {i+1}/{N0}")

G0  = np.array([G_banana(samples0[i]) for i in range(N0)])
b1  = np.quantile(G0, p0)
idx = np.argsort(G0)[:int(p0*N0)]
seeds = samples0[idx]
print(f"[SuS-HNN banana] level 0 done. b1={b1:.3f}, acc={np.mean(acc_hist0):.2f}")

def draw_level(level_idx, X, title_note=""):
    # 背景：画出 Uquad 等值线在 two levels (T_IN, T_OUT)
    xx, yy = np.meshgrid(np.linspace(-6, 6, 300), np.linspace(-6, 6, 300))
    grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
    UU = np.array([Uquad(q) for q in grid]).reshape(xx.shape)

    plt.figure(figsize=(6.5,6))
    CS1 = plt.contour(xx, yy, UU, levels=[T_IN], colors='g', linestyles='--')
    CS2 = plt.contour(xx, yy, UU, levels=[T_OUT], colors='r', linestyles='-')
    plt.clabel(CS1, inline=True, fontsize=8)
    plt.clabel(CS2, inline=True, fontsize=8)
    plt.scatter(X[:,0], X[:,1], s=6, c='k', alpha=0.6, label='samples')
    plt.legend(loc='upper right', fontsize=9)
    plt.xlabel('x1'); plt.ylabel('x2'); plt.axis('equal')
    plt.title(f"SuS-HNN banana level {level_idx} {title_note}")
    plt.grid(True, alpha=.3); plt.tight_layout()
    fname = os.path.join(OUTDIR, f"level_{level_idx}_samples.png")
    plt.savefig(fname, dpi=200); plt.close()
    print(f"✓ saved figure: {fname}")

if VIS_LEVEL:
    os.makedirs(OUTDIR, exist_ok=True)
    draw_level(0, samples0, title_note=f"| b1={b1:.3f}, acc0={np.mean(acc_hist0):.2f}")

# ===== Subsequent levels with constraint {G<=b_j} =====
levels = [dict(b=b1, acc=None)]
b_j = b1
cur  = 1

while (b_j > 0) and (cur < Jmax):
    Sj = np.zeros((N0, d))
    acc_hist = []
    chains = seeds.shape[0]
    Lchain = int(np.ceil(N0 / chains))
    k = 0

    for s in range(chains):
        y = np.zeros(args.input_dim)
        y[:d] = seeds[s]
        y[d:] = np.random.randn(d)
        for _ in range(Lchain):
            if k >= N0: break
            # 一次 HNN-HMC 轨迹并做子集筛选
            y_prop_traj = integrate_model(hnn_model, t_span, y, steps-1)
            y_prop = y_prop_traj[:, -1]
            q_prop = y_prop[:d]

            if G_banana(q_prop) > b_j:
                y[d:] = np.random.randn(d)
                acc = False
            else:
                H_prev = functions(y)
                H_star = functions(y_prop)
                alpha  = np.minimum(1.0, np.exp(H_prev - H_star))
                if alpha > uniform().rvs():
                    y = y_prop
                    acc = True
                else:
                    y[d:] = np.random.randn(d)
                    acc = False

            Sj[k, :] = y[:d]
            acc_hist.append(1.0 if acc else 0.0)
            k += 1
            if (k % LOG_EVERY) == 0:
                print(f"[level {cur}] progress: {k}/{N0}")

        if k >= N0: break

    Gj = np.array([G_banana(Sj[i]) for i in range(N0)])
    b_next = np.quantile(Gj, p0)
    levels[-1]['acc'] = float(np.mean(acc_hist))
    levels.append(dict(b=b_next, acc=None))
    print(f"[SuS-HNN banana] level {cur} done. b{cur+1}={b_next:.3f}, acc={np.mean(acc_hist):.2f}")

    if VIS_LEVEL:
        note = f"| b{cur+1}={b_next:.3f}, acc={np.mean(acc_hist):.2f}"
        draw_level(cur, Sj, title_note=note)

    idx = np.argsort(Gj)[:int(p0*N0)]
    seeds = Sj[idx]
    b_j = b_next
    cur += 1

# ===== Final estimates =====
J = len(levels) - 1
final_set = seeds if b_j <= 0 else Sj
G_final = np.array([G_banana(q) for q in final_set])
nJ = int(np.sum(G_final <= 0.0))
Pf_hat = (p0**(J-1)) * (nJ / N0) if J >= 1 else (nJ / N0)
beta_hat = -Nrm.ppf(Pf_hat) if Pf_hat > 0 else np.inf

print(f"[SuS-HNN banana] J={J}, nJ={nJ}/{N0},  Pf^={Pf_hat:.3e},  beta^={(beta_hat if np.isfinite(beta_hat) else float('inf')):.2f}")
for j in range(0, J):
    print(f"  level {j+1}: b={levels[j]['b']:.3f}, acc={levels[j]['acc']:.2f}")

np.savez(os.path.join(OUTDIR, "sus_hnn_banana_outputs.npz"),
         Pf_hat=Pf_hat, beta_hat=beta_hat,
         levels_b=np.array([lv['b'] for lv in levels[:-1]]),
         acc=np.array([lv['acc'] for lv in levels[:-1]]),
         T_IN=T_IN, T_OUT=T_OUT, a=A, b=B, rho=RHO)
print(f"✓ artifacts saved to: {OUTDIR}")