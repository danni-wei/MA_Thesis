# SuS_HNN_series.py  (with progress logging + per-level visualization)
import numpy as np
import torch, os
import matplotlib.pyplot as plt
from scipy.stats import norm as Nrm, uniform

from HNNMC.hnnmc.nn_models import MLP
from HNNMC.hnnmc.hnn import HNN
from HNNMC.hnnmc.get_args import get_args
from HNNMC.hnnmc.utils import leapfrog
from HNNMC.hnnmc.functions import functions  # H(y)

# ================== 配置开关 ==================
LOG_EVERY   = 100          # 每层内每满多少个样本打印一次
VIS_LEVEL   = True         # 每层结束后保存散点+极限状态图
VIS_TRAJ    = False        # 是否叠加少量 HNN 轨迹（默认关）
N_TRAJ_DRAW = 10           # 每层最多画几条轨迹（VIS_TRAJ=True 时生效）
OUTDIR      = "sus_hnn_figs"
os.makedirs(OUTDIR, exist_ok=True)

# ================== 串联系统 G(x) ==================
def G_series(q):
    x1, x2 = q[0], q[1]
    u = x1 - x2
    v = (x1 + x2)/np.sqrt(2.0)
    g1 = 0.1*(u**2) - v + 3.0
    g2 = 0.1*(u**2) + v + 3.0
    g3 = u + 7.0/np.sqrt(2.0)
    g4 = -u + 7.0/np.sqrt(2.0)
    return np.min([g1, g2, g3, g4])

# ================== 载入 HNN ==================
args = get_args()
args.input_dim = 4
d = args.input_dim // 2

def get_model(args, baseline=False):
    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=baseline)
    path = args.dist_name + ".tar"
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model

def integrate_model(model, t_span, y0, n, **kwargs):
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, args.input_dim)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    return leapfrog(fun, t_span, y0, n, args.input_dim)

hnn_model = get_model(args, baseline=False)

def hnn_hmc_step(y, t_span, steps):
    y_traj = integrate_model(hnn_model, t_span, y, steps-1)   # [4, steps]
    y_prop = y_traj[:, -1]
    H_prev = functions(y)
    H_star = functions(y_prop)
    alpha = np.minimum(1.0, np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        return y_prop, True, y_traj
    else:
        y_rej = y.copy()
        y_rej[d:] = np.random.randn(d)
        return y_rej, False, y_traj

# ================== SuS 超参 ==================
N0   = 1000
p0   = 0.10
Jmax = 10
L = 10
epsilon = 0.025
steps  = int(L/epsilon)
t_span = [0, L]
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}

assert (N0*p0).is_integer() and (1/p0).is_integer(), "N0*p0 与 1/p0 必须是整数"

# ================== Level 0 ==================
samples0 = np.zeros((N0, d))
acc_hist0 = []

y = np.zeros(args.input_dim)
y[:d] = 0.0
y[d:] = np.random.randn(d)

for i in range(N0):
    y, acc, _ = hnn_hmc_step(y, t_span, steps)
    samples0[i, :] = y[:d]
    acc_hist0.append(1.0 if acc else 0.0)
    if (i+1) % LOG_EVERY == 0:
        print(f"[level 0] progress: {i+1}/{N0}")

G0  = np.array([G_series(samples0[i]) for i in range(N0)])
b1  = np.quantile(G0, p0)
idx = np.argsort(G0)[:int(p0*N0)]
seeds = samples0[idx]
print(f"[SuS-HNN] level 0 done. b1={b1:.3f}, acc={np.mean(acc_hist0):.2f}")

def _draw_level_figure(level_idx, X, title_note=""):
    # 背景网格
    xx, yy = np.meshgrid(np.linspace(-6, 6, 240), np.linspace(-6, 6, 240))
    u = xx - yy
    v = (xx + yy)/np.sqrt(2.0)
    g1 = 0.1*(u**2) - v + 3.0
    g2 = 0.1*(u**2) + v + 3.0
    g3 = u + 7.0/np.sqrt(2.0)
    g4 = -u + 7.0/np.sqrt(2.0)

    plt.figure(figsize=(6.5, 6))
    plt.contour(xx, yy, g1, [0], colors='r', linestyles='-')
    plt.contour(xx, yy, g2, [0], colors='r', linestyles='--')
    plt.contour(xx, yy, g3, [0], colors='g', linestyles='-')
    plt.contour(xx, yy, g4, [0], colors='g', linestyles='--')
    plt.scatter(X[:,0], X[:,1], s=6, c='k', alpha=0.6, label='samples')
    plt.legend(loc='upper left', fontsize=9)
    plt.xlabel('x1'); plt.ylabel('x2')
    plt.title(f"SuS-HNN level {level_idx} samples {title_note}")
    plt.grid(True, alpha=.3); plt.tight_layout()
    fname = os.path.join(OUTDIR, f"level_{level_idx}_samples.png")
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"✓ saved figure: {fname}")

if VIS_LEVEL:
    _draw_level_figure(0, samples0, title_note=f"| b1={b1:.3f}")

# ================== 后续层 ==================
levels = [dict(b=b1, acc=None)]
b_j = b1
cur = 1

while (b_j > 0) and (cur < Jmax):
    Sj = np.zeros((N0, d))
    acc_hist = []

    chains = seeds.shape[0]
    Lchain = int(np.ceil(N0 / chains))
    k = 0
    # 如启用轨迹可视化，收集少量轨迹（每层最多 N_TRAJ_DRAW 条）
    traj_bank = []

    for s in range(chains):
        y = np.zeros(args.input_dim)
        y[:d] = seeds[s]
        y[d:] = np.random.randn(d)

        for _ in range(Lchain):
            if k >= N0: break

            y_traj = integrate_model(hnn_model, t_span, y, steps-1)   # [4, steps]
            y_prop = y_traj[:, -1]
            q_prop = y_prop[:d]

            if G_series(q_prop) > b_j:
                # 强拒绝
                y[d:] = np.random.randn(d)
                acc = False
            else:
                # 在子集内做 MH 校正
                H_prev = functions(y)
                H_star = functions(y_prop)
                alpha = np.minimum(1.0, np.exp(H_prev - H_star))
                if alpha > uniform().rvs():
                    y = y_prop
                    acc = True
                    # 选择性记录轨迹（仅少量，避免爆文件）
                    if VIS_TRAJ and len(traj_bank) < N_TRAJ_DRAW:
                        traj_bank.append(y_traj.copy())
                else:
                    y[d:] = np.random.randn(d)
                    acc = False

            Sj[k, :] = y[:d]
            acc_hist.append(1.0 if acc else 0.0)
            k += 1

            if (k % LOG_EVERY) == 0:
                print(f"[level {cur}] progress: {k}/{N0}")

        if k >= N0: break

    Gj = np.array([G_series(Sj[i]) for i in range(N0)])
    b_next = np.quantile(Gj, p0)
    levels[-1]['acc'] = float(np.mean(acc_hist))
    levels.append(dict(b=b_next, acc=None))
    print(f"[SuS-HNN] level {cur} done. b{cur+1}={b_next:.3f}, acc={np.mean(acc_hist):.2f}")

    # 可视化该层
    if VIS_LEVEL:
        note = f"| b{cur+1}={b_next:.3f}, acc={np.mean(acc_hist):.2f}"
        _draw_level_figure(cur, Sj, title_note=note)

        if VIS_TRAJ and len(traj_bank) > 0:
            # 在该层图上叠加少量轨迹（新画一张叠加版）
            xx, yy = np.meshgrid(np.linspace(-6, 6, 240), np.linspace(-6, 6, 240))
            u_bg = xx - yy
            v_bg = (xx + yy)/np.sqrt(2.0)
            g1 = 0.1*(u_bg**2) - v_bg + 3.0
            g2 = 0.1*(u_bg**2) + v_bg + 3.0
            g3 = u_bg + 7.0/np.sqrt(2.0)
            g4 = -u_bg + 7.0/np.sqrt(2.0)

            plt.figure(figsize=(6.5, 6))
            plt.contour(xx, yy, g1, [0], colors='r', linestyles='-')
            plt.contour(xx, yy, g2, [0], colors='r', linestyles='--')
            plt.contour(xx, yy, g3, [0], colors='g', linestyles='-')
            plt.contour(xx, yy, g4, [0], colors='g', linestyles='--')
            plt.scatter(Sj[:,0], Sj[:,1], s=6, c='k', alpha=0.4, label='samples')
            # 画轨迹（只画位置分量）
            for tr in traj_bank:
                qpath = tr[:d, :]  # [2, steps]
                plt.plot(qpath[0], qpath[1], '-', lw=1, alpha=0.8)
                plt.plot([qpath[0,0]],[qpath[1,0]],'go',ms=4,alpha=0.8)   # start
                plt.plot([qpath[0,-1]],[qpath[1,-1]],'ro',ms=4,alpha=0.8) # end
            plt.xlabel('x1'); plt.ylabel('x2')
            plt.title(f"SuS-HNN level {cur} with trajectories {note}")
            plt.grid(True, alpha=.3); plt.tight_layout()
            fname = os.path.join(OUTDIR, f"level_{cur}_traj.png")
            plt.savefig(fname, dpi=200); plt.close()
            print(f"✓ saved figure: {fname}")

    # 下层准备
    idx = np.argsort(Gj)[:int(p0*N0)]
    seeds = Sj[idx]
    b_j = b_next
    cur += 1

# ================== 终层指标 ==================
J = len(levels) - 1
final_set = seeds if b_j <= 0 else Sj
G_final = np.array([G_series(q) for q in final_set])
nJ = int(np.sum(G_final <= 0.0))
Pf_hat = (p0**(J-1)) * (nJ / N0) if J >= 1 else (nJ / N0)
beta_hat = -Nrm.ppf(Pf_hat) if Pf_hat > 0 else np.inf

print(f"[SuS-HNN] J={J}, nJ={nJ}/{N0},  Pf^={Pf_hat:.3e},  beta^={(beta_hat if np.isfinite(beta_hat) else float('inf')):.2f}")
for j in range(0, J):
    print(f"  level {j+1}: b={levels[j]['b']:.3f}, acc={levels[j]['acc']:.2f}")

np.savez(os.path.join(OUTDIR, "sus_hnn_series_outputs.npz"),
         Pf_hat=Pf_hat, beta_hat=beta_hat,
         levels_b=np.array([lv['b'] for lv in levels[:-1]]),
         acc=np.array([lv['acc'] for lv in levels[:-1]]))
print(f"✓ artifacts saved to: {OUTDIR}")