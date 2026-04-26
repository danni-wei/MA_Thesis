# hnnmc_annulus_demo.py — HNN-HMC sampling on truncated Gaussian (annulus)
import os, numpy as np, torch, matplotlib.pyplot as plt
from scipy.stats import uniform
from hnnmc.get_args import get_args
from hnnmc.nn_models import MLP
from hnnmc.hnn import HNN
from hnnmc.utils import leapfrog
from hnnmc.functions import functions  # H(y) = U(q)+0.5||p||^2

# ---------- annulus constraint ----------
CENTER = np.array([0.0, 0.0])
R_IN, R_OUT = 3.0, 5.0

def feasible_annulus(q_col):
    q = q_col[:, 0]
    r = np.linalg.norm(q - CENTER)
    return (r >= R_IN) and (r <= R_OUT)

# ---------- load HNN (trained on standard Gaussian) ----------
args = get_args()
args.input_dim = 4             # 2D q + 2D p
args.dist_name = 'nD_standard_Gaussian'
# 这些要与训练权重一致（若不一致会 size mismatch）：
# args.hidden_dim = 100
# args.nonlinearity = 'sine'
# args.field_type   = 'H'

def get_model():
    net = MLP(args.input_dim, args.hidden_dim, args.input_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=net,
                field_type=args.field_type, baseline=False)
    ckpt = os.path.join(args.load_dir, args.dist_name + ".tar")
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model

def integrate_model(model, t_span, y0, n):
    def f(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32).view(1, args.input_dim)
        dx = model.time_derivative(x).detach().numpy().reshape(-1)
        return dx
    return leapfrog(f, t_span, y0, n, args.input_dim)

hnn_model = get_model()

# ---------- single HNN-HMC step with feasibility ----------
def hnn_hmc_step(y, t_span, steps, record_path=False, clip_path_to_feasible=True):
    traj = integrate_model(hnn_model, t_span, y, steps-1)  # [4, steps]
    q_traj = traj[:2, :]  # 只看位置
    # 可选：只保留环内的轨迹段用于可视化
    if record_path and clip_path_to_feasible:
        keep = []
        for k in range(q_traj.shape[1]):
            if feasible_annulus(q_traj[:, [k]]):
                keep.append(k)
            else:
                break
        if len(keep) > 0:
            show_traj = traj[:, keep]   # [4, L_vis]
        else:
            show_traj = traj[:, :1]     # 至少留一个点
    else:
        show_traj = traj

    y_prop = traj[:, -1]
    q_prop = y_prop[:2]

    # 不可行 => 直接判拒，刷新动量
    if not feasible_annulus(q_prop[:, None]):
        y_new = y.copy()
        y_new[2:] = np.random.randn(2)
        return y_new, False, show_traj

    # MH 校正（基于 HNN 的能量）
    H_prev = functions(y)
    H_star = functions(y_prop)
    alpha  = np.minimum(1.0, np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        return y_prop, True, show_traj
    else:
        y_new = y.copy()
        y_new[2:] = np.random.randn(2)
        return y_new, False, show_traj

# ---------- viz background (Gaussian contours + ring) ----------
xx, yy = np.meshgrid(np.linspace(-7, 7, 301), np.linspace(-7, 7, 301))
grid = np.vstack([xx.ravel(), yy.ravel()])
EE = 0.5*np.sum(grid**2, axis=0)  # U(q)=0.5||q||^2
ZZ = np.exp(-EE).reshape(xx.shape)

th = np.linspace(0, 2*np.pi, 600)
c_in  = np.stack([CENTER[0] + R_IN*np.cos(th),  CENTER[1] + R_IN*np.sin(th)], axis=1)
c_out = np.stack([CENTER[0] + R_OUT*np.cos(th), CENTER[1] + R_OUT*np.sin(th)], axis=1)

plt.figure(figsize=(7,6))
plt.contour(xx, yy, ZZ, levels=20, cmap="Blues", alpha=0.5)
plt.plot(c_in[:,0],  c_in[:,1],  'g--', lw=2, label='r = R_in')
plt.plot(c_out[:,0], c_out[:,1], 'r-',  lw=2, label='r = R_out')

# ---------- run N steps & plot trajectories ----------
L = 12              # trajectory length (in "time")
eps = 0.025         # step size
steps = int(L/eps)  # leapfrog steps
t_span = [0, L]
N = 150

# 起点放在环里
y = np.zeros(4)
y[:2] = CENTER + np.array([R_IN + 0.3, 0.0])  # q
y[2:] = np.random.randn(2)                    # p

acc_cnt = 0
for t in range(N):
    y, accepted, traj = hnn_hmc_step(y, t_span, steps, record_path=True, clip_path_to_feasible=True)
    curve = traj[:2, :].T
    color = "tab:green" if accepted else "0.6"
    lw = 1.3 if accepted else 0.9
    alpha = 0.95 if accepted else 0.6
    plt.plot(curve[:,0], curve[:,1], '-', lw=lw, alpha=alpha, color=color)
    plt.scatter(curve[-1,0], curve[-1,1], c='k', s=6, zorder=3)
    acc_cnt += int(accepted)

curr = y[:2]
plt.scatter([curr[0]], [curr[1]], c='red', s=35, label=f'current (acc {acc_cnt}/{N})')
plt.axis('equal'); plt.xlim(-7,7); plt.ylim(-7,7)
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("HNN-HMC on truncated Gaussian (annulus) — leapfrog trajectories")
plt.legend(loc='upper right'); plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()