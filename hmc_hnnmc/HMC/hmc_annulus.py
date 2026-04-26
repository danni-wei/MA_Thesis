# file: hmc_annulus_demo.py
import numpy as np
import matplotlib.pyplot as plt
from hmc import HMC  # 用你现在的 HMC 类（前面我们做过的那个）

# ---------- 目标：截断到圆环内的标准高斯 ----------
def E_gauss(X):            # X: [2,B]
    return 0.5*np.sum(X**2, axis=0, keepdims=True)

def dEdX_gauss(X):
    return X

CENTER = np.array([0.0, 0.0])   # 圆心；若要平移，用比如 np.array([0.5,0.5])
R_IN, R_OUT = 0.5, 1

def feasible_annulus(x_col):     # x_col: [2,1]
    r = np.linalg.norm(x_col[:,0] - CENTER)
    return (r >= R_IN) and (r <= R_OUT)

# ---------- 初始化 HMC ----------
X0 = (CENTER + np.array([R_IN+0.2, 0.0]))[:,None]  # 放在环内起步
sampler = HMC(
    Xinit=X0,
    E=E_gauss,
    dEdX=dEdX_gauss,
    epsilon=0.05,                 # 步长：可微调到 acc≈0.6-0.9
    num_leapfrog_steps=25,        # 轨迹长度：10~30 常用
    feasible=feasible_annulus,    # 关键：圆环硬约束
    display=0
)

# ---------- 背景：画高斯等高 + 圆环边界 ----------
xx, yy = np.meshgrid(np.linspace(-7, 7, 301), np.linspace(-7, 7, 301))
grid = np.vstack([xx.ravel(), yy.ravel()])
EE = E_gauss(grid).reshape(-1)
ZZ = np.exp(-EE).reshape(xx.shape)

th = np.linspace(0, 2*np.pi, 600)
c_in  = np.stack([CENTER[0] + R_IN*np.cos(th),  CENTER[1] + R_IN*np.sin(th)], axis=1)
c_out = np.stack([CENTER[0] + R_OUT*np.cos(th), CENTER[1] + R_OUT*np.sin(th)], axis=1)

plt.figure(figsize=(7,6))
plt.contour(xx, yy, ZZ, levels=20, cmap="Blues", alpha=0.6)
plt.plot(c_in[:,0],  c_in[:,1],  'g--', lw=2, label='r = R_in')
plt.plot(c_out[:,0], c_out[:,1], 'r-',  lw=2, label='r = R_out')

# ---------- 运行若干步，并画每步轨迹 ----------
N = 100
accepted_cnt = 0

# 说明：如果你的 HMC.step 已支持 `record_path=True` 并返回 (acc_rate, path, accepted)，
# 直接用下面这段；若没有该接口，看下方“微小补丁”。
for t in range(N):
    acc_rate, path, accepted = sampler.step(record_path=True)  # path: [L+1, 2, 1]
    curve = path[:, :, 0]    # [L+1, 2]
    color = "tab:green" if accepted else "0.6"
    lw = 1.4 if accepted else 0.9
    alpha = 0.9 if accepted else 0.6
    plt.plot(curve[:,0], curve[:,1], '-', lw=lw, alpha=alpha, color=color)
    plt.scatter(curve[-1,0], curve[-1,1], c="k", s=6, alpha=0.9, zorder=3)
    accepted_cnt += int(accepted)

# 当前点标红
curr = sampler.X[:, 0]
plt.scatter([curr[0]], [curr[1]], c="red", s=35, label=f"current (acc {accepted_cnt}/{N})")

plt.axis('equal'); plt.xlim(-4,4); plt.ylim(-4,4)
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("HMC on truncated Gaussian (annulus constraint) — leapfrog trajectories")
plt.legend(loc='upper right'); plt.grid(True, alpha=.3); plt.tight_layout(); plt.show()