# run_sus_series.py
import numpy as np
import matplotlib.pyplot as plt
from ERADist import ERADist
from SuS import SuS
from g_series import g_fun

# --- 独立 N(0,1) marginals ---
d = 2
marg = ERADist('normal', 'PAR', [0.0, 1.0])
distr = [marg]*d

# --- SuS 参数（必须满足 N*p0 和 1/p0 都是整数）---
N  = 1000      # 每层样本数
p0 = 0.10      # 每层保留比例

# --- 运行 ---
Pf_SuS, delta_SuS, b, Pf, b_line, Pf_line, samplesU, samplesX, f_s_iid = \
    SuS(N=N, p0=p0, g_fun=g_fun, distr=distr, samples_return=2)

print("\n=== Subset Simulation (Series system) ===")
print(f"Pf_hat     = {Pf_SuS:.3e}")
print(f"CoV (δ)    = {delta_SuS:.3f}")
print(f"levels b   = {np.round(b,3)}")
print(f"per-level p= {np.round(Pf,5)}")


# Pf 演进曲线（横轴 b_line, 纵轴 Pf_line）
plt.figure(figsize=(6,4))
plt.semilogy(b_line, Pf_line, '.-')
plt.axvline(0, color='r', ls='--', lw=1, label='g=0')
plt.xlabel('threshold b'); plt.ylabel('P(G<=b)')
plt.title('Pf evolution over SuS levels'); plt.grid(True, which='both', alpha=.3)
plt.legend(); plt.tight_layout(); plt.show()

# 最后一层样本散点（物理空间）
if len(samplesX)>0:
    X_last = samplesX[-1]
    x1, x2 = X_last[:,0], X_last[:,1]
    plt.figure(figsize=(6,6))
    plt.scatter(x1, x2, s=4, c='k', alpha=.5, label='last-level samples')
    plt.xlabel('x1'); plt.ylabel('x2'); plt.title('SuS last-level samples')
    plt.grid(True, alpha=.3); plt.legend(); plt.tight_layout(); plt.show()