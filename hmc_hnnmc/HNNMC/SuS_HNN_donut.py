# SuS_HNN_annulus.py  — Subset Simulation with HNN-HMC (2D annulus failure: 3 <= r <= 5)
import numpy as np
import torch, os
import matplotlib.pyplot as plt
from scipy.stats import norm as Nrm, uniform

from HNNMC.hnnmc.nn_models import MLP
from HNNMC.hnnmc.hnn import HNN
from HNNMC.hnnmc.get_args import get_args
from HNNMC.hnnmc.utils import leapfrog
from HNNMC.hnnmc.functions import functions  # H(y) = U(q) + 0.5||p||^2

# ======= Ops & viz toggles =======
LOG_EVERY   = 100
VIS_LEVEL   = True
VIS_TRAJ    = False
N_TRAJ_DRAW = 10
OUTDIR      = "sus_hnn_annulus_figs"
os.makedirs(OUTDIR, exist_ok=True)

# ======= Limit state: G(x)=max{f,g} with f=r^2-25, g=9-r^2 =======
def G_annulus(q):  # q: shape (2,)
    x1, x2 = q[0], q[1]
    r2 = (x1)*(x1) + (x2)*(x2)
    f = r2 - 36         # <=0  => r<=6
    g = 25 - r2           # <=0  => r>=5
    return max(f, g)       # failure if max(f,g) <= 0  <=> both <= 0  <=> 3<=r<=5

# ======= Load HNN model (same as your hnn_hmc.py) =======
args = get_args()
args.input_dim = 4
d = args.input_dim // 2

def get_model(args, baseline=False):
    output_dim = args.input_dim
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, args.nonlinearity)
    model = HNN(args.input_dim, differentiable_model=nn_model,
                field_type=args.field_type, baseline=baseline)
    path = args.dist_name + ".tar"   # e.g., 'nD_standard_Gaussian.tar'
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
    y_traj = integrate_model(hnn_model, t_span, y, steps-1)  # [4, steps]
    y_prop = y_traj[:, -1]
    H_prev = functions(y)
    H_star = functions(y_prop)
    alpha = np.minimum(1.0, np.exp(H_prev - H_star))
    if alpha > uniform().rvs():
        return y_prop, True, y_traj
    else:
        y_rej = y.copy()
        y_rej[d:] = np.random.randn(d)  # refresh momentum
        return y_rej, False, y_traj

# ======= SuS hyper-parameters =======
N0   = 1000         # samples per level
p0   = 0.10         # keep N0*p0, 1/p0 integers
Jmax = 10
L = 10
epsilon = 0.025
steps  = int(L/epsilon)
t_span = [0, L]
kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], steps), 'rtol': 1e-10}
assert (N0*p0).is_integer() and (1/p0).is_integer(), "N0*p0 and 1/p0 must be integers."

# ======= Level 0: Unconstrained sampling =======
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

G0  = np.array([G_annulus(samples0[i]) for i in range(N0)])
b1  = np.quantile(G0, p0)
idx = np.argsort(G0)[:int(p0*N0)]
seeds = samples0[idx]
print(f"[SuS-HNN] level 0 done. b1={b1:.3f}, acc={np.mean(acc_hist0):.2f}")

def draw_level(level_idx, X, title_note=""):
    cx, cy = 0, 0
    th = np.linspace(0, 2*np.pi, 400)
    c3 = np.stack([cx + 5*np.cos(th), cy + 5*np.sin(th)], axis=1)
    c5 = np.stack([cx + 6*np.cos(th), cy + 6*np.sin(th)], axis=1)
    plt.figure(figsize=(6.5,6))
    plt.plot(c3[:,0], c3[:,1], 'g--', lw=2, label='||x-c||=25 (g=0)')
    plt.plot(c5[:,0], c5[:,1], 'r-',  lw=2, label='||x-c||=36 (f=0)')
    plt.scatter(X[:,0], X[:,1], s=6, c='k', alpha=0.6, label='samples')
    plt.legend(loc='upper right', fontsize=9)
    plt.xlabel('x1'); plt.ylabel('x2'); plt.axis('equal')
    plt.title(f"SuS-HNN level {level_idx} samples {title_note}")
    plt.grid(True, alpha=.3); plt.tight_layout()
    fname = os.path.join(OUTDIR, f"level_{level_idx}_samples.png")
    plt.savefig(fname, dpi=200); plt.close()
    print(f"✓ saved figure: {fname}")

if VIS_LEVEL:
    draw_level(0, samples0, title_note=f"| b1={b1:.3f}")

# ======= Subsequent levels: constrained by {G<=b_j} =======
levels = [dict(b=b1, acc=None)]
b_j = b1
cur = 1

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
            y_traj = integrate_model(hnn_model, t_span, y, steps-1)
            y_prop = y_traj[:, -1]
            q_prop = y_prop[:d]

            if G_annulus(q_prop) > b_j:
                y[d:] = np.random.randn(d)
                acc = False
            else:
                H_prev = functions(y)
                H_star = functions(y_prop)
                alpha = np.minimum(1.0, np.exp(H_prev - H_star))
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

    Gj = np.array([G_annulus(Sj[i]) for i in range(N0)])
    b_next = np.quantile(Gj, p0)
    levels[-1]['acc'] = float(np.mean(acc_hist))
    levels.append(dict(b=b_next, acc=None))
    print(f"[SuS-HNN] level {cur} done. b{cur+1}={b_next:.3f}, acc={np.mean(acc_hist):.2f}")

    if VIS_LEVEL:
        note = f"| b{cur+1}={b_next:.3f}, acc={np.mean(acc_hist):.2f}"
        draw_level(cur, Sj, title_note=note)

    idx = np.argsort(Gj)[:int(p0*N0)]
    seeds = Sj[idx]
    b_j = b_next
    cur += 1

# ======= Final estimates =======
J = len(levels) - 1
final_set = seeds if b_j <= 0 else Sj
G_final = np.array([G_annulus(q) for q in final_set])
nJ = int(np.sum(G_final <= 0.0))
Pf_hat = (p0**(J-1)) * (nJ / N0) if J >= 1 else (nJ / N0)
beta_hat = -Nrm.ppf(Pf_hat) if Pf_hat > 0 else np.inf

print(f"[SuS-HNN] J={J}, nJ={nJ}/{N0},  Pf^={Pf_hat:.3e},  beta^={(beta_hat if np.isfinite(beta_hat) else float('inf')):.2f}")
for j in range(0, J):
    print(f"  level {j+1}: b={levels[j]['b']:.3f}, acc={levels[j]['acc']:.2f}")

np.savez(os.path.join(OUTDIR, "sus_hnn_annulus_outputs.npz"),
         Pf_hat=Pf_hat, beta_hat=beta_hat,
         levels_b=np.array([lv['b'] for lv in levels[:-1]]),
         acc=np.array([lv['acc'] for lv in levels[:-1]]))
print(f"✓ artifacts saved to: {OUTDIR}")