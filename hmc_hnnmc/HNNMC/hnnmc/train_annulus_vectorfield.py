# train_annulus_vectorfield.py
import os, torch, numpy as np
from hnnmc.get_args import get_args
from hnnmc.nn_models import MLP
from hnnmc.hnn import HNN

args = get_args()
args.dist_name  = 'annulus'
args.input_dim  = 4              # q(2)+p(2)
# 保持与最终采样脚本一致的网络超参
# args.hidden_dim = 100; args.nonlinearity='sine'; args.field_type='H'

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("device:", device)

net = MLP(args.input_dim, args.hidden_dim, args.input_dim, args.nonlinearity)
model = HNN(args.input_dim, differentiable_model=net,
            field_type=args.field_type, baseline=False).to(device)
# --- 圆环参数（与 functions.py 对齐） ---
R_IN, R_OUT = 3.0, 5.0
LAMBDA, MARGIN = 200.0, 0.0

def true_field(y_np):
    """y=[q1,q2,p1,p2] -> dy/dt = [dq/dt, dp/dt]"""
    q1, q2, p1, p2 = y_np[:,0], y_np[:,1], y_np[:,2], y_np[:,3]
    r = np.sqrt(q1**2 + q2**2)

    # 势能 U = 0.5*||q||^2 + penalty
    # ∂U/∂q = q + ∂pen/∂q
    gradU_q1 = q1.copy()
    gradU_q2 = q2.copy()
    # 软障壁的梯度：2*lambda*(r - R_OUT)*∂r/∂q  或  -2*lambda*(R_IN - r)*∂r/∂q
    eps = 1e-12
    dr_dq1 = np.where(r>eps, q1/(r+eps), 0.0)
    dr_dq2 = np.where(r>eps, q2/(r+eps), 0.0)

    outer = (r > (R_OUT + MARGIN))
    inner = (r < (R_IN  - MARGIN))

    gradU_q1[outer] += 2*LAMBDA*(r[outer] - (R_OUT + MARGIN)) * dr_dq1[outer]
    gradU_q2[outer] += 2*LAMBDA*(r[outer] - (R_OUT + MARGIN)) * dr_dq2[outer]
    gradU_q1[inner] += -2*LAMBDA*((R_IN - MARGIN) - r[inner]) * dr_dq1[inner]
    gradU_q2[inner] += -2*LAMBDA*((R_IN - MARGIN) - r[inner]) * dr_dq2[inner]

    dqdt1, dqdt2 = p1, p2
    dpdt1, dpdt2 = -gradU_q1, -gradU_q2
    return np.stack([dqdt1, dqdt2, dpdt1, dpdt2], axis=1)

# --- 模型 ---
net = MLP(args.input_dim, args.hidden_dim, args.input_dim, args.nonlinearity)
model = HNN(args.input_dim, differentiable_model=net, field_type=args.field_type, baseline=False).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

# --- 训练数据：q 在环内随机采样，p~N(0,I) ---
def sample_batch(B=2048):
    # 先在环内均匀采 r,theta，再转成 q
    u = np.random.rand(B)
    r = np.sqrt((R_OUT**2 - R_IN**2)*u + R_IN**2)
    theta = 2*np.pi*np.random.rand(B)
    q1 = r*np.cos(theta); q2 = r*np.sin(theta)
    p  = np.random.randn(B, 2)
    y  = np.stack([q1, q2, p[:,0], p[:,1]], axis=1)
    f  = true_field(y)  # 真向量场
    return y, f

total_steps = 8000
for step in range(1, total_steps+1):
    y_np, f_np = sample_batch(B=2048)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device, requires_grad=True)
    f_t = torch.tensor(f_np, dtype=torch.float32, device=device)

    pred = model.time_derivative(y_t)
    loss = torch.mean((pred - f_t)**2)
    opt.zero_grad(); loss.backward(); opt.step()

    if step % 500 == 0:
        print(f"[train] step={step}/{total_steps}  loss={loss.item():.4e}")

# --- 保存权重 ---
ckpt = os.path.join(args.save_dir, args.dist_name + ".tar")
torch.save(model.state_dict(), ckpt)
print("✓ saved:", ckpt)