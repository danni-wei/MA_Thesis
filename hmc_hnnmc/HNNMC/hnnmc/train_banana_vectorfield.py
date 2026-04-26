import os, numpy as np, torch, torch.nn as nn
from nn_models import MLP
from hnn import HNN

# -------- Banana potential U(q) and ∇U(q)（torch version）---------
def banana_grad_U_torch(q, a=1.15, b=0.5, rho=0.9):
    # q: [B,2]
    x, y = q[:, 0], q[:, 1]
    Sigma_inv = torch.tensor([[1.0, -rho], [-rho, 1.0]], dtype=q.dtype, device=q.device) / (1.0 - rho**2)
    u = torch.stack([a * x, a * (y - b * x * x)], dim=1)  # [B,2]
    g_u = u @ Sigma_inv.T  # dU/du = Sigma_inv u
    du_dx = torch.stack([torch.full_like(x, a), -2 * a * b * x], dim=1)  # [B,2]
    du_dy = torch.stack([torch.zeros_like(y), torch.full_like(y, a)], dim=1)
    dU_dx = torch.sum(g_u * du_dx, dim=1)
    dU_dy = torch.sum(g_u * du_dy, dim=1)
    return torch.stack([dU_dx, dU_dy], dim=1)  # [B,2]

# ------------------- training setup -------------------
input_dim = 4          # q(2)+p(2)
hidden_dim = 128
nonlinearity = 'sine'
learn_rate = 1e-3
total_steps = 8000     # first 4k~8k
batch_size = 2048
save_path = "banana.tar"

device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ------------------- model and optimizer -------------------
mlp = MLP(input_dim, hidden_dim, output_dim=input_dim, nonlinearity=nonlinearity)
model = HNN(input_dim, differentiable_model=mlp, field_type='solenoidal', baseline=False).to(device)
if hasattr(model, 'M'):
    model.M = model.M.to(device)
opt = torch.optim.Adam(model.parameters(), lr=learn_rate)
mse = nn.MSELoss()

# ------------------- training loop -------------------
def sample_batch(B):
    # q within the banana region，p ~ N(0,I)
    q = np.stack([np.random.uniform(-5, 5, size=B),
                  np.random.uniform(-5, 8, size=B)], axis=1).astype(np.float32)  # [B,2]
    p = np.random.randn(B, 2).astype(np.float32)
    z = np.concatenate([q, p], axis=1)  # [B,4]
    return torch.from_numpy(z)

model.train()
for it in range(1, total_steps + 1):
    z = z = sample_batch(batch_size).to(device).requires_grad_(True)      # [B,4]
    q, p = z[:, :2], z[:, 2:]
    v_true = torch.cat([p, -banana_grad_U_torch(q)], dim=1)  # [B,4] = [dq/dt, dp/dt]

    v_pred = model.time_derivative(z)  # [B,4]
    loss = mse(v_pred, v_true)

    opt.zero_grad(); loss.backward(); opt.step()

    if it % 500 == 0:
        print(f"[train] step={it}/{total_steps}  loss={loss.item():.4e}")

# ------------------- save the weight -------------------
torch.save(model.state_dict(), save_path)
print("✓ saved:", os.path.abspath(save_path))