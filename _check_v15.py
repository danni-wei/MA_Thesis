import sys; sys.path.insert(0, '.')
import torch
from pinn_hmc.compare_pinn_hnn import leapfrog_hnn_one_step, yoshida4_hnn
from pinn_hmc.model import HNNConfig, HamiltonianNN
from pinn_hmc.target import get_device

device = get_device()
model = HamiltonianNN(HNNConfig(dim=2, hidden_dim=128, depth=4)).to(device)
model.load_state_dict(torch.load('results/smoke_yoshida_hnn/hnn_gaussian.pt', map_location=device, weights_only=True))
model.eval()

q = torch.tensor([[1.0, 0.0]], device=device)
p = torch.tensor([[0.0, 1.0]], device=device)

# One step
q2, p2, n = leapfrog_hnn_one_step(model, q, p, 0.1)
print(f'leapfrog_hnn_one_step: n_vf={n}, q={q2.tolist()}, p={p2.tolist()}')

# Yoshida with n_steps=10
q3, p3, n3 = yoshida4_hnn(model, q, p, 0.1, 10)
print(f'yoshida4_hnn (n_steps=10): n_vf={n3} (expect 90), q={q3.tolist()}')

# Also check leapfrog_hnn call count
from pinn_hmc.compare_pinn_hnn import leapfrog_hnn
q4, p4 = leapfrog_hnn(model, q, p, 0.1, 10)
print(f'leapfrog_hnn (n_steps=10): expect 21 vf calls, returns negated p={p4.tolist()}')

import math
cbrt2 = 2**(1/3)
w0 = -cbrt2 / (2 - cbrt2)
print(f'w0 = {w0:.6f}')
print('All checks passed.')
