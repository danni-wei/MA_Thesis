# Smoke test: Yoshida × HNN vs Leapfrog × HNN

## Setup

| Parameter | Value |
|-----------|-------|
| Targets | 2D standard Gaussian, 2D Banana (b=0.15) |
| step_size | 0.1 |
| n_steps | 10 |
| n_proposals | 2000 (after 200 burn-in) |
| HNN arch | 4-layer MLP, 128 hidden, Tanh |
| HNN training | 120 epochs, Adam 1e-3, 1800 trajectories |
| Yoshida w0 | ≈ -1.7024 |

## Results

| Config | Target | dH_learned | dH_true | accept | n_grad_evals |
|--------|--------|-----------|---------|--------|-------------|
| leapfrog+true | gaussian | 2.195e-03 | 2.195e-03 | 1.000 | 24200 |
| yoshida+true | gaussian | 6.716e-06 | 6.716e-06 | 1.000 | 132000 |
| leapfrog+hnn | gaussian | 6.431e-03 | 5.511e-03 | 0.998 | 24200 |
| yoshida+hnn | gaussian | 5.873e-03 | 5.248e-03 | 0.998 | 132000 |
| leapfrog+true | banana | 2.402e-03 | 2.402e-03 | 0.999 | 24200 |
| yoshida+true | banana | 8.120e-06 | 8.120e-06 | 1.000 | 132000 |
| leapfrog+hnn | banana | 1.119e-02 | 1.651e-02 | 0.992 | 24200 |
| yoshida+hnn | banana | 1.090e-02 | 1.593e-02 | 0.992 | 132000 |

## Observation

Gaussian: yoshida+hnn reduces dH_learned by factor ~1.1x vs leapfrog+hnn; dH_true differs by 4.8% between the two (HNN model error bottleneck confirmed).

## Plots

- `dH_trajectory_gaussian.png` — |ΔH_true| trace for 4 configs on Gaussian
- `dH_trajectory_banana.png`   — |ΔH_true| trace for 4 configs on Banana

---

## Stage 1.5 — Full HNN vector field (q-drift uses dH/dp)

### Setup

Same hyper-parameters as Stage 1 (step_size=0.1, n_steps=10,
n_proposals=2000, same HNN checkpoints).

**Key change**: leapfrog+hnn and yoshida+hnn now use the complete HNN vector
field for BOTH q-drift (dH/dp) and p-kick (-dH/dq).

hnn_vector_field calls per proposal:
- leapfrog_hnn  : 2*10+1 = 21 calls  (each = 1 forward + 2 backward)
- yoshida4_hnn  : 9*10   = 90 calls  (3 sub-steps x 3 calls, no merging)
- Cost ratio    : 90/21 = 4.29x

### Results

| Config | Target | dH_learned | dH_true | accept | n_vf_calls |
|--------|--------|-----------|---------|--------|-----------|
| leapfrog+true | gaussian | 2.195e-03 | 2.195e-03 | 1.000 | 24200 |
| yoshida+true | gaussian | 6.716e-06 | 6.716e-06 | 1.000 | 132000 |
| leapfrog+hnn | gaussian | 6.390e-03 | 6.843e-03 | 0.997 | 46200 |
| yoshida+hnn | gaussian | 6.069e-03 | 6.856e-03 | 0.997 | 198000 |
| leapfrog+true | banana | 2.402e-03 | 2.402e-03 | 0.999 | 24200 |
| yoshida+true | banana | 8.120e-06 | 8.120e-06 | 1.000 | 132000 |
| leapfrog+hnn | banana | 1.049e-02 | 1.514e-02 | 0.992 | 46200 |
| yoshida+hnn | banana | 1.076e-02 | 1.503e-02 | 0.991 | 198000 |

### Observation (Gaussian)

- yoshida+hnn dH_learned / leapfrog+hnn dH_learned = 1.1x (no significant improvement -- check symmetry of base step)
- yoshida+hnn dH_true    / leapfrog+hnn dH_true    = 1.00x  (HNN model error floor)
- Scissors gap marginal or absent: dH_learned drops by 1.1x but dH_true only by 1.00x

### Plots

- `dH_trajectory_gaussian_v15.png` / `dH_trajectory_banana_v15.png` — |dH_true| traces
- `scissors_gaussian.png` / `scissors_banana.png` — dH_learned vs dH_true side-by-side

---

## Stage 1.6 - Step-size convergence sweep

T_fixed=1.0, N_IC=200, h values: [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

h=0.200(n=5)  h=0.100(n=10)  h=0.050(n=20)  h=0.020(n=50)  h=0.010(n=100)  h=0.005(n=200)

| Target | Curve | log-log slope (full h range) |
|--------|-------|------------------------------|
| gaussian | true+leapfrog | 0.977 |
| gaussian | true+yoshida | 0.814 |
| gaussian | hnn+leapfrog (dH_lrn) | 1.738 |
| gaussian | hnn+yoshida (dH_lrn) | 0.988 |
| gaussian | hnn dH_true (floor) | 0.059 |
|   |   |   |
| banana | true+leapfrog | 0.978 |
| banana | true+yoshida | 0.955 |
| banana | hnn+leapfrog (dH_lrn) | 1.658 |
| banana | hnn+yoshida (dH_lrn) | 0.995 |
| banana | hnn dH_true (floor) | 0.065 |
|   |   |   |

### Plots

- `convergence_gaussian.png`
- `convergence_banana.png`

### Reading guide

- true+leapfrog slope ~2 and true+yoshida slope ~4: integrators correct, control passed.
- hnn+leapfrog and hnn+yoshida slopes: if both plateau at small h, dH_true floor
  dominates and Yoshida order advantage does not transfer.
- hnn dH_true floor slope ~0: model error is h-independent, confirming it is the
  fundamental bottleneck for energy conservation in HNN-driven integration.

---

## Stage 1.6 - Step-size convergence sweep

T_fixed=1.0, N_IC=200, h values: [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

h=0.200(n=5)  h=0.100(n=10)  h=0.050(n=20)  h=0.020(n=50)  h=0.010(n=100)  h=0.005(n=200)

| Target | Curve | log-log slope (full h range) |
|--------|-------|------------------------------|
| gaussian | true+leapfrog | 1.990 |
| gaussian | true+yoshida | 0.814 |
| gaussian | hnn+leapfrog (dH_lrn) | 1.738 |
| gaussian | hnn+yoshida (dH_lrn) | 0.988 |
| gaussian | hnn dH_true (floor) | 0.059 |
|   |   |   |
| banana | true+leapfrog | 1.988 |
| banana | true+yoshida | 0.955 |
| banana | hnn+leapfrog (dH_lrn) | 1.658 |
| banana | hnn+yoshida (dH_lrn) | 0.995 |
| banana | hnn dH_true (floor) | 0.065 |
|   |   |   |

### Plots

- `convergence_gaussian.png`
- `convergence_banana.png`

### Reading guide

- true+leapfrog slope ~2 and true+yoshida slope ~4: integrators correct, control passed.
- hnn+leapfrog and hnn+yoshida slopes: if both plateau at small h, dH_true floor
  dominates and Yoshida order advantage does not transfer.
- hnn dH_true floor slope ~0: model error is h-independent, confirming it is the
  fundamental bottleneck for energy conservation in HNN-driven integration.

---

## Stage 2 — SuS integrator comparison (leapfrog+hnn vs yoshida+hnn, N_rep=30)

N=500, p0=0.1, β=3.0, Pf_ref=1.3499e-03
step_size=0.1, n_steps=10, VF_lf=21, VF_y4=90

### Main table

| Config | Target | mean_Pf | COV | est_vf/run | mean_lv |
|--------|--------|---------|-----|------------|---------|
| leapfrog+hnn | gaussian | 1.2940e-03 | 0.343 | 2.132e+06 | 3.33 |
| yoshida+hnn | gaussian | 1.3439e-03 | 0.362 | 9.134e+06 | 3.30 |
| leapfrog+hnn | banana | 1.3927e-03 | 0.401 | 2.130e+06 | 3.20 |
| yoshida+hnn | banana | 1.3200e-03 | 0.416 | 9.138e+06 | 3.40 |

### Mechanism

**leapfrog+hnn / gaussian**

| lv | geom_rej | energy_rej | accept |
|----|----------|------------|--------|
| L1 | 0.637 | 0.001 | 0.362 |
| L2 | 0.848 | 0.001 | 0.151 |
| L3 | 0.923 | 0.001 | 0.076 |

**yoshida+hnn / gaussian**

| lv | geom_rej | energy_rej | accept |
|----|----------|------------|--------|
| L1 | 0.637 | 0.001 | 0.363 |
| L2 | 0.849 | 0.001 | 0.151 |
| L3 | 0.923 | 0.000 | 0.077 |

**leapfrog+hnn / banana**

| lv | geom_rej | energy_rej | accept |
|----|----------|------------|--------|
| L1 | 0.626 | 0.008 | 0.367 |
| L2 | 0.829 | 0.007 | 0.164 |
| L3 | 0.900 | 0.007 | 0.094 |

**yoshida+hnn / banana**

| lv | geom_rej | energy_rej | accept |
|----|----------|------------|--------|
| L1 | 0.625 | 0.007 | 0.367 |
| L2 | 0.828 | 0.007 | 0.165 |
| L3 | 0.910 | 0.007 | 0.083 |

---

## 最终结论（N_rep=30）

### 主结果表

| target | integrator | mean_Pf | Pf_ref | COV | est_vf/run | cost_ratio |
|--------|-----------|---------|--------|-----|------------|------------|
| gaussian | leapfrog+hnn | 1.2940e-03 | 1.3499e-03 | 0.343 | 2.132e+06 | 1.00 |
| gaussian | yoshida+hnn | 1.3439e-03 | 1.3499e-03 | 0.362 | 9.134e+06 | 4.29 |
| banana | leapfrog+hnn | 1.3927e-03 | 1.3499e-03 | 0.401 | 2.130e+06 | 1.00 |
| banana | yoshida+hnn | 1.3200e-03 | 1.3499e-03 | 0.416 | 9.138e+06 | 4.29 |

### 机制表

**gaussian / leapfrog+hnn**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6367 | 0.0010 | 0.00265 | 0.3624 |
| L2 | 0.8479 | 0.0007 | 0.00487 | 0.1514 |
| L3 | 0.9229 | 0.0007 | 0.00865 | 0.0764 |

**gaussian / yoshida+hnn**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6367 | 0.0007 | 0.00204 | 0.3625 |
| L2 | 0.8487 | 0.0008 | 0.00538 | 0.1505 |
| L3 | 0.9230 | 0.0002 | 0.00321 | 0.0768 |

**banana / leapfrog+hnn**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6257 | 0.0077 | 0.02058 | 0.3666 |
| L2 | 0.8293 | 0.0070 | 0.04080 | 0.1637 |
| L3 | 0.8996 | 0.0067 | 0.06642 | 0.0937 |

**banana / yoshida+hnn**

| level | geom_rej(a) | energy_rej(a) | cond_energy_rej(b) | accept |
|-------|------------|--------------|-------------------|--------|
| L1 | 0.6254 | 0.0075 | 0.01997 | 0.3671 |
| L2 | 0.8278 | 0.0073 | 0.04258 | 0.1649 |
| L3 | 0.9102 | 0.0067 | 0.07423 | 0.0831 |

两积分器 Pf 差异约 0.4–0.5 个标准误，统计上不显著（N_rep=30）；
COV 差异同样在 N=30 的抽样误差范围内（≈±0.05），两指标统计不可区分。
几何拒绝率逐层相同至小数点后三位（两积分器），能量拒绝率 <1%（无条件）
/ (b) 给出精确条件率，证明 MH 步几乎不拒绝几何合法提议。

