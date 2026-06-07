# Yoshida × HNN-HMC in Subset Simulation — 结论

## 研究问题
在 Subset Simulation (SuS) 的 HMC within-level sampler 中，用 HNN 学到的哈密顿
H_θ(q,p) 提供梯度场，并把标准二阶 leapfrog 替换为四阶 Yoshida 辛积分器。
积分器忠实积分**完整 HNN 向量场**（q-update 走 ∂H_θ/∂p，p-update 走 −∂H_θ/∂q），
Yoshida 由对称 leapfrog_hnn 基步以系数 (w₁, w₀, w₁), w₀ ≈ −1.7024 复合三次得到。
考察：Yoshida 是否改善 acceptance / Pf / COV，代价如何。

## 主结论
**Yoshida 在 HNN-HMC-SuS 中不带来任何 Pf / COV 收益，且成本约为 leapfrog 的 4.3×。**
Yoshida 所依赖的因果链——更小 ΔH → 更高 MH 接受率 → 更好采样 → 更准 Pf/COV——
在三个相互独立的环节各自断裂：

### Layer 0（积分器层 · 实用观察，依赖精度）
在学出来的非可分 H_θ 上、float32 下，Yoshida 的四阶优势在实用步长内不成立。
step-size 收敛研究（固定总时间 T，扫 h）显示：true 场上 leapfrog 斜率 ≈2、
Yoshida 大 h 段 ≈4（control 通过，积分器实现正确）；但 HNN 场上 dH_learned 的
收敛斜率退化为 leapfrog ≈1.7、Yoshida ≈1.0，两者在 h ≈ 0.07 处交叉，实用步长
（h ≤ 0.05）下 Yoshida 比 leapfrog 差 2–5×，且越细越差。
**机制（待 float64 确认）**：float32 下 NN 梯度评估的数值 roundoff 形成噪声地板，
随评估次数累积；Yoshida 每步多 4.5× 评估，噪声攒得更快，抵消其更小的截断误差。
注意：dH_learned 只涉及 H_θ 自身（autograd 精确梯度），与 HNN 的近似精度无关，
故此退化非模型误差所致。**此层依赖 float32，预计 float64 下 Yoshida 阶数恢复，
因此不作为根本结论，仅作实用区间内的附带发现。**

### Layer 1（采样层 · 根本，与精度无关）
即便 Yoshida 把 H_θ 守恒得很好，真正进入 MH 接受判据的 dH_true（对真哈密顿的能量
误差）被 **HNN 模型近似误差**托底：收敛研究中 dH_true 对 h 的斜率 ≈0（与步长解耦），
gaussian 地板 ≈7e-3、banana ≈1.35e-2。即"沿正确的流积分了一个错误的哈密顿"——
积分阶数无法触及这一项。故 Yoshida 不改善 acceptance。

### Layer 2（SuS 层 · 根本，本工作的 punchline）
即便 acceptance 改善了也无济于事，因为 SuS 中限制 acceptance 的 binding constraint
是**子集几何约束 G(q) ≤ b_k**，而非 MH 能量修正步。拒绝来源分解显示：几何拒绝占
65–92%（随层数加深递增），能量拒绝 < 0.5%，且两个积分器的拒绝构成几乎相同。
MH 能量步本身近乎 no-op，Yoshida 改善 ΔH 无处发力。

### 成本
Yoshida ≈ 4.3× 梯度评估（leapfrog 2×n_steps+1 = 21 calls/proposal；
Yoshida 9×n_steps = 90 calls/proposal，n_steps=10；每 call = 1 forward + 2 backward）。
Pf/COV 零收益对应 4.3× 成本 → 实用区间内净有害。

---

## 表格

### 表 1 · SuS 主结果（N_rep = 30，d = 2）
*[待 N_rep=30 完整跑替换；下为 N_rep=2 smoke 预览值，仅供占位]*

| Target | 积分器 | mean Pf | COV | Pf_ref | 总 grad evals | wall time |
|---|---|---|---|---|---|---|
| linear Gaussian LSF | leapfrog+hnn | [TBD] | [TBD] | 1.35e-3 | ~2.1e6 | [TBD] |
| linear Gaussian LSF | yoshida+hnn  | [TBD] | [TBD] | 1.35e-3 | ~9.1e6 (≈4.3×) | [TBD] |
| banana              | leapfrog+hnn | [TBD] | [TBD] | [ref]  | [TBD] | [TBD] |
| banana              | yoshida+hnn  | [TBD] | [TBD] | [ref]  | [TBD] (≈4.3×) | [TBD] |

判读要点：两积分器 Pf 应落在彼此 COV 区间内（无显著差异）；成本相差 ≈4.3×。

### 表 2 · 机制：每层拒绝来源分解
*[待确认：energy_rej 为"通过几何约束后"的条件率还是全体无条件率 — 见局限]*
*[下为 smoke 预览，待 N_rep=30 替换]*

| level | b_k | accept (lf) | accept (yo) | 几何拒绝占比 | 能量拒绝占比 |
|---|---|---|---|---|---|
| L1 | [TBD] | [TBD] | [TBD] | ~0.65 | ~0.001 |
| L2 | [TBD] | [TBD] | [TBD] | ~0.83 | ~0.001 |
| L3 | [TBD] | [TBD] | [TBD] | ~0.92 | ~0.000 |

### 表 3 · Step-size 收敛斜率（log-log 拟合，稳定值）

| 曲线 | Gaussian | Banana |
|---|---|---|
| true + leapfrog | +1.99 (≈2 ✓) | +1.99 (≈2 ✓) |
| true + yoshida | 大 h 段 ≈4（小 h 撞 float32 地板 ~1e-6）| 同左 |
| hnn + leapfrog (dH_learned) | +1.74 | +1.66 |
| hnn + yoshida (dH_learned) | +0.99 | +1.00 |
| hnn dH_true (floor) | +0.06 (≈0 ✓) | +0.07 (≈0 ✓) |
| crossover (yoshida 反超 leapfrog) | h ≈ 0.07 | h ≈ 0.07 |

---

## 与既有结论的联系
本工作与之前**真梯度 leapfrog-vs-Yoshida** 的结论闭环于同一机制：
SuS 的 binding constraint 是子集几何，不是 MH 能量步，故改善能量守恒（无论靠高阶
积分器还是别的）都无法转化为 Pf/COV 收益。HNN-Yoshida 从一个全新角度（学习哈密顿 +
高阶积分器）**独立复现**了这一 binding-constraint 结论（Layer 2），并新增了
**学习哈密顿特有的模型误差地板**（Layer 1）。

## 局限与待办
- **float64 复核**：确认 Layer 0 的 dH_learned 斜率退化是否为 float32 roundoff 地板
  （预计是）；这决定 Layer 0 的措辞，但不影响 Layer 1/2 主结论。
- **energy_rej 口径**：确认是否为"通过 G(q)≤b_k 之后"的条件拒绝率（最强版本）。
- **维度**：仅做 d=2（Gaussian LSF + banana）。子集几何机制预计与维度无关
  （真梯度工作已覆盖 d 至 100），但 HNN-Yoshida 高维需逐维训练 HNN，列为后续。
- **HNN 质量**：更准的 HNN 会降低 Layer 1 的 dH_true 地板，但不触及 Layer 2 与成本，
  主结论对 HNN 质量稳健。
