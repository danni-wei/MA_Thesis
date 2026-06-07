# 导师汇报提纲 — Yoshida × HNN-HMC in SuS

## 一句话 takeaway（先抛这个）
在 HNN-HMC-SuS 里把 leapfrog 换成四阶 Yoshida，**对 Pf/COV 零收益、成本 ≈4.3×**；
根本原因是 SuS 的 binding constraint 是子集几何约束、不是 MH 能量步——这与我们之前
真梯度 Yoshida 的发现是同一机制，HNN 设定从新角度独立复现了它，并额外揭示了学习
哈密顿特有的模型误差地板。

## 讲述顺序（先抛 punchline，再补支撑）
1. **punchline 先行（Layer 2）**：展示拒绝来源分解——几何拒绝 65–92%、能量拒绝 <0.5%、
   两积分器相同。MH 能量步近乎 no-op，所以"更好的 ΔH"无处发力。
2. **为什么 ΔH 本身也帮不上（Layer 1）**：dH_true 被 HNN 模型误差托底，收敛斜率 ≈0、
   与 h 无关。"沿正确的流积分了一个错误的哈密顿。"
3. **实用区间内甚至更差（Layer 0，标注为依赖精度）**：收敛研究 + crossover 图。
   明确说这是 float32 下的实用观察，机制（roundoff 噪声地板）待 float64 确认，
   **不作为根本结论**。
4. **成本**：4.3× 梯度评估，零收益 → 净有害。
5. **闭环**：与真梯度 Yoshida 同机制，叙事统一。

## 要展示的图（按顺序）
- **图 A（镇场）**：每层拒绝来源柱状图（几何 vs 能量），两积分器并列 → Layer 2。
- **图 B**：step-size 收敛图，含 true 场 control（斜率 2/4 分离）、HNN 场 dH_learned
  两条（斜率退化 + crossover）、dH_true 水平地板线 → 一张图覆盖 Layer 0 和 Layer 1。
- **图 C**：Pf vs SuS level，两积分器 + 参考线 + COV 误差棒 → 主结果。
- 成本：一句话/一个小表，4.3×。

---

## 预判导师问题（Ullmann，scientific computing / UQ，会问得很细）

**Q1. energy_rej 是条件率还是无条件率？**
答：[确认后填] 应为"在通过几何约束 G(q)≤b_k 的候选中"的条件拒绝率。若如此，则是最强
版本——连几何上合法的候选 MH 也几乎不拒，能量步彻底是 no-op。*（汇报前务必和代码对齐这一点）*

**Q2. 你的 h 是不是在 asymptotic regime 之外，所以才看不到 Yoshida 优势？**
答：做了 step-size 收敛研究排除这点。true 场在同一 h 范围内 leapfrog/Yoshida 干净给出
斜率 2 和 4（control 通过），说明 h 范围没问题；HNN 场的斜率退化是 HNN 场特有的现象，
非步长选取问题。

**Q3. Yoshida 在 HNN 场更差，是 float32 artifact 还是根本性的？**
答：很可能是 float32 roundoff 噪声地板（true+Yoshida 也在 1e-6 处撞 float32 地板，
HNN 梯度过 NN forward+backward roundoff 更大，地板更高）。float64 复核会确认
（预计 Yoshida 阶数恢复）。**但这只影响 Layer 0 的措辞；Layer 1（模型误差地板）和
Layer 2（子集几何）与精度无关，主结论不变。** —— 这条要主动讲，别等导师挖。

**Q4. dH_true 和 dH_learned 怎么定义/测量？**
答：同一条由 ∂H_θ 向量场积分出的轨迹，dH_learned = 沿轨迹 H_θ 的变化（量积分器对学习
哈密顿的守恒）；dH_true = 沿同一轨迹真哈密顿 H 的变化（量进入 MH 判据的那个量）。
dH_learned 只依赖 H_θ 和积分器，与 HNN 准不准无关。

**Q5. 非可分的学习哈密顿用显式 leapfrog 合理吗？它还是辛的吗？**
答：诚实回答——显式 leapfrog 严格只对可分 H 是辛的；对非可分 H_θ 是 HNN-HMC 文献常用的
显式近似，非严格辛（严格辛需 implicit 或扩展相空间，见 Tao 2016）。但本研究在固定轨迹
长度下考察能量误差，由截断阶（+数值噪声）主导，结论不依赖严格辛性。

**Q6. 有没有在相同计算预算下比较？**
答：主对比口径是相同每层样本数 N（标准 SuS 配置），如实报告 Yoshida 多花 4.3×。
若按相同梯度预算对齐，Yoshida 只能跑约 1/4.3 的样本 → COV 更差。两种口径 Yoshida 都不占优。

**Q7. 只做了 d=2，结论能否推广？**
答：Layer 2（子集几何 binding）机制预计与维度无关——真梯度 Yoshida 工作已覆盖 d 至 100
得到一致结论。HNN-Yoshida 高维需逐维训练 HNN，列为后续工作。

**Q8. 几何拒绝为何随层数加深递增（0.65→0.92）？**
答：深层子集阈值 b_k 更紧、可行域更小，HMC 提议更容易落到子集外 → 几何拒绝占比升高。
这正是子集几何在深层主导的直接体现（也与 PINN 在深层 collapse 的现象同源）。

**Q9. 既然高阶积分器没用，那什么才有用？**
答：瓶颈是几何性的（提议离开子集），杠杆应在**提议几何/自适应**——step size、mass matrix、
约束感知提议——而非积分器阶数。这是更有方向性的后续。

**Q10. 这个 negative result 的贡献是什么？**
答：它精确定位了 *为什么* 高阶辛积分器在 SuS-HMC（含 HNN 设定）中不起作用——
binding constraint 是子集几何而非能量修正；并在学习哈密顿设定下识别出模型误差地板。
这为"该往哪使劲"提供了清晰的负向边界，是干净、可发表/可展示的结论。

---

## 汇报前 checklist
- [ ] N_rep=30 完整跑出 Pf/COV，替换表 1、表 2 占位
- [ ] 和代码确认 energy_rej 的口径（Q1）
- [ ] （可选）float64 收敛复核，定 Layer 0 措辞（Q3）
- [ ] 三张图导出、标题/图例清晰（斜率值标在图例里）
