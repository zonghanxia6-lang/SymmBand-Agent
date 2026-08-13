# 小样本、冻结 SymmCD 的粒子条件扩散调研与执行方案

## 1. 问题定义

目标是在不更新 `epoch699.ckpt` 中任何参数的前提下，使现有空间群条件 SymmCD 支持：

> 生成可能具有 DP、空间群 194 的晶体结构。

目前 DP 标签为 97 个严格正样本和 138 个兼容空间群难负样本。这里的“难负样本”必须满足：
SOC 能带和 IRVSP 已完成、该弛豫后空间群允许 DP、但没有检测到严格 DP。未计算结构和不允许
DP 的空间群不能作为负样本。

这一数据规模适合训练轻量性质分类器并引导冻结生成器，但不足以可靠地从头学习条件扩散分布，
也不足以支持对整个生成网络做全参数微调。

## 2. 2023--2026 年 7 月的相关工作

### 与“冻结生成器”直接相符的方法

| 年份 | 工作 | 是否冻结生成器 | 与本项目的关系 |
| --- | --- | --- | --- |
| 2023 | Universal Guidance for Diffusion Models, CVPRW | 是 | 在预测的去噪样本上计算任意外部目标函数的梯度，不需要重训扩散模型。 |
| 2023 | FreeDoM, ICCV | 是 | 用与时间无关的外部能量函数引导扩散，可组合多个条件。 |
| 2023 | Twisted Diffusion Sampler, NeurIPS | 是 | 用带权扩散粒子的 SMC 做条件采样；不需要条件训练，粒子数增加时具有渐近正确性，并已用于分子/蛋白结构任务。 |
| 2024 | Manifold Preserving Guided Diffusion, ICLR | 是 | 将引导更新限制在生成数据流形附近，降低强引导导致的无效样本。 |
| 2024 | TFG, NeurIPS | 是 | 统一训练免引导方法，在 7 个扩散模型、16 个任务和 40 个目标上系统搜索引导超参数，平均提高 8.5%。评测包含等变分子扩散。 |
| 2025 | Diffusion Classifier Guidance for Non-robust Classifiers, arXiv | 是 | 使用一步去噪预测和梯度稳定化，使只在干净数据上训练的普通分类器也能参与引导。 |
| 2025/2026 | PODGen, npj Computational Materials | 是 | 生成模型与性质分类器通过 MCMC 概率重加权组合；拓扑绝缘体比例由 3.10% 提高到 15.25%，约五倍。该方法不是扩散步内梯度引导，但可直接包裹冻结 SymmCD。 |

### 支持 adapter，但不支持“235 个标签足够”的方法

| 年份 | 工作 | 标签规模/特点 | 结论 |
| --- | --- | --- | --- |
| 2023 | A Closer Look at Parameter-Efficient Tuning in Diffusion Models | 图像少样本定制，约 0.75% 附加参数 | 证明冻结主干和小 adapter 的通用可行性，不是晶体或拓扑证据。 |
| 2025 | MatterGen, Nature | 约 605k 磁性、42k 带隙、5k 体模量标签 | adapter + classifier-free guidance 能提高材料条件命中；最小公开性质数据仍显著大于本项目。 |
| 2026-07 | Property-Guided Diffusion for Inverse Design of Crystalline Materials, arXiv | DiffCrysGen adapter + CFG | 支持参数高效晶体条件生成和多目标生成，但未证明百量级拓扑标签足够。 |

### 相关但不满足冻结 epoch699 的方法

- 2026 年 Nature Communications 的拓扑材料强化微调会更新生成模型，并使用基于 38,184 个
  拓扑材料训练的预测器，因此不能作为本项目 235 标签设置的直接依据。
- 2026 年超导体 guided diffusion 使用 7,183 个超导标签微调生成器，同样不是小样本冻结设置。
- 这些工作适合作为未来 RL/adapter 对照，不应作为当前首选实现。

## 3. 推荐架构

建议同时实现两个冻结 `epoch699` 的层级。第一级风险最低，第二级才是严格意义上的扩散轨迹条件引导。

### Level A：PODGen/SMC 式黑盒条件采样

保持 SymmCD 完全不变，训练校准后的 DP 分类器，目标分布定义为：

```text
pi(C | SG, DP) proportional to
    p_epoch699(C | SG)
    * P_DP(C)^alpha
    * P_stable(C)^beta
    * P_SG_retained(C)^gamma
```

PODGen 的严格 Metropolis-Hastings 接受率依赖 CrystalFormer 能给出结构序列概率，SymmCD 不直接
提供可用的完整晶体似然，因此不能原样复制 PODGen。对 SymmCD 应采用 Twisted Diffusion Sampler
(TDS) 风格的 Sequential Monte Carlo：在扩散过程中维持带权轨迹，用当前轨迹的一步去噪结构
`x0_hat` 计算 twisting weight，然后重采样。完整生成后的 MCMC/重排序可作为更简单的基线，
但不应声称具有 PODGen 相同的目标分布保证。SMC 支持随机森林等不可微分类器，最适合先验证
97+138 标签是否真的含有可用于富集 DP 的信号。

推荐参数起点：

- 64 条并行轨迹，每 25--50 个反向步重采样一次；
- 每次保留 50% 高权重轨迹，并为重复轨迹重新注入当前噪声水平的随机扰动；
- `alpha` 从 0.5、1、2、4 扫描；
- 有效性、空间群保持和 MACE 稳定性作为硬约束或独立权重；
- 对相同 StructureMatcher 簇设置数量上限，防止条件采样坍缩。

### Level B：TFG/Universal Guidance 式扩散步内引导

冻结 `epoch699`，只训练一个对结构可微的 DP 条件模型：

```text
p_phi(DP=1 | A_t, X_t, L_t, SG, t)
```

反向扩散时，在原有 SymmCD score 上加入分类器梯度：

```text
s_guided = s_epoch699
           + lambda_x(t) * projection_SG(grad_X log P_DP)
           + lambda_l(t) * projection_SG(grad_L log P_DP)
```

若分类器只在干净结构上训练，则先由 SymmCD 得到 `x0_hat`，在 `x0_hat` 上计算分类器梯度，
对应 Universal Guidance 和 2025 年 non-robust classifier guidance 的做法。更推荐直接使用与
SymmCD 完全相同的前向加噪过程训练噪声感知分类头：每个结构随机采样 `t`，生成
`(A_t, X_t, L_t)`，再预测 DP 标签。人工加噪可以增加训练观测，但不能当作新的独立物理样本。

空间群投影是本项目必须增加的物理约束。未经投影的坐标梯度可能破坏等价原子、Wyckoff 位置和
晶格约束。实现时应只更新 SymmCD/pyxtal 给出的对称性允许自由度，再由对称操作展开完整晶胞。

如果用户已经指定化学式，保持原子种类锁定，只引导坐标和晶格；这可避免对离散原子类型求梯度。
如果未指定化学式，先由 Level A 对候选元素组合做离散重采样，再用 Level B 优化连续结构变量。

## 4. 小样本 DP 分类器

### 表征与参数量

首选冻结的等变晶体编码器加小分类头，而不是从头训练 GNN：

1. 冻结 MACE 中间图表征，或复用冻结 SymmCD decoder 的节点隐变量；
2. 对节点特征做 attention/mean pooling；
3. 拼接 SG embedding、电子计数、SOC 强度统计和 Wyckoff/site-symmetry 描述；
4. 只训练 2 层、32--64 隐维的分类头，或 LoRA rank 4--8；
5. 训练 5 个化学体系分组模型形成 ensemble，同时输出概率与 epistemic uncertainty。

MACE 表征偏向势能面，而 DP 是电子结构性质，因此必须保留一个由组成、价电子、重元素、空间群和
Wyckoff 描述符构成的可解释基线。若冻结深度表征没有显著超过该基线，不应将其用于梯度引导。

### 数据划分

- 按化学体系做 StratifiedGroupKFold；相同 StructureMatcher 簇必须在同一折；
- SG 194 的最终前瞻测试不能参与阈值和引导强度选择；
- 对目标“DP + SG 194”，另外报告 SG 194 内部指标，避免模型仅学习“哪个空间群容易出现 DP”；
- 原点平移、等价晶胞、原子排列和保持空间群的小扰动只能用于训练折增强；
- 输出 PR-AUC、Brier score、校准曲线、precision@DFT-budget 和 enrichment factor。

### 启动引导的门槛

只有同时满足以下条件，才进入 Level B：

1. 化学体系分组 OOF PR-AUC 显著高于正例基率；
2. top-20% 的精度富集倍数至少为 1.5，bootstrap 95% CI 下界大于 1；
3. SG 194 子集也观察到富集，而不是仅由 SG 特征造成；
4. 概率校准可接受，且高置信预测不是集中在一个化学体系；
5. 对坐标和晶格的引导梯度在等价原子变换下保持一致。

如果这些条件不满足，问题不是扩散引导算法，而是 235 个标签尚未提供可泛化的结构--DP 信号；
此时应继续主动学习，不应加大 guidance scale。

## 5. 实验设计

固定空间群、化学式集合、随机种子、原始采样数、MACE 数量和 DFT 数量，比较：

| 组别 | epoch699 | 条件方法 | 是否修改主模型 |
| --- | --- | --- | --- |
| A | SG-only | 无 | 否 |
| B | SG-only | 完整生成后 DP 重排序 | 否 |
| C | SG-only | Level A SMC/MCMC | 否 |
| D | SG-only | Level B 梯度引导 | 否 |
| E | SG-only | Level A + Level B | 否 |
| F | SG + DP adapter | CFG | 是，仅作为未来对照 |

每组至少生成 1,000 个原始候选、使用 5 个随机种子。主要指标不是代理模型判断的 DP 比例，而是
固定 DFT 预算下经 SOC 能带和 IRVSP 严格确认的 DP 数量及相对 A 组的富集倍数。次要指标包括：
结构有效率、弛豫后 SG 保持率、StructureMatcher 唯一率、MP20 新颖率、MACE/DFT 稳定率和化学
体系多样性。

引导强度必须报告完整 Pareto 曲线。强引导通常提高条件命中但降低多样性和物理有效性，不能只展示
最佳单点。推荐 `lambda={0, 0.25, 0.5, 1, 2, 4}`，并使用早期为 0、中期升高、末期减弱的时间
调度；同时限制单步引导梯度范数不超过原 score 范数的 10%--30%。

## 6. 结论

对于现有 97+138 个 DP 标签，最可信的路线是：

1. 先用冻结表征训练经过分组验证和概率校准的 DP 分类器；
2. 先做 PODGen/SMC 黑盒条件采样，证明固定预算富集有效；
3. 再实现带空间群梯度投影的 TFG/Universal Guidance；
4. 在前瞻 DFT 标签扩展到至少 300 正、300 难负后，才把 MatterGen 风格 adapter 纳入主实验。

这一路线从头到尾保持 `epoch699` 参数不变，并且 Level B 属于真正的条件扩散采样，而不是简单的
生成后筛选。其主要创新点可以表述为：小样本粒子分类器、空间群流形投影、冻结对称性扩散模型和
DFT 主动学习闭环的组合。

## 主要参考文献

- Bansal et al., Universal Guidance for Diffusion Models, CVPRW 2023,
  https://openaccess.thecvf.com/content/CVPR2023W/GCV/html/Bansal_Universal_Guidance_for_Diffusion_Models_CVPRW_2023_paper.html
- Yu et al., FreeDoM, ICCV 2023,
  https://openaccess.thecvf.com/content/ICCV2023/html/Yu_FreeDoM_Training-Free_Energy-Guided_Conditional_Diffusion_Model_ICCV_2023_paper.html
- Wu et al., Practical and Asymptotically Exact Conditional Sampling in Diffusion Models,
  NeurIPS 2023, https://papers.nips.cc/paper_files/paper/2023/hash/63e8bc7bbf1cfea36d1d1b6538aecce5-Abstract-Conference.html
- He et al., Manifold Preserving Guided Diffusion, ICLR 2024,
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/c355566ce402de341c3320cf69a10750-Abstract-Conference.html
- Ye et al., TFG: Unified Training-Free Guidance for Diffusion Models, NeurIPS 2024,
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/2818054fc6de6dacdda0f142a3475933-Abstract-Conference.html
- Zeni et al., A generative model for inorganic materials design, Nature 2025,
  https://doi.org/10.1038/s41586-025-08628-5
- Ye et al., Materials discovery acceleration by using conditional generative methodology,
  npj Computational Materials, published online 2025, volume 12 (2026),
  https://www.nature.com/articles/s41524-025-01930-w
- Vaeth et al., Diffusion Classifier Guidance for Non-robust Classifiers, arXiv 2025,
  https://arxiv.org/abs/2507.00687
- Mal et al., Property-Guided Diffusion for Inverse Design of Crystalline Materials,
  arXiv, July 2026, https://arxiv.org/abs/2607.21849
