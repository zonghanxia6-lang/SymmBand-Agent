# 物理分析能力提升方案

## 目标

把 Agent 从“调用计算并转述结果”提升为“给出证据可追溯、结论强度校准、能够主动暴露反例与验证路径的物理分析助手”。本方案不依赖模型记忆来补全计算事实；核心物理判断尽量由确定性代码生成，再由语言模型解释。

## 五个提升角度

1. **物理问题建模**：回答前明确体系、近似、能量参考、SOC、磁序、边界条件、温压等会改变结论的条件。
2. **数值与物理约束**：保留单位、量纲、阈值余量、费米能级距离、收敛状态；用守恒律、对称性和极限情况交叉检查。
3. **证据等级与结论边界**：区分直接输出、推断和未计算项。路径唯一匹配只能提高“分类证据”，不能替代三维拓扑不变量。
4. **工具链闭环**：结构生成 → 势能/弛豫 → SOC 能带 → IRVSP 表示 → 局部 k 加密 → 三维搜索/拓扑不变量 → 稳定性验证，各阶段输出可机器读取的证据。
5. **专项评测与持续改进**：固定问题集同时检查必须出现的物理概念和禁止出现的过度断言，比较模型或提示词版本时使用同一套题。

## 本次已实施

- `physics_evidence.py`：统一参数校验、交叉点诊断、质量检查、证据阶梯、结论边界和后续验证建议。
- `band_analysis/degeneracy_analyzer.py`：不再丢弃 detector 内部的最小能隙、相邻点能隙和最小值所在 k 点。
- `band_result_analysis.py`：报告 `E-E_F`、最小能隙、阈值比、费米邻近等级、电子收敛状态、路径 k 点数及完整证据审计。
- 新增语义安全字段 `path_unique_particle_types`；旧的 `confirmed_particle_types` 只作为兼容别名保留，避免字段名诱导过度结论。
- `structure_agent.py`：加入统一物理推理协议，并明确 MACE/DFT/形成能/凸包、有限 k 网格/严格简并、路径分类/拓扑确认之间的边界。
- `evals/physics_analysis_cases.json` 与 `physics_eval.py`：加入六类物理过度断言回归题及离线评分器。

## 证据阶梯

- `L0_no_candidate_within_search_scope`：在给定路径、能窗和阈值内未检出；不等于整个布里渊区不存在。
- `L2_symmetry_supported_crossing_candidate`：小能隙局部极小值并伴随相邻能带 IRVSP 表示对换。
- `L3_path_taxonomy_unique_symmetry_candidate`：在 L2 基础上，百科路径分类只有一个候选；仍不构成拓扑确认。
- 更强结论需要额外实现并通过局部 k 收敛、三维简并流形、Berry/Wilson/Chern 指标及边界态验证。

## 如何运行回归评测

先让待比较的 Agent 回答 `evals/physics_analysis_cases.json` 中的问题，把答案保存成以 case id 为键的 JSON：

```powershell
python physics_eval.py answers.json --output physics_eval_report.json
```

单元测试验证确定性证据层：

```powershell
python -m unittest tests.test_physics_evidence tests.test_band_result_analysis -v
```

## 下一阶段建议

1. 对每个 crossing 自动生成局部三维 k 网格并拟合最低两带的有效哈密顿量。
2. 接入 Wannier90/WannierTools，计算 Berry phase、Wilson loop、Chern number 与表面谱。
3. 增加 ENCUT、k 网格、SOC、磁序和结构扰动的收敛矩阵，把“结果稳定性”变成报告字段。
4. 增加声子、弹性常数、有限温分子动力学和竞争相凸包，独立评估动力学、机械、热与热力学稳定性。
5. 收集专家标注的错误案例，对“结论强度、必要前提、数值引用、反例意识”分别计分，而不仅看最终术语是否命中。
