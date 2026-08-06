#  Automated Topological Band Crossing Analyzer

这是一个在研究过程中为了解决重复劳动而编写的自动化工具。
目前“能够正常工作并解决一定问题”，但代码结构仍在不断优化中，依然将存在bug。

本项目基于 `atomate2` 。
它的主要功能是**自动化执行包含自旋轨道耦合（SOC）的能带计算**，
自动调用 `irvsp` 寻找和标注能带中的偶然简并点（Accidental Degeneracy）。


Atomate2 文档：
https://materialsproject.github.io/atomate2/user/index.html

偶然简并的定义请参考：
[Yu Z M, Zhang Z, Liu G B, et al. Encyclopedia of emergent particles in three-dimensional crystals[J]. Science Bulletin, 2022, 67(4): 375-380.](https://www.sciencedirect.com/science/article/pii/S2095927321006927)

IRVSP 程序包仓库：
https://github.com/zjwang11/irvsp


##  工作流程 (Workflow)

对于放入输入文件夹中的每一个结构（`.cif` 或 `POSCAR`），脚本会自动执行以下任务：
1. **Relax**: 结构弛豫
2. **SCF (SOC)**: 自洽计算
3. **Band (SOC)**: Line-mode 能带计算
4. **One-Step Sym**: 低精度、打开对称性无 SOC 静态计算（仅为提取未破缺的对称性算符）
5. **Analysis**: 
   - 自动提取 `One-Step Sym` 的对称性信息，为 `Band` 的 `OUTCAR` 打补丁。
   - 调用 `irvsp` 计算不可约表示（Irreps）。
   - 检测 Gap（此处定义为上下能带能量差）极小值，检查不可约表示是否发生对换，捕捉偶然简并点。
   - 自动绘制带有交叉点标注的能带图，同时显示对换的表示。

![计算的 CaAgBi 的能带](https://github.com/user-attachments/assets/63cec81b-d32f-4f5d-86c4-4e34a1db2fe1)

结构性质与进一步分析请参考：
[Chen C, Wang S S, Liu L, et al. Ternary wurtzite CaAgBi materials family: A playground for essential and accidental, type-I and type-II Dirac fermions[J]. Physical Review Materials, 2017, 1(4): 044201.](https://journals.aps.org/prmaterials/abstract/10.1103/PhysRevMaterials.1.044201)

##  文件结构与功能

为了方便后续扩展，项目采用模块化设计。**请根据你要修改的功能，寻找对应的文件：**

* **`config.py`**
    - 配置参数。所有的 VASP 计算参数（`INCAR`）、K点密度、输入输出路径都在这里。
* **`batch_run.py`**  **(主入口)**
    - 遍历结构，提交 `jobflow` 任务。每个文件夹里是单个 VASP 任务。
* **`agent_runner.py`**
    - Agent 专用入口。读取 JSON 中的 POSCAR 绝对路径，为每个结构建立独立计算目录，并输出机器可读的 `band_report.json`。
* **`workflow_builder.py`**  
    - 定义工作流的装配逻辑（谁依赖谁）。
* **`degeneracy_analyzer.py`** 
    - 读取 `vasprun.xml` 和 `irvsp` 输出的 `outir` 数据，寻找能带交叉点。
* **`irvsp_parser.py`** 
    - 文本解析。负责用正则表达式把 `outir` 文件转换成字典。
* **`irvsp_runner.py`** 
    - 负责给 Band 步骤 `OUTCAR` 打上低精度计算获得的对称性补丁，并执行 `irvsp` shell 命令。这一步比较取巧，或许应有更好的解决方法。
* **`irrep_plotter.py`** 
    - 画图工具。
* **`test_plotter.py`**
    - 测试脚本。用于在获得了`vasprun.xml`和`outir`文件后测试分析和绘图。

## 💻 依赖环境

在运行前，请确保环境中已安装以下 Python 库：
```bash
pip install atomate2
```

系统要求：

- 参考 Atomate2 文档配置好运行 VASP 命令、POTCAR位置等。
- 必须确保系统已编译并配置好 irvsp 环境变量，使得在终端中直接输入 irvsp -sg $sgn 可以运行。

可先检查 Agent 所用环境：

```bash
python agent_runner.py --check
```

Agent 会按以下形式调用，不再依赖 `config.py` 中固定的输入输出目录：

```bash
python agent_runner.py --manifest structures_manifest.json --output-root band_analysis --report band_report.json
```

##  建议
### 想换一个体系算？

把你的 .cif 放进 structures_pool 文件夹，修改 config.py 里的 BASIC_INCAR，然后直接跑。

### 想优化“交叉点”的判定算法？

核心算法在 degeneracy_analyzer.py 的 find_crossings_by_irreps 方法里。传入的数据是结构化的字典了，你可以尽情发挥。

Happy Computing! May your calculations always converge!
