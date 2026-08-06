# 集群统一环境

这个目录用于在 Linux/Slurm 集群上安装一个统一环境，同时运行：

- SymmCD `epoch699.ckpt` 条件结构生成
- MACE 本地势函数弛豫
- Pydantic AI + DeepSeek 对话 Agent
- atomate2/jobflow VASP 能带计算
- IRVSP 简并分析与 band 图片绘制

## 1. 目录布局

仓库已经把三个运行部分合并，不再需要三个同级源码目录：

```text
/work/$USER/SymmBand-Agent/
├── symmcd/
├── vendor/pydantic-ai/
├── band_analysis/
├── epoch699.ckpt
├── macemodel/
└── cluster_environment/
```

确认 `epoch699.ckpt` 和 `macemodel/2023-12-03-mace-128-L1_epoch-199.model` 已通过 Git LFS 拉取。使用本地 MACE 文件可以避免计算节点联网下载模型。

复制目录时必须包含 `.local_packages/` 隐藏目录；它为本地 pydantic-ai 源码提供版本元数据。推荐使用 `git clone` 和 `git lfs pull`。

## 2. 安装 Python 环境

推荐 GPU 版本，基线为 Python 3.11、PyTorch 2.5.1、CUDA 12.1：

```bash
cd /work/$USER/SymmBand-Agent/cluster_environment
bash install.sh conda gpu
conda activate symmcd-band-agent
python validate_environment.py --python-only
```

如果集群只能使用 `venv`：

```bash
bash install.sh venv gpu
source .venv/bin/activate
python validate_environment.py --python-only
```

CPU 调试环境把最后一个参数改为 `cpu`。CPU 可以检查 Agent 和工作流，但 500 步扩散采样会很慢。

安装脚本从 PyTorch 官方 CUDA 12.1 索引安装 Torch，并从 PyG 官方 `torch-2.5.1+cu121` wheel 页安装匹配的 `torch-scatter`。不要在安装完成后单独升级 Torch，否则 `torch-scatter` 的 ABI 会失配。

## 3. 配置 Agent

复制模板并编辑全部绝对路径和 DeepSeek key：

```bash
cp config/env.agent.example ../.env.agent
chmod 600 ../.env.agent
vi ../.env.agent
```

关键配置：

```dotenv
PYDANTIC_AI_SOURCE=/work/$USER/SymmBand-Agent/vendor/pydantic-ai
MACE_MODEL=/work/$USER/SymmBand-Agent/macemodel/2023-12-03-mace-128-L1_epoch-199.model
MACE_DEVICE=cuda
BAND_ANALYZER_ROOT=/work/$USER/SymmBand-Agent/band_analysis
BAND_PYTHON=/work/$USER/miniconda3/envs/symmcd-band-agent/bin/python
BAND_OUTPUT_ROOT=/work/$USER/SymmBand-Agent/band-results
```

## 4. 配置 VASP、POTCAR 和 IRVSP

VASP 和 POTCAR 受许可证约束，IRVSP 需要针对集群编译；它们不由 pip/conda 安装。先确认在计算节点上可以运行：

```bash
which vasp_std
which irvsp
irvsp 2>&1 | head
```

可以一次生成集群配置。先激活刚安装的环境并加载 VASP/IRVSP module，然后执行：

```bash
VASP_CMD="srun -n 32 /path/to/vasp_std" \
VASP_GAMMA_CMD="srun -n 32 /path/to/vasp_gam" \
VASP_NCL_CMD="srun -n 32 /path/to/vasp_ncl" \
POTCAR_ROOT="/path/to/vasp-potcar-root" \
IRVSP_BIN="/path/to/irvsp" \
bash configure.sh
```

`configure.sh` 会根据单仓库内部目录生成 `.env.agent`、`atomate2.yaml`、`jobflow.yaml` 和 `runtime.env`。已有 `.env.agent` 中的 `LLM_API_KEY` 会被保留，其余 Windows 路径会替换为集群路径；也可以通过环境变量 `LLM_API_KEY` 显式传入，或运行后直接编辑该文件。也可以不使用脚本，复制并编辑模板：

```bash
cp config/atomate2.yaml.example config/atomate2.yaml
cp config/jobflow.yaml.example config/jobflow.yaml
cp config/runtime.env.example config/runtime.env
vi config/atomate2.yaml
vi config/runtime.env
```

`atomate2.yaml` 中的 `srun -n 32` 必须与 Slurm 的 `--ntasks=32` 一致。能带代码设置了 `KPAR=4`，因此建议 MPI ranks 为 4 的倍数。`PMG_VASP_PSP_DIR` 应指向 pymatgen 能识别的 POTCAR 根目录，通常其中包含 `POT_GGA_PAW_PBE_54`。

每次登录或在 Slurm 脚本中执行：

```bash
conda activate symmcd-band-agent
source /work/$USER/SymmBand-Agent/cluster_environment/config/runtime.env
```

然后进行完整检查：

```bash
cd "$SYMMCD_ROOT"
python cluster_environment/validate_environment.py
python cluster_environment/validate_environment.py --python-only --load-models
python structure_agent.py --check-band
```

## 5. 提交任务

修改 `slurm/run_agent.slurm` 中的分区、module 和绝对路径，然后：

```bash
sbatch slurm/run_agent.slurm
```

也可以在已获得计算资源的交互节点运行：

```bash
cd "$SYMMCD_ROOT"
python structure_agent.py --prompt "我要生成10个194号空间群的NaBi结构，然后计算它的能带"
```

最终图片位于：

```text
<BAND_OUTPUT_ROOT>/NaBi_sg194/run_<timestamp>/bands/band_*.png
```

`10` 表示执行 10 次扩散采样。只有通过元素和空间群验收的 POSCAR 才进入 VASP，因此图片数量可能少于 10。

## 6. 常见问题

- `torch_scatter undefined symbol`：Torch 与 PyG wheel 不匹配。删除环境并重新执行 `install.sh`，不要单独升级 Torch。
- `POTCAR not found`：检查 `PMG_VASP_PSP_DIR` 和 POTCAR 目录命名。
- `irvsp not found`：在 Slurm 脚本中加载 IRVSP module，或把其目录加入 `PATH`。
- DeepSeek 连接失败：确认计算节点允许访问 `https://api.deepseek.com`；若集群禁止外网，需要在可联网节点运行 Agent 或配置代理。
- VASP MPI 失败：检查 `VASP_CMD` 的 `srun -n`、Slurm `--ntasks`、`KPAR=4` 和集群 module。
