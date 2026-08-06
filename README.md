# SymmBand-Agent

SymmBand-Agent 是一个对话式晶体结构生成与能带计算项目。用户可以直接输入：

```text
我要生成10个194号空间群的NaBi结构，然后计算它的能带
```

Agent 会依次执行：

1. Pydantic AI 调用 DeepSeek，将自然语言解析为化学式、空间群号和采样数。
2. SymmCD 使用 `epoch699.ckpt` 进行条件扩散采样。
3. MACE 对生成结构进行弛豫，并筛选元素和空间群都符合要求的 POSCAR。
4. atomate2/jobflow 对合格 POSCAR 执行 VASP Relax、SOC-SCF、SOC-Band 和对称性计算。
5. IRVSP 分析不可约表示，最终生成 `band_*.png`。

## 目录结构

```text
SymmBand-Agent/
├── structure_agent.py          对话 Agent 主入口
├── workflow_sym.py             SymmCD + MACE 结构生成
├── band_workflow.py            能带子进程桥接
├── symmcd/                     checkpoint 推理所需源码
├── band_analysis/              atomate2/VASP/IRVSP 能带工作流
├── vendor/pydantic-ai/         Pydantic AI 运行时源码快照
├── epoch699.ckpt               SymmCD checkpoint（Git LFS）
├── macemodel/                  本地 MACE 模型（Git LFS）
├── cluster_environment/        Conda/pip、集群和 Slurm 配置
├── tests/                      Agent 与桥接层测试
└── scripts/                    GitHub 仓库初始化脚本
```

## 快速开始

Linux/GPU 集群推荐：

```bash
cd cluster_environment
bash install.sh conda gpu
conda activate symmcd-band-agent
cd ..
```

复制并编辑 Agent 配置：

```bash
cp .env.agent.example .env.agent
chmod 600 .env.agent
vi .env.agent
```

配置完成后检查：

```bash
python cluster_environment/validate_environment.py --python-only
python structure_agent.py --show-config
python structure_agent.py --check-band
```

运行：

```bash
python structure_agent.py
python structure_agent.py --prompt "我要生成10个194号空间群的NaBi结构，然后计算它的能带"
```

完整的 VASP、POTCAR、IRVSP 和 Slurm 配置见
[`cluster_environment/README.md`](cluster_environment/README.md)。

## 输出

结构默认输出到：

```text
generated_structures/<formula>_sg<spacegroup>/run_<timestamp>/
```

能带图片默认输出到对应任务的：

```text
band_analysis/bands/band_*.png
```

如果设置 `BAND_OUTPUT_ROOT`，图片会写入：

```text
<BAND_OUTPUT_ROOT>/<formula>_sg<spacegroup>/run_<timestamp>/bands/
```

“生成 10 个”表示执行 10 次扩散采样。只有通过最终元素和空间群验收的结构才进入 VASP，所以图片数量可能少于 10。

## 上传 GitHub

`epoch699.ckpt` 约 725 MB，超过 GitHub 普通 Git 的 100 MiB 单文件限制，必须使用 Git LFS。项目已经在 `.gitattributes` 中配置好模型规则。

Windows PowerShell：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\prepare_github.ps1
```

Linux/macOS/Git Bash：

```bash
bash scripts/prepare_github.sh
```

检查 LFS 和暂存内容后提交：

```bash
git status --short
git lfs ls-files
git commit -m "Initial SymmBand-Agent monorepo"
gh repo create SymmBand-Agent --source=. --private --push
```

首次建议创建私有仓库。确认 `THIRD_PARTY_NOTICES.md` 中提到的 SymmCD、能带代码和模型权利后，再决定是否公开。

## 安全与版本控制

- `.env.agent`、API key、POTCAR、VASP 输出、生成结构和 band 结果均被 `.gitignore` 排除。
- 仓库只保留 Pydantic AI 运行所需的两个包，并完整保留其 MIT 许可证。
- 克隆仓库前需要安装 Git LFS，然后执行 `git lfs pull` 获取模型文件。
- 不要将 POTCAR 上传到 GitHub。

## 许可证

Pydantic AI 的 MIT 许可证保存在 `vendor/pydantic-ai/`。其余代码与模型在原目录中没有附带许可证；公开仓库前请先确认并补充相应授权。详见 `THIRD_PARTY_NOTICES.md`。
