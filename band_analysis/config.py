import os

# ==========================================
# 1. 路径与全局开关配置
# ==========================================
INPUT_DIR = "./structures_pool"
OUTPUT_BASE_DIR = "./calculation_results"
OUTPUT_BAND_DIR = "./band_output"
ENABLE_SOC = True 

# ==========================================
# 2. 基础 INCAR 拼装块 (复用片段)
# ==========================================
BASIC_INCAR = {
    "ENCUT": 500, 
    "LCHARG": False, 
    "ALGO": "Normal", 
    "LREAL": "Auto",
    "PREC": "Accurate",
    "ISMEAR": 0,
    "SIGMA": 0.05
}

SOC_INCAR = {
    "LSORBIT": True, 
    "LMAXMIX": 4
}

BAND_PARALLEL_INCAR = {
    "KPAR": 4, 
    "NPAR": 1, 
    "LPLANE": False
}

# ==========================================
# 3. 各计算步骤具体 INCAR & KPOINTS 参数
# ==========================================

# A. 结构弛豫 (Relax)
RELAX_INCAR = {
    **BASIC_INCAR, 
    "EDIFF": 1e-5, 
    "EDIFFG": -0.01, 
    "ISPIN": 1, 
    "ISIF": 2, 
    "LCHARG": True
}

# B. 自洽计算 (SCF)
SCF_INCAR = {
    **BASIC_INCAR, 
    **SOC_INCAR, 
    "EDIFF": 1e-6, 
    "LCHARG": True
}
SCF_KPOINTS_DENSITY = 400 

# C. 能带计算1 (Standard Band)
BAND1_INCAR = {
    **BASIC_INCAR, 
    **SOC_INCAR, 
    **BAND_PARALLEL_INCAR, 
    "ISYM": 2, 
    "EDIFF": 1e-6, 
    "LWAVE": True, 
    "LCHARG": False
}
BAND1_KPOINTS_LINE_DENSITY = 80 

# D. 能带计算2 (Refined - 供后续扩展使用)
BAND2_INCAR = {
    **BASIC_INCAR, 
    **SOC_INCAR, 
    **BAND_PARALLEL_INCAR, 
    "EDIFF": 1e-6
}
BAND2_KPOINTS_NUM = 100

# E. 提取对称性算符的单步静态计算 (One step SCF)
ONE_STEP_INCAR = {
    **BASIC_INCAR, 
    "ISYM": 2,           # 强行开启对称性
    "SYMPREC": 1e-4,     # 放宽容差
    "LSORBIT": False,    # 关闭 SOC
    "ISPIN": 1           # 关闭自旋
}