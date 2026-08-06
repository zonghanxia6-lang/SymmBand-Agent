# irvsp_runner.py
import os
import re
import shutil
import subprocess
import logging
import gzip

# 1. 统一日志入口：不再自己配 Handler，而是接入全局的 "workflow" 体系
logger = logging.getLogger("workflow.runner")

def ensure_unzipped(filepath):
    """
    检查文件是否存在，如果不存在则尝试寻找 .gz 文件并解压。
    """
    if os.path.exists(filepath):
        return True
    
    gz_path = filepath + ".gz"
    if os.path.exists(gz_path):
        logger.info(f"发现压缩文件，正在自动解压: {gz_path} -> {filepath}")
        try:
            with gzip.open(gz_path, 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return True
        except Exception as e:
            logger.error(f"解压 {gz_path} 失败: {e}")
            return False
            
    return False

def patch_outcar(soc_outcar, sym_outcar):
    """
    专职文本处理：将 sym_outcar 中的对称性块替换到 soc_outcar 中
    """
    logger.info("开始提取对称性信息并写入 OUTCAR 补丁...")
    with open(sym_outcar, 'r', encoding='utf-8', errors='ignore') as f:
        sym_text = f.read()
    with open(soc_outcar, 'r', encoding='utf-8', errors='ignore') as f:
        soc_text = f.read()

    # 跨行匹配四大段对称性输出的正则
    pattern = r'(Analysis of symmetry for initial positions \(statically\):.*?Space group operators:.*?\n\s*\n)'
    
    sym_match = re.search(pattern, sym_text, re.DOTALL)
    if not sym_match:
        raise ValueError(f"未找到完整的对称性块，请确认 {sym_outcar} 对应的计算跑了 ISYM=2！")
    perfect_sym_block = sym_match.group(1)

    soc_match = re.search(pattern, soc_text, re.DOTALL)
    if not soc_match:
        raise ValueError(f"在 {soc_outcar} 中未找到需替换的对称性块！")
    broken_sym_block = soc_match.group(1)

    patched_soc_text = soc_text.replace(broken_sym_block, perfect_sym_block)

    # 备份原 OUTCAR 
    backup_outcar = soc_outcar + ".orig"
    if not os.path.exists(backup_outcar):
        shutil.copy2(soc_outcar, backup_outcar)

    # 写入补丁
    with open(soc_outcar, 'w', encoding='utf-8') as f:
        f.write(patched_soc_text)
    
    logger.info("OUTCAR 对称性补丁写入成功！")
    return backup_outcar

def run_irvsp_with_patched_outcar(soc_dir, sym_dir, sg_num, outir_filename="outir"):
    """
    主控流程：调度文件检查 -> 调用打补丁 -> 运行命令行
    """
    soc_dir = os.path.abspath(soc_dir)
    sym_dir = os.path.abspath(sym_dir)
    
    soc_outcar = os.path.join(soc_dir, "OUTCAR")
    sym_outcar = os.path.join(sym_dir, "OUTCAR")
    wavecar_path = os.path.join(soc_dir, "WAVECAR")
    
    # --- 1. 文件检查 ---
    if not ensure_unzipped(soc_outcar):
        raise FileNotFoundError(f"找不到 SOC OUTCAR: {soc_outcar}")
    if not ensure_unzipped(sym_outcar):
        raise FileNotFoundError(f"找不到 SYM OUTCAR: {sym_outcar}")
    if not ensure_unzipped(wavecar_path):
        logger.warning(f"未找到 WAVECAR，irvsp 可能会报错！路径: {wavecar_path}")

    # --- 2. 打补丁 ---
    backup_outcar = patch_outcar(soc_outcar, sym_outcar)

    # --- 3. 运行命令行 ---
    logger.info(f"开始调用 irvsp -sg {sg_num} ...")
    cmd = ["irvsp", "-sg", str(sg_num)]
    outir_path = os.path.join(soc_dir, outir_filename)
    
    try:
        with open(outir_path, "w", encoding='utf-8') as f_out:
            subprocess.run(
                cmd,
                cwd=soc_dir,
                stdout=f_out,               
                stderr=subprocess.PIPE,     
                text=True,
                check=True                  
            )
        logger.info(f"irvsp 运行成功！结果已保存至: {outir_path}")
        return outir_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"irvsp 运行失败！错误输出：\n{e.stderr}")
        # 失败时恢复原来的 OUTCAR
        if os.path.exists(backup_outcar):
            shutil.copy2(backup_outcar, soc_outcar)
        raise