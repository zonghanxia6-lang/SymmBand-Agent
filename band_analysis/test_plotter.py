# test_plotter.py
import os
import logging
from logger_setup import setup_logger

# 引入重构后的模块
from degeneracy_analyzer import DegeneracyAnalyzer
from irrep_plotter import plot_irrep_crossings

def main():
    """
    专为调试画图和分析算法设计的独立测试脚本 (单元测试)
    """
    # 1. 初始化一个独立的调试 Logger (不需要写文件，终端打印即可)
    logger = setup_logger("debug_plot", level=logging.INFO)
    logger.info("Starting standalone plotting test...")

    # ==========================================
    # 请在这里手动填入你的测试文件路径
    # ==========================================
    test_vasprun_xml = "/path/to/vasprun.xml(.gz)"
    test_outir       = "/path/to/outir"
    
    # 指定输出的 PNG 文件名
    material_name = "your_material_name"
    output_png = f"debug_plot_{material_name}.png"
    # ==========================================

    # 路径校验
    if not os.path.exists(test_vasprun_xml):
        logger.critical(f"找不到 vasprun.xml: {test_vasprun_xml}")
        return
    if not os.path.exists(test_outir):
        logger.critical(f"找不到 outir: {test_outir}")
        return

    logger.info(f"Processing data for: {material_name} ...")

    try:
        # 2. 纯分析步骤：计算交叉点
        logger.info("Step 1: Analyzing data...")
        analyzer = DegeneracyAnalyzer(test_vasprun_xml)
        if not analyzer.valid:
            logger.error("无法实例化 Analyzer，xml 解析失败。")
            return
        
        # 使用我们在 Analyzer 里已经净化好的混合算法
        crossings = analyzer.find_crossings_by_irreps(test_outir)
        logger.info(f"Found {len(crossings)} topological crossings.")

        # 3. 纯绘图步骤：生成配图
        logger.info("Step 2: Plotting and customizing...")
        
        # 注意我们在 logger_setup 里通过“父子 Logger继承”原理
        # irrep_plotter 默认用的是 workflow.plotter，为了在终端看到输出，
        # 我们这里临时把子模块日志劫持到控制台。
        # (这是一种临时调试手段，你可以忽略这行代码的原理)
        logging.getLogger("workflow.plotter").setLevel(logging.INFO)

        plot_irrep_crossings(
            bs=analyzer.bs,             # 传入能带对象
            crossings=crossings,        # 传入算法分析结果
            output_filename=output_png, 
            material_info=material_name,
            # 指定纵坐标范围为老师要求的 +-1
            y_lim=[-1.0, 1.0] 
        )
        
        logger.info(f"Test finished! Check your output: {output_png}")

    except Exception as e:
        logger.exception(f"Test failed with error:")

if __name__ == "__main__":
    main()
