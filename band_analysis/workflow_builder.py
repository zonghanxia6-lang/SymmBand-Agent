import os
import logging
from jobflow import job, Flow, Response
from atomate2.vasp.jobs.core import RelaxMaker, StaticMaker, NonSCFMaker
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet, MPNonSCFSet
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from irvsp_runner import run_irvsp_with_patched_outcar
from irrep_plotter import plot_irrep_crossings
from degeneracy_analyzer import DegeneracyAnalyzer
from config import *

logger = logging.getLogger("workflow.builder")

@job
def degeneracy_check_job(structure, band_dir, sym_dir, output_band_dir, task_name=""):
    """
    步骤 5: 分析 Job (打补丁 -> 算 Irrep -> 读取分析 -> 绘图标注)
    """
    band_path = str(band_dir).split(":")[-1]
    sym_path = str(sym_dir).split(":")[-1]
    
    logger.info(f"[{task_name}] 开始拓扑分析...")
    
    # 1. 查空间群
    sga = SpacegroupAnalyzer(structure)
    sg_num = sga.get_space_group_number()

    # 2. 打补丁并获取 outir
    try:
        outir_file = run_irvsp_with_patched_outcar(soc_dir=band_path, sym_dir=sym_path, sg_num=sg_num)
    except Exception as e:
        logger.error(f"[{task_name}] irvsp 执行失败: {e}")
        return Response()

    # --- 👇 下面是解耦后的核心逻辑 👇 ---
    
    xml_path = os.path.join(band_path, "vasprun.xml.gz")
    if not os.path.exists(xml_path):
        xml_path = os.path.join(band_path, "vasprun.xml")
        
    # 3. 数据分析（只算数据）
    analyzer = DegeneracyAnalyzer(xml_path)
    if not analyzer.valid:
        logger.error(f"[{task_name}] 无法解析 xml，终止分析。")
        return Response()
    
    crossings = analyzer.find_crossings_by_irreps(outir_file)

    # 4. 数据可视化（只管画图）
    filename = f"band_{task_name}.png"
    output_png = os.path.join(output_band_dir, filename)

    plot_irrep_crossings(
        bs=analyzer.bs,             # 传入能带数据
        crossings=crossings,        # 传入计算出的交叉点列表
        output_filename=output_png, 
        material_info=task_name,
        y_lim=[-1.0, 1.0] 
    )
    
    # ------------------------------------

    logger.info(f"[{task_name}] 任务链圆满结束！")
    
    wavecar_path = os.path.join(band_path, "WAVECAR")
    if os.path.exists(wavecar_path):
        os.remove(wavecar_path)
        
    return Response()

def get_makers(enable_soc=False):
    """ 定义并返回各个计算步骤的 Maker """
    relax_set = MPRelaxSet(user_incar_settings=RELAX_INCAR, user_kpoints_settings={"reciprocal_density": SCF_KPOINTS_DENSITY})
    relax_maker = RelaxMaker(input_set_generator=relax_set)

    static_set = MPStaticSet(user_incar_settings=SCF_INCAR, user_kpoints_settings={"reciprocal_density": SCF_KPOINTS_DENSITY})
    static_maker = StaticMaker(input_set_generator=static_set)
    
    band_set = MPNonSCFSet(user_incar_settings=BAND1_INCAR, kpoints_line_density=BAND1_KPOINTS_LINE_DENSITY, validate_magmom=False, mode="line")
    band_maker = NonSCFMaker(input_set_generator=band_set)

    one_step_set = MPStaticSet(user_incar_settings=ONE_STEP_INCAR, user_kpoints_settings={"reciprocal_density": 100})
    one_step_maker = StaticMaker(input_set_generator=one_step_set)

    return relax_maker, static_maker, band_maker, one_step_maker

def build_degeneracy_flow(raw_structure, task_name, output_band_dir):
    """
    组装完整的 Jobflow 工作流
    """
    # 0. 标准化结构
    sga = SpacegroupAnalyzer(raw_structure, symprec=0.01)
    std_structure = sga.get_primitive_standard_structure()
    sg_info = sga.get_space_group_symbol()
    logger.info(f"[{task_name}] 已将结构标准化为: {sg_info} 标准原胞")
    
    # 1. 获取 Makers
    relax_maker, static_maker, band_maker, one_step_maker = get_makers(enable_soc=ENABLE_SOC)
    
    # 2. 创建 Jobs 并定义依赖关系
    relax_job = relax_maker.make(std_structure)
    relax_job.name = "relax"
    
    static_job = static_maker.make(structure=relax_job.output.structure, prev_dir=relax_job.output.dir_name)
    static_job.name = "scf_soc"

    band_job = band_maker.make(structure=static_job.output.structure, prev_dir=static_job.output.dir_name, mode="line")
    band_job.name = "band_soc"
    
    one_step_job = one_step_maker.make(structure=relax_job.output.structure, prev_dir=relax_job.output.dir_name)
    one_step_job.name = "one_step_sym"
    
    analysis_job = degeneracy_check_job(
        structure=relax_job.output.structure, 
        band_dir=band_job.output.dir_name, 
        sym_dir=one_step_job.output.dir_name, 
        output_band_dir=output_band_dir,
        task_name=task_name
    )
    analysis_job.name = "degeneracy_check"

    # 3. 打包返回 Flow
    return Flow([relax_job, static_job, band_job, one_step_job, analysis_job], name=f"{task_name}_workflow")
