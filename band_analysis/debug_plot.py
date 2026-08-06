import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg') # 保证服务器端无头模式运行
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.io.vasp import Vasprun

# 引入分析器
try:
    from degeneracy_analyzer import DegeneracyAnalyzer
except ImportError:
    print("错误: 当前目录下找不到 degeneracy_analyzer.py")
    sys.exit(1)

def get_scalar(val):
    """最暴力的标量提取函数，防止 numpy 报错"""
    try:
        if hasattr(val, '__len__'): # 如果是数组/列表
            flat = np.array(val).flatten()
            if len(flat) > 0:
                return float(flat[0])
            return 0.0
        return float(val)
    except:
        return 0.0

def execute_plotting(xml_path, output_filename="debug_band_crossings.png"):
    print(f"--- 处理文件: {xml_path} ---")

    # 1. 初始化分析器 (用来找红圈)
    try:
        analyzer = DegeneracyAnalyzer(xml_path)
    except Exception as e:
        print(f"读取 vasprun 失败: {e}")
        return

    crossings, _ = analyzer.find_crossings(
        gap_tolerance=0.02, 
        sym_tolerance=0.002, 
        e_window=1.0
    )
    print(f"发现交叉点: {len(crossings)} 个")

    # 2. 准备绘图数据 (用来画线)
    bs = analyzer.bs # 直接复用 BS 对象，确保源头一致
    if bs is None:
        print("错误: BandStructure 为空")
        return

    plotter = BSPlotter(bs)
    if callable(plotter.bs_plot_data):
        data = plotter.bs_plot_data(zero_to_efermi=True)
    else:
        data = plotter.bs_plot_data

    # ================= 修正偏移量的核心逻辑 =================
    
    # A. 获取 BS 中的绝对能量 (Band 0, K 0)
    # 只要数据里有，就肯定能取到
    spin_keys = list(bs.bands.keys())
    spin_key_bs = spin_keys[0]
    ref_val_abs = get_scalar(bs.bands[spin_key_bs][0][0])

    # B. 获取 Plotter 中的绘图能量 (Band 0, K 0)
    # 这里的 data['energy'] 结构比较乱，可能分段(list)也可能不分(dict)
    e_data = data['energy']
    ref_val_plot = 0.0
    
    if isinstance(e_data, list):
        # 分段模式: List[Dict[Spin, List[List[float]]]]
        branch0 = e_data[0]
        s_key = list(branch0.keys())[0]
        ref_val_plot = get_scalar(branch0[s_key][0][0])
        is_branched = True
    elif isinstance(e_data, dict):
        # 扁平模式: Dict[Spin, List[List[float]]]
        s_key = list(e_data.keys())[0]
        ref_val_plot = get_scalar(e_data[s_key][0][0])
        is_branched = False
    else:
        print("错误: 无法识别 data['energy'] 结构")
        return

    # C. 计算偏移量
    # Shift = 绝对值 - 绘图值
    # 之后我们计算红圈坐标时：Y = (红圈绝对能量) - Shift
    plotter_shift = ref_val_abs - ref_val_plot
    
    print(f"检测到能量偏移 (Shift): {plotter_shift:.4f} eV")
    print(f"  (BS绝对值: {ref_val_abs:.4f}, Plot绘图值: {ref_val_plot:.4f})")
    
    # =======================================================

    # 3. 绘图
    plot_obj = plotter.get_plot(ylim=[-1.5, 1.5], zero_to_efermi=True)
    
    # 拿到 Axes 对象
    if isinstance(plot_obj, matplotlib.axes.Axes):
        ax = plot_obj
        fig = ax.get_figure()
    elif hasattr(plot_obj, "gca"):
        ax = plot_obj.gca()
        fig = plot_obj.gcf()
    else:
        fig = plt.gcf()
        ax = plt.gca()

    # 4. 叠加红圈
    for i, cross in enumerate(crossings):
        # --- 算 X 坐标 ---
        x_coord = 0.0
        
        if is_branched:
            # 分段逻辑
            target_start = cross['branch_info']['start_index']
            # 找到对应的 branch index
            b_idx = -1
            for idx, branch in enumerate(bs.branches):
                if branch['start_index'] == target_start:
                    b_idx = idx
                    break
            
            if b_idx != -1 and b_idx < len(data['distances']):
                dists = data['distances'][b_idx]
                k_loc = cross['k_index_local']
                if k_loc < len(dists):
                    x_coord = dists[k_loc]
                else: continue
            else: continue
        else:
            # 扁平逻辑 (对应之前的 KeyError 情况)
            k_glob = cross['k_index_global']
            # data['distances'] 可能是 [d1, d2...] 也可能是 [[d1..], [d2..]]
            # 无论哪种，我们都把它展平成一维数组来查
            all_dists = np.array(data['distances']).flatten()
            if k_glob < len(all_dists):
                x_coord = all_dists[k_glob]
            else: continue

        # --- 算 Y 坐标 (修正后) ---
        # analyzer 算出的 energy 是相对 E_fermi 的，先还原回绝对能量
        # cross['energy'] = E_abs - bs.efermi
        abs_energy = cross['energy'] + bs.efermi
        
        # 减去 plotter 的偏移量，对齐到图上
        y_coord = abs_energy - plotter_shift

        ax.scatter(
            x_coord, y_coord,
            color='red', marker='o', s=80,
            facecolors='none', edgecolors='red', linewidth=2,
            zorder=20, label="Crossing" if i == 0 else ""
        )

    # 5. 保存
    try:
        if crossings:
            ax.legend(loc='upper right', framealpha=0.9)
        ax.set_title(f"Refined Bands (Corrected Offset)")
        fig.tight_layout()
        fig.savefig(output_filename, dpi=300)
        print(f"保存成功: {output_filename}")
    except Exception as e:
        print(f"保存失败: {e}")

if __name__ == "__main__":
    # 自动找文件
    target_xml = "vasprun.xml"
    if os.path.exists("vasprun.xml.gz"):
        target_xml = "vasprun.xml.gz"
    elif not os.path.exists(target_xml):
        print("找不到 vasprun.xml(.gz)")
        sys.exit(1)
        
    execute_plotting(target_xml)