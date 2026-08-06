# irrep_plotter.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymatgen.electronic_structure.plotter import BSPlotter
import logging

logger = logging.getLogger("workflow.plotter")

def get_distance_for_k_global(data, bs, k_global):
    """
    安全地将全局 K 点索引映射为画图时的 X 坐标 (Distance)。
    """
    for b_idx, branch in enumerate(bs.branches):
        s_idx = branch['start_index']
        e_idx = branch['end_index']
        if s_idx <= k_global <= e_idx:
            k_local = k_global - s_idx
            if b_idx < len(data['distances']) and k_local < len(data['distances'][b_idx]):
                return data['distances'][b_idx][k_local]
    return 0.0

def plot_irrep_crossings(bs, crossings, output_filename, material_info, y_lim=[-1.0, 1.0]):
    """
    纯净版：只接收数据（BandStructure和交叉点列表），专注画图。
    """
    logger.info("Plotting band structure and irrep crossings...")
    
    if not bs:
        logger.error("Plotter Error: BandStructure 对象为空。")
        return

    plotter = BSPlotter(bs)
    
    # 获取绘图数据
    if callable(plotter.bs_plot_data):
        data = plotter.bs_plot_data(zero_to_efermi=True)
    else:
        data = plotter.bs_plot_data

    # get_plot 会把 Fermi 能级设置在 y=0 处
    plotter.get_plot(ylim=y_lim, zero_to_efermi=True)
    fig = plt.gcf()
    ax = plt.gca()
    fig.set_size_inches(16.0, 8.0)
    
    # 添加费米能级
    ax.axhline(
        0, 
        color='gray',          # 使用不显眼的灰色
        linestyle='--',       # 虚线
        linewidth=1.0,        # 细一点
        alpha=0.8,           # 稍微透明
        zorder=1,            # 放在背景层，能带之下
        label="$E_F$"
    )

    standard_ticks = [-1.0, -0.5, 0.0, 0.5, 1.0]
    ax.set_yticks(standard_ticks)
    padding = 0.05
    ax.set_ylim(min(standard_ticks) - padding, max(standard_ticks) + padding)

    # ==========================================
    
    # 叠加不可约表示反转的交叉点圈圈
    for i, cross in enumerate(crossings):
        k1_global = cross['k_interval'][0] - 1
        k2_global = cross['k_interval'][1] - 1
        
        x1 = get_distance_for_k_global(data, bs, k1_global)
        x2 = get_distance_for_k_global(data, bs, k2_global)
        x_coord = (x1 + x2) / 2.0
        
        y_coord = cross['energy_approx']

        if y_lim[0] <= y_coord <= y_lim[1]:
            ax.scatter(
                x_coord, y_coord,
                color='red', marker='o', s=120,
                facecolors='none', edgecolors='red', linewidth=2,
                zorder=10, label="Crossing" if i == 0 else ""
            )
            
            irrep1, irrep2 = cross['irreps_swapped']
            ax.annotate(
                f"{irrep1} <-> {irrep2}",
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                color='red',
                fontsize=8,
                zorder=15
            )

    # 将 E_F 图例也加进去
    ax.legend(loc='upper right', fontsize=10)
    
    ax.set_title(f"{material_info} Band Structure")
    fig.tight_layout()
    
    try:
        fig.savefig(output_filename, dpi=300)
        logger.info(f"Plot successfully saved to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")
    finally:
        plt.close(fig)
