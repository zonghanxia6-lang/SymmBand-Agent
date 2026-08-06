from pymatgen.io.vasp import Vasprun
import logging

# 引入刚拆分出去的独立解析函数
from irvsp_parser import parse_outir

logger = logging.getLogger("workflow.analyzer")

class DegeneracyAnalyzer:
    def __init__(self, xml_file):
        """
        初始化分析器：依赖 vasprun.xml 实例化核心状态 (self.bs)
        """
        logger.info(f"Parsing VASP output: {xml_file}")
        try:
            self.run = Vasprun(xml_file)
            self.bs = self.run.get_band_structure(line_mode=True)
            self.valid = True
        except Exception as e:
            logger.error(f"Error parsing BS: {e}")
            self.bs = None
            self.valid = False

    def find_crossings_by_irreps(self, outir_path="outir", e_window=1.0, gap_tol=0.03):
        """
        利用稳健的 Gap 寻找局部极小值(谷底)，然后提取谷底两端的表示进行对换校验。
        """
        if not self.valid or not self.bs:
            return []

        # 调用外部的纯文本解析函数获取数据
        outir_data = parse_outir(outir_path)
        if not outir_data:
            return []

        crossings = []
        efermi = self.bs.efermi
        logger.info("Scanning outir with Hybrid Gap-Irrep Mode...")
        
        for branch in self.bs.branches:
            s_idx = branch['start_index']
            e_idx = branch['end_index']
            
            if e_idx - s_idx < 2:
                continue
            
            for k_local in range(s_idx + 1, e_idx):
                knum_k = k_local + 1     
                knum_prev = knum_k - 1   
                knum_next = knum_k + 1   
                
                if knum_prev not in outir_data or knum_k not in outir_data or knum_next not in outir_data:
                    continue
                    
                b_prev = {b['bnd']: b for b in outir_data[knum_prev]['bands']}
                b_k    = {b['bnd']: b for b in outir_data[knum_k]['bands']}
                b_next = {b['bnd']: b for b in outir_data[knum_next]['bands']}
                
                common_bnds = sorted(list(set(b_prev.keys()) & set(b_k.keys()) & set(b_next.keys())))
                
                valid_bnds = []
                for bnd in common_bnds:
                    if -e_window <= (b_k[bnd]['eigval'] - efermi) <= e_window:
                        valid_bnds.append(bnd)
                        
                for i in range(len(valid_bnds) - 1):
                    bnd_L = valid_bnds[i]
                    bnd_U = valid_bnds[i+1]
                    
                    if bnd_U != bnd_L + 1:
                        continue
                        
                    gap_prev = abs(b_prev[bnd_U]['eigval'] - b_prev[bnd_L]['eigval'])
                    gap_k    = abs(b_k[bnd_U]['eigval'] - b_k[bnd_L]['eigval'])
                    gap_next = abs(b_next[bnd_U]['eigval'] - b_next[bnd_L]['eigval'])
                    
                    if gap_k <= gap_prev and gap_k < gap_next:
                        if gap_k < gap_tol:
                            rep_prev_L = b_prev[bnd_L]['irrep']
                            rep_prev_U = b_prev[bnd_U]['irrep']
                            rep_next_L = b_next[bnd_L]['irrep']
                            rep_next_U = b_next[bnd_U]['irrep']
                            
                            if (rep_prev_L == rep_next_U) and (rep_prev_U == rep_next_L) and (rep_prev_L != rep_prev_U):
                                e_approx = (b_k[bnd_L]['eigval'] + b_k[bnd_U]['eigval']) / 2.0 - efermi
                                
                                crossings.append({
                                    'k_interval': (knum_prev, knum_next),
                                    'k1_coords': outir_data[knum_prev]['k_coord'],
                                    'k2_coords': outir_data[knum_next]['k_coord'],
                                    'band_indices': (bnd_L, bnd_U),
                                    'irreps_swapped': (rep_prev_L, rep_prev_U),
                                    'energy_approx': e_approx
                                })
                                
                                logger.info(
                                    f"HYBRID CROSSING | k-points: {knum_prev} -> {knum_next} (min at {knum_k}) | "
                                    f"Bands: {bnd_L}-{bnd_U} | "
                                    f"Irreps Swap: {rep_prev_L} <-> {rep_prev_U} | "
                                    f"E: {e_approx:.3f} eV | Gap_min: {gap_k:.4f} eV"
                                )

        return crossings