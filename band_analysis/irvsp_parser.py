import re
import logging

logger = logging.getLogger("workflow.parser")

def parse_outir(outir_path="outir"):
    """
    独立出来的纯函数：解析 irvsp 的 outir 文件 (在解析阶段直接根据 ndg 展开简并带)
    """
    logger.info(f"Parsing irvsp output: {outir_path}")
    parsed_data = {}
    current_knum = None
    in_band_table = False
    
    try:
        with open(outir_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith("knum"):
                    match = re.search(r'knum\s*=\s*(\d+).*?kname=\s*(\S*)', line)
                    if match:
                        current_knum = int(match.group(1))
                        kname = match.group(2)
                        parsed_data[current_knum] = {
                            'kname': kname,
                            'k_coord': [],
                            'bands': []
                        }
                    in_band_table = False
                    continue
                    
                if current_knum is not None:
                    if line.startswith("k ="):
                        matches = re.findall(r'-?\d+\.\d+', line)
                        if len(matches) >= 3:
                            coords = [float(x) for x in matches[:3]]
                            parsed_data[current_knum]['k_coord'] = coords
                        continue
                        
                    if line.startswith("bnd") and "eigval" in line:
                        in_band_table = True
                        continue
                        
                    if in_band_table and (not line or line.startswith("***")):
                        in_band_table = False
                        continue
                        
                    if in_band_table and "=" in line:
                        left_part, irrep_part = line.split("=")
                        irrep = irrep_part.strip()
                        
                        num_match = re.match(r'^\s*(\d+)\s+(\d+)\s*(-?\d+\.\d+)', left_part)
                        if num_match:
                            bnd_start = int(num_match.group(1))
                            ndg = int(num_match.group(2))
                            eigval = float(num_match.group(3))
                            
                            for step in range(ndg):
                                parsed_data[current_knum]['bands'].append({
                                    'bnd': bnd_start + step, 
                                    'eigval': eigval,
                                    'irrep': irrep
                                })
                        else:
                            logger.warning(f"跳过无法解析的行: {line}")
                            
    except Exception as e:
        logger.error(f"Failed to parse {outir_path}: {e}")
        
    return parsed_data