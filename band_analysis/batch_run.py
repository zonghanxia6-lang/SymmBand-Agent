import os
import glob
import logging
from pymatgen.core import Structure
from jobflow import run_locally, SETTINGS

from logger_setup import setup_logger
logger = setup_logger("workflow", log_file="workflow.log", level=logging.INFO)

from config import INPUT_DIR, OUTPUT_BASE_DIR, OUTPUT_BAND_DIR
from workflow_builder import build_degeneracy_flow

def main():
    if not os.path.exists(INPUT_DIR):
        logger.critical(f"Input directory {INPUT_DIR} does not exist!")
        return
    
    output_band_dir = os.path.abspath(OUTPUT_BAND_DIR)
    os.makedirs(output_band_dir, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_DIR, "*.cif")) + glob.glob(os.path.join(INPUT_DIR, "POSCAR*"))
    logger.info(f"Found {len(files)} structures...\n")

    for idx, fpath in enumerate(files):
        filename = os.path.basename(fpath)
        task_name = filename.replace(".cif", "").replace("POSCAR_", "").replace("POSCAR", "struct")
        
        logger.info(f"[{idx+1}/{len(files)}] Building workflow: {task_name} ...")
        task_dir = os.path.join(OUTPUT_BASE_DIR, task_name)
        abs_task_dir = os.path.abspath(task_dir)

        try:
            struct = Structure.from_file(fpath)
            
            # 👇 核心组装逻辑被封装成了一行代码
            full_flow = build_degeneracy_flow(struct, task_name, output_band_dir)

            # 运行环境配置
            os.makedirs(abs_task_dir, exist_ok=True)
            docs_dir = os.path.join(abs_task_dir, "docs")
            data_dir = os.path.join(abs_task_dir, "data")
            os.makedirs(docs_dir, exist_ok=True)
            os.makedirs(data_dir, exist_ok=True)
            
            SETTINGS.JOB_STORE.docs_store.paths = [os.path.join(docs_dir, 'doc.json')]
            SETTINGS.JOB_STORE.additional_stores['data'].paths = [os.path.join(data_dir, 'data.json')]

            # 执行
            logger.info(f"  Starting execution in {abs_task_dir}...")
            run_locally(full_flow, create_folders=True, root_dir=abs_task_dir)
            logger.info(f"  Done: {task_name}\n")

        except Exception as e:
            logger.exception(f"[{task_name}] Workflow failed with error:") 
            continue

if __name__ == "__main__":
    main()
