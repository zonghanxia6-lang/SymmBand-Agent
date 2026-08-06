#!/usr/bin/env bash
#SBATCH --job-name=symmcd-band
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=symmcd-band-%j.out
#SBATCH --error=symmcd-band-%j.err

set -euo pipefail

# Adapt module names to the cluster. VASP and IRVSP must be available in this job.
module purge
module load cuda/12.1
module load vasp
module load irvsp

source /ABSOLUTE/PATH/miniconda3/etc/profile.d/conda.sh
conda activate symmcd-band-agent
source /ABSOLUTE/PATH/SymmBand-Agent/cluster_environment/config/runtime.env

cd "${SYMMCD_ROOT}"
python cluster_environment/validate_environment.py
python structure_agent.py \
  --prompt "我要生成10个194号空间群的NaBi结构，然后计算它的能带"
