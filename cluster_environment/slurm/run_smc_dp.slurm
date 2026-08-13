#!/usr/bin/env bash
#SBATCH --job-name=symmcd-dp-smc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=smc-dp-%A_%a.out
#SBATCH --error=smc-dp-%A_%a.err

set -euo pipefail

module purge
module load cuda/12.1

source /ABSOLUTE/PATH/miniconda3/etc/profile.d/conda.sh
conda activate symmcd-band-agent
cd /ABSOLUTE/PATH/SymmBand-Agent

# Five paired seeds: array tasks 0-4 are alpha=0, tasks 5-9 are alpha=3.
SEEDS=(20260813 20260814 20260815 20260816 20260817)
FORMULA="BiTe"
if (( SLURM_ARRAY_TASK_ID < 5 )); then
  ALPHA=0
  SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
else
  ALPHA=3
  SEED="${SEEDS[$((SLURM_ARRAY_TASK_ID - 5))]}"
fi

OUT="research_data/smc_runs/dp_sg194/${FORMULA}/alpha${ALPHA}/seed${SEED}"
python inverse_design_cli.py smc-generate \
  --formula "${FORMULA}" \
  --spacegroup 194 \
  --model research_models/particles/particle_dp_surrogate.joblib \
  --output-dir "${OUT}" \
  --particles 64 \
  --diffusion-steps 500 \
  --resample-interval 25 \
  --guidance-start-fraction 0.5 \
  --alpha "${ALPHA}" \
  --ess-threshold 0.8 \
  --seed "${SEED}" \
  --device cuda
