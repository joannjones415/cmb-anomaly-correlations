#!/bin/bash
#SBATCH --job-name=cmb-anomaly-sims
#SBATCH --array=0-99
#SBATCH --time=08:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=cluster/logs/sims_%a.out

# submit from the repo root with
#   mkdir -p cluster/logs && sbatch cluster/run_sims.sbatch
# 100 tasks of 10^6 realizations give 10^8 in total, about 4 hours per task.
# the array index is the seed, so no two tasks share realizations

set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate cmb-anomalies
# namaster is openmp, keep it to the one cpu requested
export OMP_NUM_THREADS=1

python -u simulate.py --nsim 1000000 --seed "$SLURM_ARRAY_TASK_ID"
