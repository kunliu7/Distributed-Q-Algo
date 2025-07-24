#!/bin/bash
#SBATCH --job-name=eval_nisq_cswap
#SBATCH --output=logs/eval_nisq_cswap_%j.out
#SBATCH --error=logs/eval_nisq_cswap_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-149
#SBATCH --mail-type=ALL


module load miniconda
conda activate dqalgo

n_trgts=$(echo "$1" | tr -d '[:space:]')
p2=$(echo "$2" | tr -d '[:space:]')
method=$(echo "$3" | tr -d '[:space:]')

echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "Extra arguments:" "$@"

python scripts/eval_nisq_cswap_parallel.py --n_trgts "$n_trgts" --p2 "$p2" --method "$method" --slurm_index ${SLURM_ARRAY_TASK_ID}