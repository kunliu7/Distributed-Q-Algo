#!/bin/bash
#SBATCH --job-name=eval_nisq_cswap
#SBATCH --output=logs/eval_nisq_cswap_%j.out
#SBATCH --error=logs/eval_nisq_cswap_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-149
#SBATCH --mem=8G
#SBATCH --mail-type=ALL


module load miniconda
conda activate dqalgo

python scripts/eval_nisq_cswap_parallel.py "$@"