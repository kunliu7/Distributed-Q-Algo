
import argparse


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-n", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args.add_argument("-p", type=float, nargs="+", default=[0.001, 0.003, 0.005])
    args.add_argument("-m", type=str, nargs="+", default=["teledata", "telegate"])
    args.add_argument("-o", type=str, default="0724")
    args.add_argument("-c", type=int, default=4)
    args = args.parse_args()
    HEADER = """#!/bin/bash
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
"""
    n_trgts_lst = args.n
    p2s = args.p
    methods = args.m
    for n_trgts in n_trgts_lst:
        for p2 in p2s:
            for method in methods:
                temp = f"python scripts/eval_nisq_cswap_parallel.py --n_trgts {n_trgts} --p2 {p2} --method {method} --slurm_index ${{SLURM_ARRAY_TASK_ID}}"
                file = HEADER + temp + "\n"
                with open(f"bashes/yale-{method}-{n_trgts}-{p2}.sh", "w") as f:
                    f.write(file)


if __name__ == "__main__":
    main()