def main():
    n_trgts_lst = range(1, 7)
    p2s = [0.001, 0.003, 0.005]
    methods = ["teledata", "telegate"]
    for n_trgts in n_trgts_lst:
        for p2 in p2s:
            for method in methods:
                cmd = f"python ./scripts/eval_nisq_cswap_parallel.py --n_trgts {n_trgts} --p2 {p2} --method {method} --output_file_prefix './data/nisq/cswap/example_outputs_folder' --slurm_index $SLURM_ARRAY_TASK_ID"
                print(cmd)


if __name__ == "__main__":
    main()