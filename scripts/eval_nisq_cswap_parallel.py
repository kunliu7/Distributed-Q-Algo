import argparse
import time

from dqalgo.nisq.eval_cswap import (eval_CSWAP_teledata_single_thread,
                                    eval_CSWAP_telegate_single_thread)

"""
A single job to be executed by HPC
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trgts", "-t", type=int, default=4)
    parser.add_argument("--p2", type=float, default=0.001)
    parser.add_argument("--n_shots", type=int, default=1024)
    parser.add_argument("--iters_per_input", type=int, default=1)
    parser.add_argument("--method", type=str, default="telegate", choices=["teledata", "telegate"])
    parser.add_argument("--slurm_index", type=int)
    parser.add_argument("--output_file_prefix", type=str)

    args = parser.parse_args()
    print(args)
    if args.method == "teledata":
        print("Using teledata method")
        eval_func = eval_CSWAP_teledata_single_thread
    else:
        eval_func = eval_CSWAP_telegate_single_thread

    time_start = time.time()
    fid = eval_func(
        args.n_trgts,
        p2=args.p2,
        circs_per_input=args.iters_per_input,
        shots_per_circ=args.n_shots)
    time_end = time.time()

    with open(f'{args.output_file_prefix}_{args.slurm_index}.txt', 'w') as f_out:
        f_out.write(str(fid))

    print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":
    main()
