import argparse
import itertools
import time

from dqalgo.data_mgr import NISQFanoutDataMgr
from dqalgo.nisq.eval import eval_Baumer_Fanout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trgts", "-t", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--p2", type=float, nargs="+", default=[0.001, 0.003, 0.005])
    parser.add_argument("--n_shots", "-s", type=int, default=100000)
    args = parser.parse_args()

    dmgr = NISQFanoutDataMgr()
    error_counts_lst = []
    for n_trgts, p2 in itertools.product(args.n_trgts, args.p2):
        time_start = time.time()
        p1 = p2 / 10
        pm = p2
        print(f"n_trgts: {n_trgts}, p1: {p1}, p2: {p2}, pm: {pm}")
        error_counts = eval_Baumer_Fanout(n_trgts, p1, p2, pm, args.n_shots)
        # print(error_counts)
        error_counts_lst.append(error_counts)
        dmgr.save((error_counts_lst, args.n_trgts, args.p2),
                  n_trgts=args.n_trgts, p2=args.p2, n_shots=args.n_shots)
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":
    main()
