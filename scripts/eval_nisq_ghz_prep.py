import argparse
import itertools
import time

import pandas as pd

from dqalgo.data_mgr import NISQPrepareGHZDataMgr
from dqalgo.nisq.eval import eval_GHZ_prep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_parties", "-t", type=int, nargs="+", default=[4])
    parser.add_argument("--p2", type=float, nargs="+", default=[0.001])

    args = parser.parse_args()

    dmgr = NISQPrepareGHZDataMgr()
    fid_data = []
    time_start = time.time()
    for n_parties, p2 in itertools.product(args.n_parties, args.p2):
        p1 = p2 / 10
        pm = p2
        print(f"n_parties: {n_parties}, p1: {p1}, p2: {p2}, pm: {pm}")
        fid = eval_GHZ_prep(n_parties, p1)
        print(fid)
        fid_data.append([n_parties, p2, fid])

    out_df = pd.DataFrame(fid_data, columns=["n_parties", "p2", "fid"])
    dmgr.save(out_df, n_parties=args.n_parties, p2=args.p2)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":
    main()
