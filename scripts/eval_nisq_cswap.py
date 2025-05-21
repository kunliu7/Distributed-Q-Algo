import argparse
import itertools
import time
import pandas as pd

from dqalgo.data_mgr import NISQCswapDataMgr
from dqalgo.nisq.eval import eval_CSWAP_teledata, eval_CSWAP_telegate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trgts", "-t", type=int, nargs="+", default=[2])
    parser.add_argument("--p2", type=float, nargs="+", default=[0.001])
    args = parser.parse_args()

    dmgr = NISQCswapDataMgr()
    fid_data = []
    for n_trgts, p2 in itertools.product(args.n_trgts, args.p2):
        time_start = time.time()
        p1 = p2 / 10
        pm = p2
        print(f"n_trgts: {n_trgts}, p1: {p1}, p2: {p2}, pm: {pm}")
        mean_fid, std_fid = eval_CSWAP_telegate(n_trgts, p1)
        fid_data.append([n_trgts, p2, mean_fid, std_fid])

    out_df = pd.DataFrame(fid_data, columns=["n_trgts", "p2", "mean_fid", "std_fid"])
    dmgr.save(out_df, n_trgts=args.n_trgts, p2=args.p2)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":
    main()
