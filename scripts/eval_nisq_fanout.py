import argparse
import itertools
import time

from dqalgo.data_mgr import NISQFanoutDataMgr
from dqalgo.nisq.eval import get_truth_table_tomography_for_Fanout
from dqalgo.nisq.utils import get_depolarizing_noise_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trgts", "-t", type=int, nargs="+", default=[3])
    parser.add_argument("--p_2q", "-p2", type=float, nargs="+", default=[0.0])
    parser.add_argument("--n_shots", "-s", type=int, default=2048)
    args = parser.parse_args()

    time_start = time.time()

    input_to_fid_lst = []
    for n_trgts, p_2q in itertools.product(args.n_trgts, args.p_2q):
        print(f"============ Testing {n_trgts} targets with p_2q={p_2q} =============")
        p_1q = p_2q / 10
        p_meas = p_2q
        noise_model = get_depolarizing_noise_model(p_1q=p_1q, p_2q=p_2q, p_meas=p_meas)

        input_to_fid = get_truth_table_tomography_for_Fanout(
            n_trgts, noise_model, args.n_shots,
        )

        input_to_fid_lst.append(input_to_fid)

    dmgr = NISQFanoutDataMgr()
    dmgr.save((input_to_fid_lst, args.n_trgts, args.p_2q),
              n_trgts=args.n_trgts, p_2q=args.p_2q)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")


if __name__ == "__main__":
    main()
