import argparse


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-n", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    args.add_argument("-p", type=float, nargs="+", default=[0.001, 0.003, 0.005])
    args.add_argument("-m", type=str, nargs="+", default=["teledata", "telegate"])
    args.add_argument("-o", type=str, default="0724")
    args.add_argument("-c", type=int, default=4)
    args.add_argument("-M", "--mem-per-cpu", type=int, default=8)
    args.add_argument("-t", "--time", type=str, default="4:00:00")
    args = args.parse_args()

    n_trgts_lst = args.n
    p2s = args.p
    methods = args.m
    for n_trgts in n_trgts_lst:
        for p2 in p2s:
            for method in methods:
                cmd = f"sbatch -c {args.c} --mem-per-cpu={args.mem_per_cpu}G -t {args.time} bashes/yale-{method}-{n_trgts}-{p2}.sh"
                print(cmd)


if __name__ == "__main__":
    main()