import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file_prefix", type=str)
    n_files = 150

    args = parser.parse_args()
    fids = []
    for idx in range(n_files):
        filename = f'{args.output_file_prefix}_{idx}.txt'
        with open(filename) as file:
            fids.append(int(file.read()))

    mean = np.mean(fids)
    std = np.std(fids)
    print(mean, std)

if __name__ == "__main__":
    main()