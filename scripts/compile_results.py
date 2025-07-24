import argparse
import numpy as np
import os

"""
Files outputted by HPC jobs will be a single-line .txt file with the fidelity.
This script computes the mean and standard deviation of all fidelities in a given folder
and prints them.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--n_files", type=int, default=150)

    args = parser.parse_args()
    n_files = args.n_files
    output_folder = args.output_folder

    fids = []
    filenames = os.listdir(output_folder)
    filenames = [f for f in filenames if '.txt' in f]
    if len(filenames) < n_files:
        print(f'Only has {len(filenames)} < {n_files} files')
        return

    for filename in filenames:
        path = f'{output_folder}/{filename}'
        with open(path) as file:
            fids.append(float(file.read()))

    mean = np.mean(fids)
    std = np.std(fids)
    print(f"Mean fidelity:\n{mean}\n Standard deviation:\n{std}")

if __name__ == "__main__":
    main()