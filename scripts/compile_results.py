import argparse
import os
import re

import numpy as np
import pandas as pd

"""
Files outputted by HPC jobs will be a single-line .txt file with the fidelity.
This script computes the mean and standard deviation of all fidelities in a given folder
and prints them.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    data_dir = args.data_dir
    # Pattern with capture groups for method, n_trgts, and p2
    pattern = re.compile(r'^([a-zA-Z]+)-t=(\d+)-p=([\d\.]+)$')

    # Dictionary to collect all fids for each (method, n_trgts, p2)
    results = {}

    # List all subdirectories matching the pattern
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        match = pattern.match(folder)
        if os.path.isdir(folder_path) and match:
            method, n_trgts, p2 = match.groups()
            n_trgts = int(n_trgts)
            p2 = float(p2)
            key = (method, n_trgts, p2)
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path) as f:
                        try:
                            value = float(f.read().strip())
                            results.setdefault(key, []).append(value)
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")

    # Prepare data for DataFrame
    rows = []
    for (method, n_trgts, p2), fids in results.items():
        mean_fid = np.mean(fids)
        std_fid = np.std(fids)
        rows.append({
            'method': method,
            'n_trgts': n_trgts,
            'p2': p2,
            'mean_fid': mean_fid,
            'std_fid': std_fid,
            'n_samples': len(fids)
        })

    if not rows:
        print("No fidelity values found.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(f'{data_dir}/collected.csv', index=False)
    print(df)

if __name__ == "__main__":
    main()