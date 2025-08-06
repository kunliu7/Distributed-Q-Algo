import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def fidelity_model(n, a, b):
    return a + b*n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default='teledata', choices=['teledata', 'telegate'])
    parser.add_argument("--min_trgts", type=int, default=1)
    parser.add_argument("--max_trgts", type=int, default=999)

    args = parser.parse_args()
    csv_path = f'./data/nisq/cswap/{args.method}/collected.csv'
    save_path = f'./data/nisq/cswap_graphs/{args.method}_simulations.pdf'
    df = pd.read_csv(csv_path)
    df = df[(df['n_trgts'] >= args.min_trgts) & (df['n_trgts'] <= args.max_trgts)]

    colors = ['#1f77b4', '#2ca02c', '#9467bd']
    plt.figure(figsize=(6,4))

    for idx, p2_val in enumerate(sorted(df['p2'].unique())):
        subset = df[(df['p2'] == p2_val)]
        n_vals = subset['n_trgts']
        fid_vals = subset['mean_fid']
        std_vals = subset['std_fid']

        # plot the data
        plt.errorbar(
                x=n_vals, y=fid_vals, yerr=std_vals,
                label='$p_{2q} = ' + str(round(p2_val, 5)) + '$',
                marker='o', capsize=3, linestyle='-', markersize=5, linewidth=1, c=colors[idx]
            )

        # fit the model to the data
        p_opt, _ = curve_fit(fidelity_model, n_vals, fid_vals, p0=[1, 0.05])
        print(p_opt)
        # plot the model
        n_fit = np.linspace(min(n_vals), max(n_vals), 100)
        fid_fit = fidelity_model(n_fit, *p_opt)
        # only add fit label the first time
        if idx == 0:
            plt.plot(n_fit, fid_fit, linestyle='--', label='Linear Fit', c=colors[idx], alpha=0.6)
        else:
            plt.plot(n_fit, fid_fit, linestyle='--', c=colors[idx], alpha=0.6)


    plt.xlabel(r'Target State Size ($n$)', fontsize=12)
    plt.ylabel('Fidelity (Classical)', fontsize=12)
    plt.title(f'Mean Fidelity of Two-Party CSWAP ({args.method.title()} Scheme)', fontsize=13)
    plt.xticks(df['n_trgts'].unique())
    plt.ylim(0.75, 1)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()