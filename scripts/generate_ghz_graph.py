import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def fidelity_model(n, a, b):
    return a + b*n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data/nisq/ghz/collected.csv')
    parser.add_argument("--min_parties", type=int, default=1)
    parser.add_argument("--max_parties", type=int, default=999)

    args = parser.parse_args()
    csv_path = args.data_dir
    save_path = f'./data/nisq/ghz_graphs/ghz_simulations.pdf'
    df = pd.read_csv(csv_path)
    df = df[(df['n_parties'] >= args.min_parties) & (df['n_parties'] <= args.max_parties)]

    colors = ['#1f77b4', '#2ca02c', '#9467bd']
    plt.figure(figsize=(6,4))

    for idx, p2_val in enumerate(sorted(df['p2'].unique())):
        subset = df[(df['p2'] == p2_val)]
        n_vals = subset['n_parties']
        fid_vals = subset['fid']



        # fit the model to the data
        p_opt, _ = curve_fit(fidelity_model, n_vals, fid_vals, p0=[1, 0.05])
        print(p_opt)
        # plot the model
        n_fit = np.linspace(min(n_vals), max(n_vals), 100)
        fid_fit = fidelity_model(n_fit, *p_opt)
        print(p_opt)
        # only add fit label the first time
        if idx == 0:
            plt.plot(n_fit, fid_fit, linestyle='--', label='Linear Fit', c=colors[idx], alpha=0.6)
        else:
            plt.plot(n_fit, fid_fit, linestyle='--', c=colors[idx], alpha=0.6)

        plt.plot(
                n_vals, fid_vals,
                label='$p_{2q} = ' + str(round(p2_val, 5)) + '$',
                marker='o', linestyle='-', markersize=5, linewidth=1, c=colors[idx]
            )


    plt.xlabel(r'Number of Parties ($k$)', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title(f'Fidelity of GHZ State', fontsize=13)
    plt.xticks(df['n_parties'].unique())
    plt.ylim(0.6, 1)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()