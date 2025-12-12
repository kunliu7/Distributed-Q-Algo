import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


def fidelity_model(n, a, b):
    return a + b*n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teledata_path", type=str, default='./data/nisq/cswap/n_trgts=1,2,3,4,5-p2=0.001,0.003,0.005-method=teledata.csv')
    parser.add_argument("--telegate_path", type=str, default='./data/nisq/cswap/n_trgts=1,2,3,4,5-p2=0.001,0.003,0.005-method=telegate.csv')
    parser.add_argument("--save_path", type=str, default='./data/nisq/cswap_graphs/cswap_simulations.pdf')
    args = parser.parse_args()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    paths = [args.teledata_path, args.telegate_path]
    schemes = ['Teledata', 'Telegate']

    for index, (path, ax, scheme) in enumerate(zip(paths, axs, schemes)):
        df = pd.read_csv(path)

        colors = ['#1f77b4', '#2ca02c', '#9467bd']
        # plt.figure(figsize=(6,4))

        for idx, p2_val in enumerate(sorted(df['p2'].unique())):
            subset = df[(df['p2'] == p2_val)]
            n_vals = subset['n_trgts']
            fid_vals = subset['mean_fid']
            std_vals = subset['std_fid']

            # plot the data
            ax.errorbar(
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
                ax.plot(n_fit, fid_fit, linestyle='--', label='Linear Fit', c=colors[idx], alpha=0.6)
            else:
                ax.plot(n_fit, fid_fit, linestyle='--', c=colors[idx], alpha=0.6)


        ax.set_xlabel(r'Target State Size ($n$)', fontsize=18)
        if index == 0:
            ax.set_ylabel('Fidelity (Classical)', fontsize=18)
        else:
            ax.tick_params(labelleft=False)

        # ax.set_title(f'Two-Party CSWAP Fidelity ({scheme})', fontsize=18)
        ax.set_xticks(df['n_trgts'].unique())
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_ylim(0.75, 1)
        ax.legend(fontsize=15)
        ax.grid(alpha=0.3)

    params = {'legend.fontsize': 16,
            #   ‘figure.figsize’: (8, 5),
              'axes.labelsize': 18,  # ‘x-large’,
              'axes.titlesize': 30,  # ‘x-large’,
              'xtick.labelsize': 20,
              'ytick.labelsize': 18,
              'pdf.fonttype': 42,
              'ps.fonttype': 42, }
    plt.rcParams.update(params)
    fig.tight_layout()
    plt.savefig(args.save_path)
    plt.show()


if __name__ == "__main__":
    main()