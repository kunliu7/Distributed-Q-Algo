import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

k_vals = [8,12]
min_n = 1
max_n = 10
colors = {0.001: '#1f77b4', 0.003: '#2ca02c', 0.005: '#9467bd'}
line_styles = {8: '--', 12: ':'}

def fidelity_model(n, a, b):
    return a + b*n

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teledata_path", type=str, default='./data/nisq/cswap/n_trgts=1,2,3,4,5-p2=0.001,0.003,0.005-method=teledata.csv')
    parser.add_argument("--telegate_path", type=str, default='./data/nisq/cswap/n_trgts=1,2,3,4,5-p2=0.001,0.003,0.005-method=telegate.csv')
    parser.add_argument("--ghz_path", type=str, default='./data/nisq/ghz/n_parties=4,6,8,10,12-p2=0.001,0.003,0.005.csv')
    parser.add_argument("--save_path", type=str, default='./data/nisq/overall_error_graphs/overall_simulations.pdf')
    args = parser.parse_args()

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    paths = [args.teledata_path, args.telegate_path]
    schemes = ['Teledata', 'Telegate']
    ghz_df = pd.read_csv(args.ghz_path)

    for index, (cswap_path, ax, scheme) in enumerate(zip(paths, axs, schemes)):
        cswap_df = pd.read_csv(cswap_path)

        for p2_val in sorted(cswap_df['p2'].unique()):
            for k in k_vals:
                subset = cswap_df[cswap_df['p2'] == p2_val]
                n_vals = subset['n_trgts']
                fid_vals = subset['mean_fid']

                ghz_fid = ghz_df[(ghz_df['n_parties'] == k//2) & (ghz_df['p2'] == p2_val)].iloc[0].fid

                # fit the model to the data
                p_opt, _ = curve_fit(fidelity_model, n_vals, fid_vals, p0=[1, 0.05])
                print(p_opt)
                # plot the model
                n_fit = np.linspace(min_n, max_n, 1000)
                cswap_fid = fidelity_model(n_fit, *p_opt)

                final_fid = ghz_fid * (cswap_fid ** (k - 1))

                if line_styles[k] == '--':
                    ax.plot(n_fit, final_fid, color=colors[p2_val], linestyle=line_styles[k], label=f'$p_{{2q}} = {p2_val}$')
                else:
                    ax.plot(n_fit, final_fid, color=colors[p2_val], linestyle=line_styles[k])


        ax.set_xlabel(r'Target State Size ($n$)', fontsize=18)
        if index == 0:
            ax.set_ylabel('Fidelity (Estimate)', fontsize=18)
        else:
            ax.tick_params(labelleft=False)
        # ax.set_title(f'Overall QRACD Fidelity ({scheme})', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=15)
        ax.grid(alpha=0.3)

    params = {'legend.fontsize': 16,
            #   ‘figure.figsize’: (8, 5),
              'axes.labelsize': 18,  # ‘x-large’,
              'axes.titlesize': 20,  # ‘x-large’,
              'xtick.labelsize': 14,
              'ytick.labelsize': 18,
              'pdf.fonttype': 42,
              'ps.fonttype': 42, }
    plt.rcParams.update(params)

    fig.tight_layout()
    plt.savefig(args.save_path)
    plt.show()


if __name__ == "__main__":
    main()