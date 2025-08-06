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

    args = parser.parse_args()
    cswap_path = f'./data/nisq/cswap/{args.method}/collected.csv'
    ghz_path = f'./data/nisq/ghz/collected.csv'
    save_path = f'./data/nisq/overall_error_graphs/overall_{args.method}_simulations.pdf'
    cswap_df = pd.read_csv(cswap_path)
    ghz_df = pd.read_csv(ghz_path)

    colors = {0.001: '#1f77b4', 0.003: '#2ca02c', 0.005: '#9467bd'}
    line_styles = {8: '--', 12: ':'}
    plt.figure(figsize=(6,4))

    k_vals = [8,12]
    min_n = 1
    max_n = 10

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
                plt.plot(n_fit, final_fid, color=colors[p2_val], linestyle=line_styles[k], label=f'$p_{{2q}} = {p2_val}$')
            else:
                plt.plot(n_fit, final_fid, color=colors[p2_val], linestyle=line_styles[k])


    plt.xlabel(r'Target State Size ($n$)', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title(f'Overall Fidelity Estimate of QRACD ({args.method.title()} Scheme)', fontsize=13)
    plt.xticks()
    plt.ylim(0, 1)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    main()