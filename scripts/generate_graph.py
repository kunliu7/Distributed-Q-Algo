import argparse
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def fidelity_model(n, a, b, c):
    return a + b*n + c*n*n #a * np.exp(-b * n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='../data/nisq/cswap/n_trgts=1,2,3,4,5,6-p2=0.001,0.003,0.005-method=teledata.csv')
    parser.add_argument("--save_path", type=str, default='../data/nisq/cswap_graphs/teledata_simulations.pdf')

    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)

    colors = ['#1f77b4', '#2ca02c', '#9467bd']
    plt.figure(figsize=(6,4))

    for idx, p2_val in enumerate(sorted(df['p2'].unique())):
        subset = df[df['p2'] == p2_val]
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
        p_opt, _ = curve_fit(fidelity_model, n_vals, fid_vals, p0=[1, 0.05, 0.05])
        print(p_opt)
        # plot the model
        n_fit = np.linspace(min(n_vals), max(n_vals), 100)
        fid_fit = fidelity_model(n_fit, *p_opt)
        print(p_opt)
        # only add fit label the first time
        if idx == 0:
            plt.plot(n_fit, fid_fit, linestyle='--', label='Poly Fit (degree 2)', c=colors[idx], alpha=0.6)
        else:
            plt.plot(n_fit, fid_fit, linestyle='--', c=colors[idx], alpha=0.6)


    plt.xlabel(r'Number of Targets ($n$)', fontsize=12)
    plt.ylabel('Fidelity', fontsize=12)
    plt.title('Mean Fidelity of Two-Party CSWAP (Teledata Scheme)', fontsize=13)
    plt.ylim(0.85, 1.01)
    plt.xticks(df['n_trgts'].unique())
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)

    plt.savefig(args.save_path)
    plt.show()


if __name__ == "__main__":
    main()