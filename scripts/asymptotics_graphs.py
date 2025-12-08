import argparse
import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def k_bound(eps, p, n):
    return np.log(1 - eps)/((2 + 6*n)*np.log(1 - 3/4*p))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default=f'./data/asymptotics-graphs/k_bound_plot.pdf')

    args = parser.parse_args()

    save_dir = args.save_path
    sns.set_theme(style="darkgrid")
    plt.figure(figsize=(8, 4))

    p = np.linspace(1, 8, num=5000)
    p = np.power(10, -p)
    n = 100
    epsilon_values = [-1, -2, -3, -4]
    x_lims = (1e-8, 1e-3)
    y_lims = (1, n)

    colors = sns.color_palette()
    markers = ['*', 'o', 's', 'X', '^']

    # plot theoretical bounds for different epsilon values
    for eps in epsilon_values:
        k = k_bound(10**eps, p, n)
        plt.plot(p, k, linestyle='--', label="$\epsilon = 10^{" + str(eps) + "}$", alpha=0.5)
        plt.fill_between(p, k, alpha=0.2)

    # get entanglement distillation data
    with open('./data/entanglement-dist/ataides_2025.json', 'r') as f:
        entanglement_dist_data = json.load(f)

    # plot data from entanglement distillation
    marker_idx = 0
    for data in entanglement_dist_data:
        p_val = data['LER'][0]/data['k']
        epsilons = np.linspace(min(epsilon_values), max(epsilon_values), num=100)
        code_vals = [k_bound(10**(eps), p_val, n) for eps in epsilons]
        plt.plot([p_val]*len(epsilons), code_vals, color='black', linestyle=':', alpha=0.4)

        hit = False
        # only plot points on the graph if they are within the limits
        for eps_idx, eps in enumerate(epsilon_values):
            code_k_bound = k_bound(10**eps, p_val, n)
            if x_lims[0] < p_val < x_lims[1] and y_lims[0] < code_k_bound:
                if not hit:
                    plt.scatter([p_val], [code_k_bound], color=colors[eps_idx], marker=markers[marker_idx],
                                label=f"{data['code']} [[{data['n']}, {data['k']}, {data['d']}]]")
                else:
                    plt.scatter([p_val], [code_k_bound], color=colors[eps_idx], marker=markers[marker_idx])

                hit = True

        if hit:
            marker_idx += 1


    plt.xlabel("Bell pair logical error rate")
    plt.ylabel("Number of QPUs")
    plt.xscale('log')
    plt.ylim(y_lims)
    plt.xlim(x_lims)
    plt.title(f"Upper bounds on $k$ for fixed values of $\epsilon$, $n = {n}$")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(save_dir)
    plt.show()

if __name__ == "__main__":
    main()