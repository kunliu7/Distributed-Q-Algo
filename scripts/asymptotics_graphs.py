import json
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="darkgrid")

def k_bound(eps, p, n):
    return np.log(1 - eps)/((2 + 6*n)*np.log(1 - 3/4*p))

p = np.linspace(1, 8, num=5000)
p = np.power(10, -p)
n = 100
epsilon_values = [-1, -2, -3, -4]
x_lims = (1e-8, 1e-3)
y_lims = (1, n)

colors = sns.color_palette()
markers = ['*', 'o', 's', 'X', '^']
codes = ['SC', 'LP']

# plot theoretical bounds for different epsilon values
for eps in epsilon_values:
    k = k_bound(10**eps, p, n)
    plt.plot(p, k, linestyle='--', label="$\epsilon = 10^{" + str(eps) + "}$", alpha=0.5)
    plt.fill_between(p, k, alpha=0.2)

# get entanglement distillation data
with open('../data/entanglement-dist/ataides_2025.json', 'r') as f:
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


# Labels and title
plt.xlabel("Bell pair logical error rate")
plt.ylabel("Number of QPUs")
plt.xscale('log')
plt.ylim(y_lims)
plt.xlim(x_lims)
plt.title(f"Upper bounds on $k$ for fixed values of $\epsilon$, $n = {n}$")
plt.legend()

# Save plot
plt.savefig("../data/asymptotics-graphs/k_bound_plot.pdf")
plt.show()