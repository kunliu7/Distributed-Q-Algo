from typing import Sequence

from matplotlib.axes import Axes


def vis_fid_vs_p2q(input_to_fid: dict[str, float]):
    pass


def vis_bar_fid_vs_input(ax: Axes, input_to_fid: dict[str, float]):
    keys = sorted(input_to_fid.keys())
    values = [input_to_fid[k] for k in keys]
    # Create the bar plot
    ax.bar(keys, values)
    ax.set_xlabel("Input Bitstring")
    ax.set_ylabel("Fidelity")
    ax.set_title("Fidelity vs Input State")
    ax.set_ylim(0, 1.05)  # Set y-axis to go up to 1.05 for clarity
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def vis_line_avg_fid_vs_n_trgts(ax: Axes, n_trgts_lst: list[int], avg_fids: Sequence[float]):
    ax.plot(n_trgts_lst, avg_fids, marker='o')
    ax.set_xticks(n_trgts_lst)
    ax.set_xlabel("Number of Targets")
    ax.set_ylabel("Average Fidelity")
    ax.set_title("Average Fidelity vs Number of Targets")
