from dqalgo.data_mgr import (
    NISQFanoutDataMgr,
    NISQTeledataDataMgr,
    NISQTeleportedCNOTsDataMgr
)

"""
Retrieve experimenatlly determined noise model for the various circuits

Format:
{n_targets: {error: probability}}

n_targets, the number of target qubits for the fanout
error, the pauli string of the error type
probability, the probability of the error
"""

def get_fanout_error_probs(n_trgts: int, p2: float) -> list[tuple[str, float]]:
    """Get the experimentally determined error probabilities for the fanout gate."""
    n_trgts_lst = [1,2,3,4,5,6,7,8,9,10,11,12]
    p2s = [0.001, 0.003, 0.005]
    n_shots = 100000

    if n_trgts not in n_trgts_lst:
        raise ValueError(f"n_trgts must be one of {n_trgts_lst}")
    if p2 not in p2s:
        raise ValueError(f"p2 must be one of {p2s}")

    dmgr = NISQFanoutDataMgr()
    data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
    error_counts_lst, _, _ = data

    n_trgts_idx = n_trgts_lst.index(n_trgts)
    p2_idx = p2s.index(p2)
    error_counts = error_counts_lst[n_trgts_idx * len(p2s) + p2_idx]
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    errors_tuple = [(error[1:].replace('_', 'I'), count/100000) for error, count in top_errors]

    return errors_tuple

def get_teledata_error_probs(n_trgts: int, p2: float) -> list[tuple[str, float]]:
    """Get the experimentally determined error probabilities for teleporting the qubits."""
    n_trgts_lst = [1,2,3,4,5,6]
    p2s = [0.001, 0.003, 0.005]
    n_shots = 100000

    if n_trgts not in n_trgts_lst:
        raise ValueError(f"n_trgts must be one of {n_trgts_lst}")
    if p2 not in p2s:
        raise ValueError(f"p2 must be one of {p2s}")

    dmgr = NISQTeledataDataMgr()
    data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
    error_counts_lst, _, _ = data

    n_trgts_idx = n_trgts_lst.index(n_trgts)
    p2_idx = p2s.index(p2)
    error_counts = error_counts_lst[n_trgts_idx * len(p2s) + p2_idx]
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    errors_tuple = [(error[1:].replace('_', 'I'), count/100000) for error, count in top_errors]

    return errors_tuple

def get_pre_teletoffoli_error_probs(n_trgts: int, p2: float) -> list[tuple[str, float]]:
    """
    Get the experimentally determined error probabilities for pre-toffoli step
    of the Toffoli teleportation scheme.
    """
    n_trgts_lst = [1,2,3,4,5,6,7,8]
    p2s = [0.001, 0.003, 0.005]
    n_shots = 100000

    if n_trgts not in n_trgts_lst:
        raise ValueError(f"n_trgts must be one of {n_trgts_lst}")
    if p2 not in p2s:
        raise ValueError(f"p2 must be one of {p2s}")

    dmgr = NISQTeledataDataMgr()
    data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
    error_counts_lst, _, _ = data

    n_trgts_idx = n_trgts_lst.index(n_trgts)
    p2_idx = p2s.index(p2)
    error_counts = error_counts_lst[n_trgts_idx * len(p2s) + p2_idx]
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    errors_tuple = [(error[1:].replace('_', 'I'), count/100000) for error, count in top_errors]

    return errors_tuple


def get_telecnot_error_probs(n_trgts: int, p2: float) -> list[tuple[str, float]]:
    """
    Get the experimentally determined error probabilities for teleporting the CNOT gate.
    """
    n_trgts_lst = [1,2,3,4,5,6]
    p2s = [0.001, 0.003, 0.005]
    n_shots = 100000

    if n_trgts not in n_trgts_lst:
        raise ValueError(f"n_trgts must be one of {n_trgts_lst}")
    if p2 not in p2s:
        raise ValueError(f"p2 must be one of {p2s}")

    dmgr = NISQTeleportedCNOTsDataMgr()
    data = dmgr.load(n_trgts=n_trgts_lst, p2=p2s, n_shots=n_shots)
    error_counts_lst, _, _ = data

    n_trgts_idx = n_trgts_lst.index(n_trgts)
    p2_idx = p2s.index(p2)
    error_counts = error_counts_lst[n_trgts_idx * len(p2s) + p2_idx]
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    errors_tuple = [(error[1:].replace('_', 'I'), count/100000) for error, count in top_errors]

    return errors_tuple


