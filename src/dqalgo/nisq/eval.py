
import itertools

import numpy as np
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .circuits import get_Fanout_circ_by_GHZ_w_reset
from .utils import get_register_counts


def normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {state: count/total for state, count in counts.items()}


def compute_classical_fidelity(ideal_probs: dict[str, float], noisy_probs: dict[str, float]) -> float:
    # Compute fidelity for a given input state using square-root overlap
    fidelity = 0
    # Ensure we sum over all possible states; union of keys from both dictionaries
    all_states = set(ideal_probs.keys()).union(noisy_probs.keys())
    # print(all_states)
    for state in all_states:
        p_ideal = ideal_probs.get(state, 0)
        p_noisy = noisy_probs.get(state, 0)
        fidelity += np.sqrt(p_ideal * p_noisy)
    return fidelity


def get_truth_table_tomography_for_Fanout(
    n_trgts: int,
    noise_model: NoiseModel,
    n_shots: int,
) -> dict[str, float]:
    n_qubits = n_trgts + 1

    input_to_fid: dict[str, float] = {}
    for init_bits in itertools.product([0, 1], repeat=n_qubits):
        # in QuantumCircuit.initialize, leftmost bit is the (N-1)-th qubit, and rightmost bit is the 0-th qubit
        # here rightmost bit is the control qubit
        ctrl_bit = init_bits[-1]
        init_bitstr = "".join(map(str, init_bits))
        expected_trgt_bits = [int((trgt_bit + ctrl_bit) % 2) for trgt_bit in init_bits[:-1]]
        expected_trgt_bitstr = "".join(map(str, expected_trgt_bits + [ctrl_bit]))
        initial_state = [reg for pair in zip([0] * n_qubits, init_bits) for reg in pair]
        init_bitstr_w_anc = "".join(map(str, initial_state))

        qc = get_Fanout_circ_by_GHZ_w_reset(n_trgts, init_bitstr_w_anc, meas_all=True)
        # counts = AerSimulator(max_parallel_threads=max_parallel_threads).run(
        #     qc, shots=n_shots).result().get_counts()
        # reg_counts = get_register_counts(counts, [2*n_qubits, n_qubits], 't', ['a', 't'])
        # ideal_counts = normalize_counts(reg_counts)
        ideal_counts = {expected_trgt_bitstr: 1.0}

        counts = AerSimulator(noise_model=noise_model).run(
            qc, shots=n_shots).result().get_counts()
        reg_counts = get_register_counts(counts, [2*n_qubits, n_qubits], 't', ['a', 't'])
        noisy_counts = normalize_counts(reg_counts)

        fid = compute_classical_fidelity(ideal_counts, noisy_counts)
        input_to_fid[init_bitstr] = fid

    return input_to_fid
