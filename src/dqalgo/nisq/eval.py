
import itertools

import numpy as np
import stim
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from tqdm import tqdm

from dqalgo.nisq.experimental_noise import get_fanout_error_probs

from .circuits import (
    get_CSWAP_teledata_fewer_ancillas_circ,
    get_CSWAP_telegate_fewer_ancillas_circ,
    get_Fanout_circ_by_GHZ_w_reset,
)

from .fanouts import BaumerFanoutBuilder
from .utils import classically_compute_CSWAP, get_counts_of_first_n_regs, get_depolarizing_noise_model, get_register_counts, sample_bitstrings, update_total_counts


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

        counts = AerSimulator(noise_model=noise_model).run(qc, shots=n_shots).result().get_counts()
        reg_counts = get_register_counts(counts, [2*n_qubits, n_qubits], 't', ['a', 't'])
        noisy_counts = normalize_counts(reg_counts)

        fid = compute_classical_fidelity(ideal_counts, noisy_counts)
        input_to_fid[init_bitstr] = fid

    return input_to_fid


def eval_Baumer_Fanout(n_trgts: int, p1: float, p2: float, pm: float, n_shots: int
                       ) -> dict[str, int]:
    """Evaluate the Baumer Fanout circuit with the given noise parameters.

    Returns:
        dict[str, int]: error counts, e.g. {'XZ': 100, 'YZ': 100}
    """
    ctrl_bit = 0
    init_trgt_bits = [0] * n_trgts
    ideal_simulator, _ = BaumerFanoutBuilder(
        n_trgts=n_trgts, ctrl_bit=ctrl_bit, init_trgt_bits=list(init_trgt_bits)).build_in_stim()
    ideal_inv_tableau = ideal_simulator.current_inverse_tableau()
    error_counts = {}

    for _ in tqdm(range(n_shots), desc=f"Sim {n_trgts=}, {p2=}"):
        builder = BaumerFanoutBuilder(n_trgts=n_trgts, ctrl_bit=ctrl_bit,
                                      init_trgt_bits=list(init_trgt_bits),
                                      p1=p1, p2=p2, pm=pm)
        noisy_simulator, _ = builder.build_in_stim()
        noisy_inv_tableau = noisy_simulator.current_inverse_tableau()
        pauli_error = (noisy_inv_tableau.inverse() * ideal_inv_tableau).to_pauli_string()
        remaining_pauli_error = pauli_error[::2]  # remove ancillary qubits
        if remaining_pauli_error != stim.PauliString("I"*(n_trgts+1)):
            key = str(remaining_pauli_error)
            error_counts[key] = error_counts.get(key, 0) + 1

    return error_counts

def eval_CSWAP_teledata(n_trgts: int, p_err: float) -> tuple[float, float]:
    n_data_qubits = 2*n_trgts + 1

    shots_per_circ = 128 # Crashes with anything higher than 128
    circs_per_input = 10 # Repeat 10 times to compensate for low shots
    n_samples = 150 # Sample space gets large so choose 150 random input bitstrings
    n_samples = min(n_samples, 2**n_data_qubits)

    n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=10*p_err)
    two_n_fanout_errors = get_fanout_error_probs(n_trgts=2*n_trgts, p2=10*p_err)

    fids = []
    noise_model = get_depolarizing_noise_model(p_1q=p_err, p_2q=p_err*10, p_meas=p_err)
    sim = AerSimulator(noise_model=noise_model)

    print('Constructing circuits')

    for input_bitstr in tqdm(sample_bitstrings(n_data_qubits, n_samples), total=n_samples, dynamic_ncols=False, leave=True):
        expected_output_bitstr = classically_compute_CSWAP(input_bitstr)
        ideal_counts = {expected_output_bitstr: 1.0}

        total_counts = {}
        for _ in range(circs_per_input):
            qc = get_CSWAP_teledata_fewer_ancillas_circ(
                input_bitstr=input_bitstr,
                meas_all=True,
                n_fanout_errors=n_fanout_errors,
                two_n_fanout_errors=two_n_fanout_errors
            )

            results = sim.run(qc, shots=shots_per_circ).result()
            counts = results.get_counts()
            reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
            update_total_counts(total_counts, reg_counts)

        normed_noisy_counts = normalize_counts(total_counts)
        fid = compute_classical_fidelity(ideal_counts, normed_noisy_counts)
        fids.append(fid)

    mean_fid = np.mean(fids)
    stddev_fid = np.std(fids)

    return mean_fid, stddev_fid


def eval_CSWAP_telegate(n_trgts: int, p_err: float, shots_per_circ=128, circs_per_input=10, n_samples=150) -> tuple[float, float]:
    # The script seems to crash with anything higher than 128 shots
    n_data_qubits = 2*n_trgts + 1

    n_samples = min(n_samples, 2**n_data_qubits)

    n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=10*p_err)
    two_n_fanout_errors = get_fanout_error_probs(n_trgts=2*n_trgts, p2=10*p_err)

    fids = []
    noise_model = get_depolarizing_noise_model(p_1q=p_err, p_2q=p_err*10, p_meas=p_err)
    sim = AerSimulator(noise_model=noise_model)

    print('Constructing circuits')

    for input_bitstr in tqdm(sample_bitstrings(n_data_qubits, n_samples), total=n_samples, dynamic_ncols=False, leave=True):
        expected_output_bitstr = classically_compute_CSWAP(input_bitstr)
        ideal_counts = {expected_output_bitstr: 1.0}

        total_counts = {}
        for _ in range(circs_per_input):
            qc = get_CSWAP_telegate_fewer_ancillas_circ(
                input_bitstr=input_bitstr,
                meas_all=True,
                n_fanout_errors=n_fanout_errors,
                two_n_fanout_errors=two_n_fanout_errors
            )

            results = sim.run(qc, shots=shots_per_circ).result()
            counts = results.get_counts()
            reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
            update_total_counts(total_counts, reg_counts)

        normed_noisy_counts = normalize_counts(total_counts)
        fid = compute_classical_fidelity(ideal_counts, normed_noisy_counts)
        fids.append(fid)

    mean_fid = np.mean(fids)
    stddev_fid = np.std(fids)

    return mean_fid, stddev_fid