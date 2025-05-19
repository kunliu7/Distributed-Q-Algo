import itertools

import numpy as np


from dqalgo.nisq.circuits import get_parallel_toffoli_via_fanout_circ
from dqalgo.nisq.eval import compute_classical_fidelity, normalize_counts
from dqalgo.nisq.experimental_noise import get_fanout_error_probs
from dqalgo.nisq.utils import get_depolarizing_noise_model, get_register_counts

from qiskit import transpile
from qiskit.quantum_info import process_fidelity
from qiskit.visualization import plot_state_city
from qiskit_aer import AerSimulator
from qiskit_experiments.library import ProcessTomography

from tqdm import tqdm
import random

def test_parallel_toffoli_ideal():
    for n_trgts in [1]: #range(1, 8, 2):  # only odd number of targets are supported
        print(f"Testing {n_trgts} targets")
        n_qubits = 2*n_trgts + 1

        # test all possible initial basis states
        for input_bitstr in itertools.product([0, 1], repeat=n_qubits):
            # in QuantumCircuit.initialize, leftmost bit is the (N-1)-th qubit, and rightmost bit is the 0-th qubit
            # here rightmost bit is the control qubit
            ctrl_bit_1 = input_bitstr[-1]
            ctrl_bits_2 = input_bitstr[:n_trgts]
            trgt_bits = input_bitstr[n_trgts:-1]
            input_bitstr = "".join(map(str, input_bitstr))
            expected_trgt_bits = [int((trgt_bit + (ctrl_bit_1 * ctrl_bit_2) % 2)) for ctrl_bit_2, trgt_bit in zip(ctrl_bits_2, trgt_bits)]
            expected_trgt_bitstr = "".join(map(str, expected_trgt_bits + ctrl_bits_2 + [ctrl_bit_1]))
            initial_state = [reg for pair in zip([0] * n_qubits, input_bitstr) for reg in pair]


            # expected_trgt_bits = [int((trgt_bit + ctrl_bit) % 2) for trgt_bit in input_bitstr[:-1]]
            # initial_state = [reg for pair in zip([0] * n_qubits, input_bitstr) for reg in pair]
            input_bitstr = "".join(map(str, initial_state))

            qc = get_parallel_toffoli_via_fanout_circ(n_trgts, input_bitstr, meas_all=True)
            counts = AerSimulator().run(qc).result().get_counts()
            reg_counts = get_register_counts(counts, [n_qubits], 't', ['t'])
            assert len(reg_counts) == 1
            assert list(reg_counts.keys())[0] == expected_trgt_bitstr


def get_bitstrings(n_qubits: int, n_samples: int):
    sample_indices = random.sample(range(2**n_qubits), min(2**n_qubits, n_samples))
    for index in sample_indices:
        yield bin(index)[2:].zfill(n_qubits)


def get_expected_output(input_bitstr: str | list[int]) -> str:
    n_trgts = (len(input_bitstr) - 1)//2
    ctrl_bit_1 = input_bitstr[-1]
    ctrl_bits_2 = list(input_bitstr[n_trgts:-1])
    trgt_bits = list(input_bitstr[:n_trgts])
    expected_target_bits = [int(((int(trgt_bit) + (int(ctrl_bit_1) * int(ctrl_bit_2))) % 2)) for ctrl_bit_2, trgt_bit in zip(ctrl_bits_2, trgt_bits)]
    return ''.join(map(str, expected_target_bits + ctrl_bits_2 + [ctrl_bit_1]))

def update_total_counts(total_counts: dict[str, int], sub_counts: dict[str, int]) -> None:
    for k, v in sub_counts.items():
        if k not in total_counts:
            total_counts[k] = 0

        total_counts[k] += v

def test_truth_table_tomography():
    n_trgts = 3
    n_qubits = 2*n_trgts + 1
    p_err = 0.0001

    shots_per_circ = 128
    circs_per_input = 10
    n_samples = 150
    n_samples = min(n_samples, 2**n_qubits)

    n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=10*p_err)
    two_n_fanout_errors = get_fanout_error_probs(n_trgts=2*n_trgts, p2=10*p_err)

    fids = []
    noise_model = get_depolarizing_noise_model(p_1q=p_err, p_2q=p_err*10, p_meas=p_err)
    sim = AerSimulator(noise_model=noise_model)

    print('Constructing circuits')

    for input_bitstr in tqdm(get_bitstrings(n_qubits, n_samples), total=n_samples):
        # in QuantumCircuit.initialize, leftmost bit is the (N-1)-th qubit, and rightmost bit is the 0-th qubit
        # here rightmost bit is the control qubit
        expected_output_bitstr = get_expected_output(input_bitstr)
        ideal_counts = {expected_output_bitstr: 1.0}

        total_counts = {}
        for _ in range(circs_per_input):
            qc = get_parallel_toffoli_via_fanout_circ(
                n_trgts=n_trgts,
                input_bitstr=input_bitstr,
                meas_all=True,
                n_fanout_errors=n_fanout_errors,
                two_n_fanout_errors=two_n_fanout_errors
            )
            results = sim.run(qc, shots=shots_per_circ).result()
            counts = results.get_counts()
            update_total_counts(total_counts, counts)

        normed_noisy_counts = normalize_counts(total_counts)
        fid = compute_classical_fidelity(ideal_counts, normed_noisy_counts)
        fids.append(fid)

    mean_fid = np.mean(fids)
    stddev_fid = np.std(fids)

    print(f"mean fidelity: {mean_fid}")
    print(f"stddev fidelity: {stddev_fid}")

