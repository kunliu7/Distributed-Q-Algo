import itertools

import numpy as np
from qiskit.quantum_info import process_fidelity
from qiskit.visualization import plot_state_city
from qiskit_aer import AerSimulator
from qiskit_experiments.library import ProcessTomography

from dqalgo.nisq.circuits import get_parallel_toffoli_via_fanout_circ
from dqalgo.nisq.eval import compute_classical_fidelity, normalize_counts
from dqalgo.nisq.experimental_noise import get_fanout_error_probs
from dqalgo.nisq.utils import get_depolarizing_noise_model, get_register_counts

def test_parallel_toffoli_ideal():
    for n_trgts in [1]: #range(1, 8, 2):  # only odd number of targets are supported
        print(f"Testing {n_trgts} targets")
        n_qubits = 2*n_trgts + 1

        # test all possible initial basis states
        for init_bitstr in itertools.product([0, 1], repeat=n_qubits):
            # in QuantumCircuit.initialize, leftmost bit is the (N-1)-th qubit, and rightmost bit is the 0-th qubit
            # here rightmost bit is the control qubit
            ctrl_bit_1 = init_bitstr[-1]
            ctrl_bits_2 = init_bitstr[:n_trgts]
            trgt_bits = init_bitstr[n_trgts:-1]
            init_bitstr = "".join(map(str, init_bitstr))
            expected_trgt_bits = [int((trgt_bit + (ctrl_bit_1 * ctrl_bit_2) % 2)) for ctrl_bit_2, trgt_bit in zip(ctrl_bits_2, trgt_bits)]
            expected_trgt_bitstr = "".join(map(str, expected_trgt_bits + ctrl_bits_2 + [ctrl_bit_1]))
            initial_state = [reg for pair in zip([0] * n_qubits, init_bitstr) for reg in pair]


            # expected_trgt_bits = [int((trgt_bit + ctrl_bit) % 2) for trgt_bit in init_bitstr[:-1]]
            # initial_state = [reg for pair in zip([0] * n_qubits, init_bitstr) for reg in pair]
            init_bitstr = "".join(map(str, initial_state))

            qc = get_parallel_toffoli_via_fanout_circ(n_trgts, init_bitstr, meas_all=True)
            counts = AerSimulator().run(qc).result().get_counts()
            reg_counts = get_register_counts(counts, [n_qubits], 't', ['t'])
            assert len(reg_counts) == 1
            assert list(reg_counts.keys())[0] == expected_trgt_bitstr


def test_truth_table_tomography():
    n_trgts = 4
    n_qubits = 2*n_trgts + 1
    p_err = 0.0001

    n_fanout_errors1 = get_fanout_error_probs(n_trgts=n_trgts, p2=10*p_err)
    two_n_fanout_errors = get_fanout_error_probs(n_trgts=2*n_trgts, p2=10*p_err)

    input_to_fid: dict[str, float] = {}

    for init_bits in itertools.product([0, 1], repeat=n_qubits):
        # in QuantumCircuit.initialize, leftmost bit is the (N-1)-th qubit, and rightmost bit is the 0-th qubit
        # here rightmost bit is the control qubit
        ctrl_bit_1 = init_bits[-1]
        ctrl_bits_2 = list(init_bits[n_trgts:-1])
        trgt_bits = list(init_bits[:n_trgts])
        expected_target_bits = [int(((trgt_bit + (ctrl_bit_1 * ctrl_bit_2)) % 2)) for ctrl_bit_2, trgt_bit in zip(ctrl_bits_2, trgt_bits)]
        expected_output_bitstr = "".join(map(str, expected_target_bits + ctrl_bits_2 + [ctrl_bit_1]))
        init_bitstr = "".join(map(str, init_bits))

        print(init_bitstr, expected_output_bitstr)

        ideal_counts = {expected_output_bitstr: 1.0}

        noise_model = get_depolarizing_noise_model(p_1q=p_err, p_2q=p_err*10, p_meas=p_err)

        total_reg_counts = {}
        for _ in range(100):
            qc = get_parallel_toffoli_via_fanout_circ(
                n_trgts=n_trgts,
                init_bitstr=init_bitstr,
                meas_all=True,
                n_fanout_errors=n_fanout_errors1,
                two_n_fanout_errors=two_n_fanout_errors
            )
            counts = AerSimulator(noise_model=noise_model).run(qc, shots=20).result().get_counts()
            reg_counts = get_register_counts(counts, [n_qubits], 't', ['t'])

            for k, v in reg_counts.items():
                if k not in total_reg_counts:
                    total_reg_counts[k] = 0
                total_reg_counts[k] += v

        noisy_counts = normalize_counts(total_reg_counts)
        # builder = ParallelToffoliBuilder(n_trgts, ctrl_bit_1, ctrl_bits_2, trgt_bits)
        # noisy_counts = builder.simulate(init_bits, shots=shots)

        fid = compute_classical_fidelity(ideal_counts, noisy_counts)
        input_to_fid[init_bits] = fid
        print(f"fid for {init_bits}: {fid}")

    print(input_to_fid)

    keys = sorted(input_to_fid.keys())
    mean_fid = sum(input_to_fid.values())/2**n_qubits

    print(f"mean fidelity: {mean_fid}")

    # Convert the dictionary values into a numpy array in the sorted order.
    # We cast to complex to meet the input requirements for plot_state_city.
    state_vector = np.array([input_to_fid[k] for k in keys], dtype=complex)

    fig = plot_state_city(state_vector)

