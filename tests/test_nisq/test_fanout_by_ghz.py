import itertools

import numpy as np
from qiskit.quantum_info import process_fidelity
from qiskit.visualization import plot_state_city
from qiskit_aer import AerSimulator
from qiskit_experiments.library import ProcessTomography

from dqalgo.nisq.circuits import get_Fanout_circ_by_GHZ_w_reset
from dqalgo.nisq.eval import compute_classical_fidelity, normalize_counts
from dqalgo.nisq.utils import get_depolarizing_noise_model, get_register_counts


def test_fanout_by_ghz_ideal():
    for n_trgts in range(1, 8, 2):  # only odd number of targets are supported
        print(f"Testing {n_trgts} targets")
        n_qubits = n_trgts + 1

        # test all possible initial basis states
        for init_bitstr in itertools.product([0, 1], repeat=n_qubits):
            # in QuantumCircuit.initialize, leftmost bit is the (N-1)-th qubit, and rightmost bit is the 0-th qubit
            # here rightmost bit is the control qubit
            ctrl_bit = init_bitstr[-1]
            expected_trgt_bits = [int((trgt_bit + ctrl_bit) % 2) for trgt_bit in init_bitstr[:-1]]
            initial_state = [reg for pair in zip([0] * n_qubits, init_bitstr) for reg in pair]
            init_bitstr = "".join(map(str, initial_state))

            qc = get_Fanout_circ_by_GHZ_w_reset(n_trgts, init_bitstr, meas_all=True)
            counts = AerSimulator().run(qc).result().get_counts()
            reg_counts = get_register_counts(counts, [2*n_qubits, n_qubits], 't', ['a', 't'])
            assert len(reg_counts) == 1
            assert list(reg_counts.keys())[0] == "".join(map(str, expected_trgt_bits + [ctrl_bit]))


def test_fanout_by_ghz_noisy_with_superop():
    # fail to use `SuperOp`: qiskit.exceptions.QiskitError: 'Circuits with control flow operations cannot be converted to an instruction.'
    noise_model = get_depolarizing_noise_model(p_1q=0.001, p_2q=0.001, p_meas=0.001)
    n_trgts = 3
    print(f"Testing {n_trgts} targets")
    n_qubits = n_trgts + 1

    qc = get_Fanout_circ_by_GHZ_w_reset(n_trgts, None, meas_all=True)
    qptexp = ProcessTomography(qc)
    qptdata = qptexp.run(backend=AerSimulator(noise_model=noise_model,
                         shots=1000, seed_simulator=12345)).block_for_results()
    choi_out = qptdata.analysis_results("state", dataframe=True).iloc[0].values

    fid = process_fidelity(ideal_superop, noisy_superop)
    assert fid > 0.99


def test_truth_table_tomography():
    n_trgts = 3
    n_qubits = n_trgts + 1

    input_to_fid: dict[str, float] = {}
    for init_bits in itertools.product([0, 1], repeat=n_qubits):
        # in QuantumCircuit.initialize, leftmost bit is the (N-1)-th qubit, and rightmost bit is the 0-th qubit
        # here rightmost bit is the control qubit
        ctrl_bit = init_bits[-1]
        init_bitstr = "".join(map(str, init_bits))
        expected_trgt_bits = [int((trgt_bit + ctrl_bit) % 2) for trgt_bit in init_bits[:-1]]
        initial_state = [reg for pair in zip([0] * n_qubits, init_bits) for reg in pair]
        init_bitstr_w_anc = "".join(map(str, initial_state))

        qc = get_Fanout_circ_by_GHZ_w_reset(n_trgts, init_bitstr_w_anc, meas_all=True)
        counts = AerSimulator().run(qc, shots=2048).result().get_counts()
        reg_counts = get_register_counts(counts, [2*n_qubits, n_qubits], 't', ['a', 't'])
        ideal_counts = normalize_counts(reg_counts)

        noise_model = get_depolarizing_noise_model(p_1q=0.0001, p_2q=0.0001, p_meas=0.0001)
        counts = AerSimulator(noise_model=noise_model).run(qc, shots=2048).result().get_counts()
        reg_counts = get_register_counts(counts, [2*n_qubits, n_qubits], 't', ['a', 't'])
        noisy_counts = normalize_counts(reg_counts)

        fid = compute_classical_fidelity(ideal_counts, noisy_counts)
        input_to_fid[init_bitstr] = fid
        print(f"fid for {init_bitstr}: {fid}")
    print(input_to_fid)

    keys = sorted(input_to_fid.keys())

    # Convert the dictionary values into a numpy array in the sorted order.
    # We cast to complex to meet the input requirements for plot_state_city.
    state_vector = np.array([input_to_fid[k] for k in keys], dtype=complex)

    plot_state_city(state_vector)
