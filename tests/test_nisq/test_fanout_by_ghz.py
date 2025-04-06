import itertools

from qiskit_aer import AerSimulator

from dqalgo.nisq.circuits import get_Fanout_circ_by_GHZ_w_reset
from dqalgo.nisq.utils import get_register_counts


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
