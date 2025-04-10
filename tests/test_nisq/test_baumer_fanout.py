import itertools

import numpy as np
import stim
from qiskit_aer import AerSimulator

from dqalgo.nisq.fanouts import BaumerFanoutBuilder


def test_baumer_fanout():
    for n_trgts in [4, 6, 8]:
        for ctrl_bit in [0, 1]:
            print(f"n_trgts: {n_trgts}, ctrl_bit: {ctrl_bit}")
            for init_trgt_bits in itertools.product([0, 1], repeat=n_trgts):
                builder = BaumerFanoutBuilder(n_trgts=n_trgts, ctrl_bit=ctrl_bit, init_trgt_bits=list(init_trgt_bits))
                expected_trgt_bits = [(_bit + ctrl_bit) % 2 for _bit in init_trgt_bits]
                expected_trgt_bitstr = "".join(map(str, expected_trgt_bits))
                circ = builder.build_w_fewer_cnots()
                rst = AerSimulator().run(circ).result()
                counts = rst.get_counts()
                for measurement in counts:
                    measurement = measurement.replace(" ", "")
                    # Assuming the bit string format is "ctrl trgts ancs"
                    rst_trgt_bits = measurement[:builder.n_trgts]  # middle bits are targets
                    rst_ctrl_bit = measurement[-1]  # last bit is ctrl
                    assert rst_trgt_bits == expected_trgt_bitstr, \
                        f"trgt_bits: {rst_trgt_bits}, expected_trgt_bits: {expected_trgt_bitstr}"
                    assert rst_ctrl_bit == str(ctrl_bit), \
                        f"ctrl_bit: {rst_ctrl_bit}, expected_ctrl_bit: {ctrl_bit}"


def test_baumer_fanout_stim_ideal():
    """Make sure that the ideal Baumer fanout circuit is correct for all the 
    possible computational basis states."""
    for n_trgts in [4, 6, 8]:
        for ctrl_bit in [0, 1]:
            print(f"n_trgts: {n_trgts}, ctrl_bit: {ctrl_bit}")
            for init_trgt_bits in itertools.product([0, 1], repeat=n_trgts):
                # print(f"init_trgt_bits: {init_trgt_bits}")
                for _ in range(1000):
                    builder = BaumerFanoutBuilder(n_trgts=n_trgts, ctrl_bit=ctrl_bit,
                                                  init_trgt_bits=list(init_trgt_bits))
                    _, rst = builder.build_in_stim()
                    rst_trgt_bits = rst["target_measurements"][::-1]
                    rst_ctrl_bit = rst["control_measurement"]
                    expected_trgt_bits = [(_bit + ctrl_bit) % 2 for _bit in init_trgt_bits]
                    assert rst_ctrl_bit == ctrl_bit, \
                        f"ctrl_bit: {rst_ctrl_bit}, expected_ctrl_bit: {ctrl_bit}"
                    assert np.all(rst_trgt_bits == expected_trgt_bits), \
                        f"trgt_bits: {rst_trgt_bits}, expected_trgt_bits: {expected_trgt_bits}"


def test_baumer_fanout_stim_noisy():
    """Initialize in all 0 states and get the Pauli error distribution of the noisy circuit.
    If p2 set to 0, then error_counts is empty.
    """
    p2 = 0.001
    p1 = p2 / 10
    pm = p2
    print(f"p1: {p1}, p2: {p2}, pm: {pm}")
    n_shots = 100000
    for n_trgts in [4, 6, 8]:
        ctrl_bit = 0
        print(f"n_trgts: {n_trgts}, ctrl_bit: {ctrl_bit}")
        keep_ids = list(range(0, 2*n_trgts+1, 2))  # data qubits, including control qubit
        # for init_trgt_bits in itertools.product([0, 1], repeat=n_trgts):
        init_trgt_bits = [0] * n_trgts
        # print(f"init_trgt_bits: {init_trgt_bits}")
        ideal_simulator, _ = BaumerFanoutBuilder(
            n_trgts=n_trgts, ctrl_bit=ctrl_bit, init_trgt_bits=list(init_trgt_bits)).build_in_stim()
        ideal_inv_tableau = ideal_simulator.current_inverse_tableau()
        error_counts = {}

        for _ in range(n_shots):
            builder = BaumerFanoutBuilder(n_trgts=n_trgts, ctrl_bit=ctrl_bit,
                                          init_trgt_bits=list(init_trgt_bits),
                                          p1=p1, p2=p2, pm=pm)
            noisy_simulator, _ = builder.build_in_stim()
            noisy_inv_tableau = noisy_simulator.current_inverse_tableau()
            pauli_error = (ideal_inv_tableau.inverse() * noisy_inv_tableau).to_pauli_string()
            remaining_pauli_error = pauli_error[::2]  # remove ancillary qubits
            # print(f"remaining_pauli_error: {remaining_pauli_error}, pauli_error: {pauli_error}")

            # print(pauli_error)
            if remaining_pauli_error != stim.PauliString("I"*(n_trgts+1)):
                key = str(remaining_pauli_error)
                error_counts[key] = error_counts.get(key, 0) + 1

        # Sort error_counts by value in descending order and get top 5
        top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 errors:")
        for error, count in top_errors:
            print(f"  {error}: {count}/{n_shots}")
        # print("All errors:")
        # print(error_counts)
