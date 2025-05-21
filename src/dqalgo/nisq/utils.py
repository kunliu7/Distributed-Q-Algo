import stim
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error
from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
import random

def get_register_counts(counts: dict[str, int], creg_sizes: list[int], target_reg_name: str,
                        reg_names: list[str]) -> dict[str, int]:
    """If the circuit has multiple target registers, this function can be used to get the counts of a specific target register.

    Args:
        counts (dict[str, int]): the counts of the circuit, e.g. {'0000000': 100, '0000001': 100, '0000010': 100, '0000011': 100, '0000100': 100, '0000101': 100, '0000110': 100, '0000111': 100}
        reg_names (list[str]): the names of the classical registers, e.g. ['a', 't']
        creg_sizes (list[int]): the sizes of the corresponding classical registers of `reg_names`, e.g. [3, 4]
        target_reg_name (str): the name of the register that we want to get the counts of, e.g. 't'

    Returns:
        dict[str, int]: the counts of the target register; the rightmost bit is the least significant bit (0-th bit)
    """
    idx = reg_names.index(target_reg_name)
    start = sum(creg_sizes[:idx])
    end = start + creg_sizes[idx]

    reg_counts = {}
    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "")  # remove spaces
        bitstring = bitstring[::-1]  # little endian
        reg_bits = bitstring[start:end][::-1]
        reg_counts[reg_bits] = reg_counts.get(reg_bits, 0) + count
    return reg_counts


def get_depolarizing_noise_model(
    p_1q: float, p_2q: float, p_meas: float,
) -> NoiseModel:
    """Get a depolarizing noise model.

    Args:
        p_1q (float): the probability of a 1-qubit depolarizing noise
        p_2q (float): the probability of a 2-qubit depolarizing noise
        p_meas (float): the probability of a measurement noise
    """
    noise_model = NoiseModel()
    if p_1q > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p_1q, 1), ['u1', 'u2', 'u3'])
    if p_2q > 0:
        noise_model.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ['cx'])
    if p_meas > 0:
        noise_model.add_all_qubit_readout_error(ReadoutError([[1-p_meas, p_meas], [p_meas, 1-p_meas]]))
    return noise_model


def add_fanout_monte_carlo_error(qc: QuantumCircuit, qubit_indices: list[QuantumRegister], error_probs: tuple[str, float]) -> None:
    for pauli_str, prob in error_probs:
        if np.random.rand() < prob:
            # Apply the Pauli error to the qubits in the specified indices.
            for i, qubit_index in enumerate(qubit_indices):
                if pauli_str[i] == "X":
                    qc.x(qubit_index)
                elif pauli_str[i] == "Y":
                    qc.y(qubit_index)
                elif pauli_str[i] == "Z":
                    qc.z(qubit_index)

def add_fanout_custom_error_injection(qc: QuantumCircuit, qubit_indices: list[QuantumRegister], error_probs: tuple[str, float]) -> None:
    for pauli_str, prob in error_probs:
        fanout_error = pauli_error([
            (pauli_str, prob),
            ("I" * len(pauli_str), 1 - prob)
        ]).to_instruction()
        qc.append(fanout_error, qubit_indices)



def reduce_stabilizers(stabs: list[stim.PauliString], keep_ids: list[int]) -> list[stim.PauliString]:
    reduced = []
    for stab in stabs:
        # Convert to string representation.
        full_str = str(stab)
        # Build the reduced Pauli string using only the kept indices.
        # (Make sure the ordering is what you expect.)
        pauli_str = full_str[1:]
        sign = full_str[0]
        # print(f"pauli_str: {pauli_str}, sign: {sign}")
        reduced_pauli_str = "".join(pauli_str[i] for i in keep_ids)
        # Optionally, you might skip generators that become trivial.
        if set(reduced_pauli_str) != {"_"}:
            # print(f"{stab=}, reduced_pauli_str: {reduced_pauli_str}, set(reduced_pauli_str): {set(reduced_pauli_str)}")
            reduced.append(stim.PauliString(sign + reduced_pauli_str))
    print(f"reduced: {reduced}")
    return reduced


def reduce_tableau(tab: stim.Tableau, keep_ids: list[int]) -> stim.Tableau:
    stabs = tab.to_stabilizers()
    reduced_stabs = reduce_stabilizers(stabs, keep_ids)
    return stim.Tableau.from_stabilizers(reduced_stabs, allow_redundant=True)

def sample_bitstrings(n_qubits: int, n_samples: int):
    sample_indices = random.sample(range(2**n_qubits), min(2**n_qubits, n_samples))
    for index in sample_indices:
        yield bin(index)[2:].zfill(n_qubits)

def update_total_counts(total_counts: dict[str, int], sub_counts: dict[str, int]) -> None:
    for k, v in sub_counts.items():
        if k not in total_counts:
            total_counts[k] = 0

        total_counts[k] += v

def classically_compute_toffoli(input_bitstr: str | list[int]) -> str:
    n_trgts = (len(input_bitstr) - 1)//2
    ctrl_bit_1 = input_bitstr[-1]
    ctrl_bits_2 = list(input_bitstr[n_trgts:-1])
    trgt_bits = list(input_bitstr[:n_trgts])
    expected_target_bits = [int(((int(trgt_bit) + (int(ctrl_bit_1) * int(ctrl_bit_2))) % 2)) for ctrl_bit_2, trgt_bit in zip(ctrl_bits_2, trgt_bits)]
    return ''.join(map(str, expected_target_bits + ctrl_bits_2 + [ctrl_bit_1]))

def classically_compute_CSWAP(input_bitstr: str | list[int]) -> str:
    n_trgts = (len(input_bitstr) - 1)//2
    ctrl_bit = int(input_bitstr[-1])
    state_A = list(input_bitstr[n_trgts:-1])
    state_B = list(input_bitstr[:n_trgts])
    expected_bitstr = (
            (state_B + state_A + [ctrl_bit])
            if ctrl_bit == 0
            else (state_A + state_B + [ctrl_bit])
        )
    return ''.join(map(str, expected_bitstr))