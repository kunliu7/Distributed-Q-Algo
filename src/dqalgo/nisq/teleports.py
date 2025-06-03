"""Simulate teleportation circuits using TableauSimulator s.t. we can get the Pauli error distribution."""

import itertools

import stim
from tqdm import tqdm


def apply_Bell_pair_prep(circ: stim.Circuit, bell_A: list[int], bell_B: list[int]) -> None:
    """Apply Bell pair preparation to the given qubits."""
    circ.append("H", bell_A) # type: ignore
    cnot_pairs = itertools.chain(*zip(bell_A, bell_B))
    circ.append("CNOT", cnot_pairs) # type: ignore


def build_teleportation_circuit(n: int) -> stim.Circuit:
    circ = stim.Circuit()
    input_qubits = list(range(n))
    bell_ancilla_qubits = list(range(n, 2 * n))
    output_qubits = list(range(2 * n, 3 * n))
    anc_qubits = list(range(3 * n, 4 * n))

    apply_Bell_pair_prep(circ, bell_ancilla_qubits, output_qubits)
    circ.append("TICK")
    # entangle input states with Bell pairs
    # cnot_pairs = itertools.chain(*zip(input_qubits, bell_ancilla_qubits))
    # circ.append("CNOT", cnot_pairs) # type: ignore
    
    m_in_indices = []
    m_ancilla_indices = []
    for i in range(n):
        q_in = input_qubits[i]
        q_bell_ancilla = bell_ancilla_qubits[i]
        q_out = output_qubits[i]

        # Bell-basis measurement
        circ.append("CNOT", [q_in, q_bell_ancilla])
        circ.append("H", [q_in])
        circ.append("M", [q_in])
        circ.append("M", [q_bell_ancilla])
        m_in_indices.append(2 * i)
        m_ancilla_indices.append(2 * i + 1)

    # Apply corrections based on classical measurement results
    for i in range(n):
        q_out = output_qubits[i]
        m_in = m_in_indices[i]
        m_anc = m_ancilla_indices[i]
        # If m_anc == 1, apply X
        circ.append("CX", [stim.target_rec(m_anc - circ.num_measurements), q_out])
        # If m_in == 1, apply Z
        circ.append("CZ", [stim.target_rec(m_in - circ.num_measurements), q_out])

    for i in range(n):
        q_out = output_qubits[i]
        q_anc = anc_qubits[i]
        circ.append("CNOT", [q_out, q_anc])

    return circ

def analyze_pauli_error(ideal_sim: stim.TableauSimulator, noisy_sim: stim.TableauSimulator, n: int) -> stim.PauliString:
    ideal_inv = ideal_sim.current_inverse_tableau()
    noisy_inv = noisy_sim.current_inverse_tableau()

    # Derive P = U_ideal^† U_noisy
    pauli_error = (ideal_inv.inverse() * noisy_inv).to_pauli_string()

    # Only keep Bob’s output qubits (every 2nd qubit from n to 3n-1)
    # output_indices = [n + 2 * i + 1 for i in range(n)]
    print(pauli_error)
    remaining_error = pauli_error[2*n:3*n]
    print(remaining_error)
    return remaining_error


class TeleportCircBuilder:
    def __init__(self, n: int, p1: float = 0.0, p2: float = 0.0, pm: float = 0.0):
        self.n = n
        self.total_qubits = 3 * n
        self.input_qubits = list(range(n))
        self.bell_ancillas = list(range(n, 2 * n))
        self.output_qubits = list(range(2 * n, 3 * n))
        self.anc_qubits = list(range(3 * n, 4 * n))
        self.sim = stim.TableauSimulator()
        self.measurement_results = []

        self.p1 = p1
        self.p2 = p2
        self.pm = pm

        self._build()

    def _build(self):
        self._prepare_bell_pairs()
        self._teleport_inputs()

    def _prepare_bell_pairs(self):
        for i in range(self.n):
            a = self.bell_ancillas[i]
            b = self.output_qubits[i]
            self.sim.h(a)
            self.apply_1q_gate_error(self.sim, a)
            self.sim.cx(a, b)
            self.apply_2q_gate_error(self.sim, a, b)

    def _teleport_inputs(self):
        for i in range(self.n):
            q_in = self.input_qubits[i]
            q_anc = self.bell_ancillas[i]
            q_out = self.output_qubits[i]

            self.sim.cx(q_in, q_anc)
            self.apply_2q_gate_error(self.sim, q_in, q_anc)
            self.sim.h(q_in)
            self.apply_1q_gate_error(self.sim, q_in)

            self.apply_measurement_error(self.sim, q_in)
            m_in = self.sim.measure(q_in)
            self.apply_measurement_error(self.sim, q_anc)
            m_anc = self.sim.measure(q_anc)
            self.measurement_results.append((m_in, m_anc))

            if m_in:
                self.sim.z(q_out)
                self.apply_1q_gate_error(self.sim, q_out)
            if m_anc:
                self.sim.x(q_out)
                self.apply_1q_gate_error(self.sim, q_out)

        for i in range(self.n):
            q_out = self.output_qubits[i]
            q_anc = self.anc_qubits[i]
            self.sim.cx(q_out, q_anc)
            self.apply_2q_gate_error(self.sim, q_out, q_anc)
        

    def apply_1q_gate_error(self, sim: stim.TableauSimulator, qid: int):
        """Apply a 1-qubit gate error to the qubit at index qid."""
        if self.p1 > 0:
            sim.depolarize1(qid, p=self.p1)

    def apply_2q_gate_error(self, sim: stim.TableauSimulator, qid1: int, qid2: int):
        """Apply a 2-qubit gate error to the qubits at indices qid1 and qid2."""
        if self.p2 > 0:
            sim.depolarize2(qid1, qid2, p=self.p2)

    def apply_measurement_error(self, sim: stim.TableauSimulator, qid: int):
        """Apply a measurement error to the qubit at index qid."""
        if self.pm > 0:
            sim.x_error(qid, p=self.pm)

def eval_teleport_circ(n: int, p1: float, p2: float, pm: float, n_shots: int, verbose: bool = False) -> dict[str, int]:
    ideal_sim = TeleportCircBuilder(n).sim
    ideal_inv_tableau = ideal_sim.current_inverse_tableau()
    error_counts = {}
    for i in tqdm(range(n_shots), desc=f"{n=}, {p2=}", disable=not verbose):
        builder = TeleportCircBuilder(n, p1, p2, pm)
        noisy_sim = builder.sim
        noisy_inv_tableau = noisy_sim.current_inverse_tableau()
        pauli_error = (ideal_inv_tableau.inverse() * noisy_inv_tableau).to_pauli_string()
        remaining_pauli_error = pauli_error[3*n:4*n]

        if remaining_pauli_error != stim.PauliString("I"*n):
            key = str(remaining_pauli_error)
            error_counts[key] = error_counts.get(key, 0) + 1
        else:
            assert remaining_pauli_error == stim.PauliString("I"*n)
    
    return error_counts