

import stim
from tqdm import tqdm


class TelegateCircBuilder:
    def __init__(self, n: int, p1: float = 0.0, p2: float = 0.0, pm: float = 0.0):
        self.n = n
        self.input_qubits = list(range(n))
        self.bell_ancillas = list(range(n, 2 * n))
        self.output_qubits = list(range(2 * n, 3 * n))
        self.sim = stim.TableauSimulator()
        self.measurement_results = []

        self.p1 = p1
        self.p2 = p2
        self.pm = pm

        self._build()

    def _build(self):
        self._prepare_bell_pairs()
        self._telegate_inputs()

    def _prepare_bell_pairs(self):
        for i in range(self.n):
            a = self.bell_ancillas[i]
            b = self.output_qubits[i]
            self.sim.h(a)
            self.apply_1q_gate_error(self.sim, a)
            self.sim.cx(a, b)
            self.apply_2q_gate_error(self.sim, a, b)

    def _telegate_inputs(self):
        for i in range(self.n):
            q_in = self.input_qubits[i]
            q_anc = self.bell_ancillas[i]
            q_out = self.output_qubits[i]

            self.sim.cx(q_in, q_anc)
            self.apply_2q_gate_error(self.sim, q_in, q_anc)
            self.apply_measurement_error(self.sim, q_anc)
            m_anc = self.sim.measure(q_anc)
            self.measurement_results.append(m_anc)

            if m_anc:
                self.sim.x(q_out)
                self.apply_1q_gate_error(self.sim, q_out) 

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

def eval_telegate_circ(n: int, p1: float, p2: float, pm: float, n_shots: int, verbose: bool = False) -> dict[str, int]:
    ideal_sim = TelegateCircBuilder(n).sim
    ideal_inv_tableau = ideal_sim.current_inverse_tableau()
    error_counts = {}
    for i in tqdm(range(n_shots), desc=f"{n=}, {p2=}", disable=not verbose):
        builder = TelegateCircBuilder(n, p1, p2, pm)
        noisy_sim = builder.sim
        noisy_inv_tableau = noisy_sim.current_inverse_tableau()
        pauli_error = (ideal_inv_tableau.inverse() * noisy_inv_tableau).to_pauli_string()
        remaining_pauli_error = pauli_error[2*n:3*n]

        if remaining_pauli_error != stim.PauliString("I"*n):
            key = str(remaining_pauli_error)
            error_counts[key] = error_counts.get(key, 0) + 1
        else:
            assert remaining_pauli_error == stim.PauliString("I"*n)
    
    return error_counts
