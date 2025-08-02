import stim
from tqdm import tqdm

class TeleportedCnotCircBuilder:
    def __init__(self, n: int, p1: float = 0.0, p2: float = 0.0, pm: float = 0.0):
        self.n = n
        self.control_qubits = list(range(n))
        self.control_bell_ancillas = list(range(n, 2 * n))
        self.target_bell_ancillas = list(range(2 * n, 3 * n))
        self.target_qubits = list(range(3 * n, 4 * n))
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
            a = self.control_bell_ancillas[i]
            b = self.target_bell_ancillas[i]
            self.sim.h(a)
            self.apply_1q_gate_error(self.sim, a)
            self.sim.cx(a, b)
            self.apply_2q_gate_error(self.sim, a, b)

    def _telegate_inputs(self):
        for i in range(self.n):
            q_ctrl = self.control_qubits[i]
            q_bell_ctrl = self.control_bell_ancillas[i]
            q_bell_targ = self.target_bell_ancillas[i]
            q_targ = self.target_qubits[i]

            self.sim.cx(q_ctrl, q_bell_ctrl)
            self.apply_2q_gate_error(self.sim, q_ctrl, q_bell_ctrl)
            self.sim.cx(q_bell_targ, q_targ)
            self.apply_2q_gate_error(self.sim, q_bell_targ, q_targ)

            self.sim.h(q_bell_targ)
            self.apply_1q_gate_error(self.sim, q_bell_targ)

            self.apply_measurement_error(self.sim, q_bell_ctrl)
            m_bell_ctrl = self.sim.measure(q_bell_ctrl)
            self.measurement_results.append(m_bell_ctrl)

            if m_bell_ctrl:
                self.sim.x(q_targ)
                self.apply_1q_gate_error(self.sim, q_targ)

            self.apply_measurement_error(self.sim, q_bell_targ)
            m_bell_targ = self.sim.measure(q_bell_targ)
            self.measurement_results.append(m_bell_targ)

            if m_bell_targ:
                self.sim.z(q_ctrl)
                self.apply_1q_gate_error(self.sim, q_ctrl)

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

def eval_teleported_CNOT_circ(n: int, p1: float, p2: float, pm: float, n_shots: int, verbose: bool = False) -> dict[str, int]:
    ideal_sim = TeleportedCnotCircBuilder(n).sim
    ideal_inv_tableau = ideal_sim.current_inverse_tableau()
    error_counts = {}
    for i in tqdm(range(n_shots), desc=f"{n=}, {p2=}", disable=not verbose):
        builder = TeleportedCnotCircBuilder(n, p1, p2, pm)
        noisy_sim = builder.sim
        noisy_inv_tableau = noisy_sim.current_inverse_tableau()
        pauli_error = (ideal_inv_tableau.inverse() * noisy_inv_tableau).to_pauli_string()
        remaining_pauli_error = pauli_error[:n] + pauli_error[-n:]

        if remaining_pauli_error != stim.PauliString("I"*(2*n)):
            key = str(remaining_pauli_error)
            error_counts[key] = error_counts.get(key, 0) + 1
        else:
            assert remaining_pauli_error == stim.PauliString("I"*(2*n))

    return error_counts


