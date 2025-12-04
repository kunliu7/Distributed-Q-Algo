import stim
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr


class BaseFanoutBuilder:
    pass


class BaumerFanoutBuilder(BaseFanoutBuilder):
    """Reproduce the Fanout circuit from Baumer et al.
    https://arxiv.org/abs/2308.13065

    Two ways to build the circuit: qiskit and stim.
    """
    def __init__(self, n_trgts: int, ctrl_bit: int, init_trgt_bits: list[int],
                 p1: float = 0.0, p2: float = 0.0, pm: float = 0.0):
        self.n_trgts = n_trgts
        # rightmost bit is the control qubit/0-th qubit
        input_bits = [reg for pair in zip(init_trgt_bits, [0] * n_trgts) for reg in pair] + [ctrl_bit]
        self.init_bitstr = "".join(map(str, input_bits))
        self.p1 = p1
        self.p2 = p2
        self.pm = pm

    def build_w_more_cnots(self) -> QuantumCircuit:
        """Fig. 11, last figure, Baumer et al."""
        n_ancs = self.n_trgts
        ctrl = QuantumRegister(1, 'c')
        ancs = [QuantumRegister(1, f'a{i}') for i in range(n_ancs)]
        tgts = [QuantumRegister(1, f't{i}') for i in range(self.n_trgts)]
        anc_cregs = ClassicalRegister(self.n_trgts, f'anc_cregs')
        tgt_cregs = ClassicalRegister(self.n_trgts, f'tgt_cregs')
        ctrl_creg = ClassicalRegister(1, f'ctrl_creg')
        all_regs = [reg for pair in zip(ancs, tgts) for reg in pair]
        # cregs = ClassicalRegister(2*n_tgts + 1, f'cregs')
        qc = QuantumCircuit(ctrl, *all_regs, ctrl_creg, anc_cregs, tgt_cregs)
        qc.initialize(self.init_bitstr)

        for i in range(0, n_ancs, 2):
            # step 1
            qc.h(ancs[i])
            qc.cx(tgts[i], ancs[i+1])

            # step 2
            qc.cx(ancs[i], tgts[i])

        # step 3
        for i in range(-1, self.n_trgts-1):
            if i == -1:
                qc.cx(ctrl, ancs[0])
            else:
                qc.cx(tgts[i], ancs[i+1])

        for i in range(1, self.n_trgts, 2):
            # step 4:
            qc.cx(ancs[i], tgts[i])

            # step 5
            qc.h(ancs[i])

        # step 5':
        for i in range(1, self.n_trgts-1, 2):
            qc.cx(tgts[i], ancs[i+1])

        # return qc
        # step 6
        for anc, creg in zip(ancs, anc_cregs):
            qc.measure(anc, creg)

        # step 7-1
        parity_for_Z_on_ctrl = expr.lift(anc_cregs[i])
        for i in range(3, n_ancs, 2):
            parity_for_Z_on_ctrl = expr.bit_xor(anc_cregs[i], parity_for_Z_on_ctrl)

        with qc.if_test(parity_for_Z_on_ctrl):  # type: ignore
            qc.z(ctrl)

        # step 7-2
        exprs_X_on_tgts = []
        for i in range(0, n_ancs, 2):
            if i == 0:
                e = expr.lift(anc_cregs[i])
            else:
                e = expr.bit_xor(anc_cregs[i], exprs_X_on_tgts[-1])
            exprs_X_on_tgts.append(e)

            with qc.if_test(e):  # type: ignore
                qc.x(tgts[i])
                qc.x(tgts[i+1])

        qc.measure(ctrl, ctrl_creg)

        for tgt, creg in zip(tgts, tgt_cregs):
            qc.measure(tgt, creg)

        return qc

    def build_w_fewer_cnots(self) -> QuantumCircuit:
        """Fig. 11, before the last figure, Baumer et al."""
        n_ancs = self.n_trgts
        ctrl = QuantumRegister(1, 'c')
        ancs = [QuantumRegister(1, f'a{i}') for i in range(n_ancs)]
        tgts = [QuantumRegister(1, f't{i}') for i in range(self.n_trgts)]
        anc_cregs = ClassicalRegister(self.n_trgts, f'anc_cregs')
        tgt_cregs = ClassicalRegister(self.n_trgts, f'tgt_cregs')
        ctrl_creg = ClassicalRegister(1, f'ctrl_creg')
        all_regs = [reg for pair in zip(ancs, tgts) for reg in pair]
        # cregs = ClassicalRegister(2*n_tgts + 1, f'cregs')
        qc = QuantumCircuit(ctrl, *all_regs, ctrl_creg, anc_cregs, tgt_cregs)
        qc.initialize(self.init_bitstr)

        # step 1, entangle every two ancillae
        for i in range(0, n_ancs, 2):
            qc.h(ancs[i])
            if i < n_ancs - 1:
                qc.cx(ancs[i], ancs[i+1])

        # step 2, entangle every ancilla with its target
        for i in range(self.n_trgts):
            qc.cx(ancs[i], tgts[i])

        # step 3, entangle every pairs of Bell pairs
        for i in range(1, self.n_trgts - 1, 2):
            qc.cx(ancs[i], ancs[i+1])

        # step 4, cnot from ctrl to first ancilla
        qc.cx(ctrl, ancs[0])

        # step 5, measure all ancillae
        for i in range(n_ancs):
            if i % 2 == 1:
                qc.h(ancs[i])
            qc.measure(ancs[i], anc_cregs[i])

        # step 6, measure all qubits
        for anc, creg in zip(ancs, anc_cregs):
            qc.measure(anc, creg)

        # step 7-1
        parity_for_Z_on_ctrl = expr.lift(anc_cregs[1])
        for i in range(3, n_ancs, 2):
            parity_for_Z_on_ctrl = expr.bit_xor(anc_cregs[i], parity_for_Z_on_ctrl)

        with qc.if_test(parity_for_Z_on_ctrl):  # type: ignore
            qc.z(ctrl)

        # step 7-2
        exprs_X_on_tgts = []
        for i in range(0, n_ancs, 2):
            if i == 0:
                e = expr.lift(anc_cregs[i])
            else:
                e = expr.bit_xor(anc_cregs[i], exprs_X_on_tgts[-1])
            exprs_X_on_tgts.append(e)

            with qc.if_test(e):  # type: ignore
                qc.x(tgts[i])
                if i < n_ancs - 1:
                    qc.x(tgts[i+1])

        qc.measure(ctrl, ctrl_creg)

        for tgt, creg in zip(tgts, tgt_cregs):
            qc.measure(tgt, creg)

        return qc

    def build_in_stim(self):
        """
        Runs a Stim tableau simulation that mimics the Qiskit circuit from Baumer et al.

        The mapping is as follows (for n_trgts targets):
        - Control qubit: index 0.
        - For each target i (0-based):
            * Ancilla qubit: index 1 + 2*i.
            * Target qubit:  index 2 + 2*i.

        :param n_trgts: Number of targets (and ancillas).
        :param init_bitstr: Optional list/tuple of length num_qubits providing the initial state
                            as bits (0 or 1). Default is all-zeros.
        :return: A dictionary with final measurement outcomes.
        """
        num_qubits = 1 + 2 * self.n_trgts
        sim = stim.TableauSimulator()

        # Initialize the qubits.
        # Stim’s TableauSimulator always starts in |0...0>.
        # If a different initial state is desired (via qc.initialize in Qiskit),
        # you can apply X gates to the qubits that should start in |1>.
        if self.init_bitstr is not None:
            # reverse the bitstring, since the least significant bit is at the end
            for q, bit in enumerate(self.init_bitstr[::-1]):
                if bit == "1":
                    sim.x(q)

        # --- Step 1: Entangle every two ancillae ---
        # For i in range(0, n_trgts, 2):
        #    apply H to ancilla a_i; if there is a subsequent ancilla, apply CNOT from a_i to a_{i+1}.
        for i in range(0, self.n_trgts, 2):
            anc_i = 1 + 2 * i  # index for ancilla i
            sim.h(anc_i)
            self.apply_1q_gate_error(sim, anc_i)
            if i < self.n_trgts - 1:
                anc_next = 1 + 2 * (i + 1)
                sim.cx(anc_i, anc_next)
                self.apply_2q_gate_error(sim, anc_i, anc_next)

        # --- Step 2: Entangle every ancilla with its target ---
        # For i in range(n_trgts): apply CNOT from ancilla a_i to target t_i.
        for i in range(self.n_trgts):
            anc_i = 1 + 2 * i
            tgt_i = 2 + 2 * i
            sim.cx(anc_i, tgt_i)
            self.apply_2q_gate_error(sim, anc_i, tgt_i)

        # --- Step 3: Entangle pairs of Bell pairs ---
        # For i in range(1, n_trgts - 1, 2): apply CNOT from ancilla a_i to ancilla a_{i+1}.
        for i in range(1, self.n_trgts - 1, 2):
            anc_i = 1 + 2 * i
            anc_next = 1 + 2 * (i + 1)
            sim.cx(anc_i, anc_next)
            self.apply_2q_gate_error(sim, anc_i, anc_next)

        # --- Step 4: CNOT from control to first ancilla ---
        sim.cx(0, 1)  # control (index 0) to ancilla a_0 (index 1)
        self.apply_2q_gate_error(sim, 0, 1)

        # --- Step 5: Measure all ancillae (with a Hadamard on every odd-indexed ancilla) ---
        # In Qiskit: if i is odd, apply H then measure.
        ancilla_meas = {}
        for i in range(self.n_trgts):
            anc_i = 1 + 2 * i
            if i % 2 == 1:
                sim.h(anc_i)
                self.apply_1q_gate_error(sim, anc_i)
            self.apply_measurement_error(sim, anc_i)
            m = sim.measure(anc_i)  # measurement returns 0 or 1
            ancilla_meas[i] = m  # store measurement for ancilla i

        # (The Qiskit circuit shows a second measurement of the ancilla registers,
        # but here we assume one measurement per ancilla suffices.)

        # --- Step 7-1: Conditional Z on control based on ancilla parity ---
        # In Qiskit, parity_for_Z_on_ctrl is the XOR (parity) of ancilla measurements for odd-indexed ancillas.
        parity_for_Z_on_ctrl = 0
        for i in range(1, self.n_trgts, 2):  # consider ancillas with odd index: 1, 3, 5, ...
            parity_for_Z_on_ctrl ^= ancilla_meas[i]

        if parity_for_Z_on_ctrl == 1:
            sim.z(0)  # apply Z on control (qubit 0) if parity is 1
            self.apply_1q_gate_error(sim, 0)
        # --- Step 7-2: Conditional X on targets based on cumulative XOR of even-indexed ancilla measurements ---
        # For even-indexed ancillas, compute a cumulative XOR and use it as the condition.
        cumulative = 0
        for i in range(0, self.n_trgts, 2):
            anc_val = ancilla_meas[i]
            if i == 0:
                condition = anc_val
            else:
                condition = cumulative ^ anc_val
            cumulative = condition  # update cumulative XOR
            if condition == 1:
                tgt_i = 2 + 2 * i
                sim.x(tgt_i)
                self.apply_1q_gate_error(sim, tgt_i)
                if i < self.n_trgts - 1:
                    # Also apply X to the next target (corresponding to the next ancilla pair)
                    tgt_next = 2 + 2 * (i + 1)
                    sim.x(tgt_next)
                    self.apply_1q_gate_error(sim, tgt_next)

        # --- Final Measurements ---
        ctrl_meas = int(sim.measure(0))
        target_meas = [int(sim.measure(2 + 2 * i)) for i in range(self.n_trgts)]

        # Return the measurement outcomes.
        return sim, {
            # "ancilla_measurements": ancilla_meas,
            "control_measurement": ctrl_meas,
            "target_measurements": target_meas,
        }

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
