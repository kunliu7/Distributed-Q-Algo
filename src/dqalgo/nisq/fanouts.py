from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr


class BaseFanoutBuilder:
    pass


class BaumerFanoutBuilder(BaseFanoutBuilder):
    def __init__(self, n_trgts: int, ctrl_bit: int, init_trgt_bits: list[int]):
        self.n_trgts = n_trgts
        # rightmost bit is the control qubit/0-th qubit
        input_bits = [reg for pair in zip(init_trgt_bits, [0] * n_trgts) for reg in pair] + [ctrl_bit]
        self.init_bitstr = "".join(map(str, input_bits))

    def build(self) -> QuantumCircuit:
        assert self.n_trgts % 2 == 0
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

        qc.measure(ctrl, ctrl_creg)
        for i in range(self.n_trgts):
            qc.measure(tgts[i], tgt_cregs[i])
            qc.measure(ancs[i], anc_cregs[i])    # return qc

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
