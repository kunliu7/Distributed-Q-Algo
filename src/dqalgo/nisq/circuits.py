from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.library import UnitaryGate
import numpy as np
from dqalgo.nisq.utils import add_fanout_monte_carlo_error, add_fanout_custom_error_injection


def apply_GHZ_prep_circ_w_reset(
        circ: QuantumCircuit,
        qr: list[QuantumRegister],
        anc_cr: list[ClassicalRegister],
        n: int, meas_all: bool = False):
    """Ref: http://arxiv.org/abs/2206.15405, Fig. 1
    """
    assert n % 2 == 0
    half_n = n // 2

    for m in range(half_n):
        circ.h(qr[2 * m])

    for m in range(half_n):
        circ.cx(qr[2 * m], qr[2 * m + 1])

    for m in range(half_n - 1):
        circ.cx(qr[2 * m + 1], qr[2 * m + 2])

    for m in range(half_n - 1):
        circ.measure(qr[2 * m + 2], anc_cr[2 * m + 2])

    for m in range(half_n - 1):
        measured_idx = 2 * m + 2
        _condition = expr.lift(anc_cr[measured_idx]) if m == 0 else expr.bit_xor(_condition, anc_cr[measured_idx])
        with circ.if_test(expr.equal(_condition, 1)):  # type: ignore
            circ.x(qr[2 * m + 3])

    circ.barrier()

    for i in range(half_n - 1):
        circ.reset(qr[2 * i + 2])
        circ.cx(qr[2 * i + 1], qr[2 * i + 2])

    circ.barrier()

    if meas_all:
        for i in range(n):
            circ.measure(qr[i], anc_cr[i])


def get_Fanout_circ_by_GHZ_w_reset(n_tgts: int, init_bitstr: str | None = None, meas_all: bool = False) -> QuantumCircuit:
    """Ref: http://arxiv.org/abs/2403.18768, Fig. 3(a)"""
    # c, a1, t1, a2, t2, ...
    # 0,  1,  2,  3,  4, ...
    n_GHZ = n_tgts + 1
    ancs = [QuantumRegister(1, f'a{i}') for i in range(n_GHZ)]
    qubits = [QuantumRegister(1, f't{i}') for i in range(n_GHZ)]
    anc_cregs = [ClassicalRegister(1, f'anc_cregs{i}') for i in range(2*n_GHZ)]
    qubit_cregs = ClassicalRegister(n_GHZ, f'data_cregs')
    ctrl = qubits[0]
    all_regs = [reg for pair in zip(qubits, ancs) for reg in pair]

    qc = QuantumCircuit(*all_regs, *anc_cregs, qubit_cregs)
    if init_bitstr is not None:
        qc.initialize(init_bitstr)

    # suggested by ChatGPT, use existing qreg and creg to apply circuit instead of composing new circuit
    # to avoid conflicts led by dynamic circuit
    apply_GHZ_prep_circ_w_reset(qc, ancs, anc_cregs[:n_GHZ], n_GHZ)

    qc.cx(ctrl, ancs[0])
    qc.measure(ancs[0], anc_cregs[n_GHZ+0])
    for i in range(n_tgts):
        qc.cx(ancs[i+1], qubits[i+1])
        qc.h(ancs[i+1])
        qc.measure(ancs[i+1], anc_cregs[n_GHZ+i+1])

    qc.barrier()

    for i in range(n_tgts):
        with qc.if_test(expr.equal(anc_cregs[n_GHZ+0], 1)):  # type: ignore
            qc.x(qubits[i+1])

    parity_for_Z_on_ctrl = expr.lift(anc_cregs[n_GHZ+1])
    for i in range(2, n_GHZ):
        parity_for_Z_on_ctrl = expr.bit_xor(anc_cregs[n_GHZ+i], parity_for_Z_on_ctrl)

    with qc.if_test(expr.equal(parity_for_Z_on_ctrl, 1)):  # type: ignore
        qc.z(ctrl)

    if meas_all:
        for i in range(n_GHZ):
            qc.measure(qubits[i], qubit_cregs[i])
    return qc

def get_fanout_gate_by_custom_unitary(n_tgts: int) -> UnitaryGate:
    """
    Creates fanout gate circuit from unitary matrix to be used as a blackbox
    """
    n = n_tgts + 1  # total qubits
    dim = 2**n
    U = np.zeros((dim, dim))

    # Create the unitary matrix for the fanout operation
    for i in range(dim):
        bits = [(i >> k) & 1 for k in range(n)]  # little-endian
        ctrl = bits[0]  # qubit 0 is control
        new_bits = bits.copy()
        if ctrl:
            for t in range(1, n):  # flip all target bits
                new_bits[t] ^= 1
        j = sum(b << k for k, b in enumerate(new_bits))
        U[j, i] = 1

    return UnitaryGate(U, label=f"{n_tgts}-fanout")

def get_parallel_toffoli_via_fanout_circ(
        n_trgts: int,
        input_bitstr: str | None = None,
        meas_all: bool = False,
        n_fanout_errors: tuple[str, float] | None = None,
        two_n_fanout_errors: tuple[str, float] | None = None,
    ) -> QuantumCircuit:
    """
    Red circuit from Fig. 5(e) of Distributed Quantum Signal Processing
    """
    n_toffoli = 2*n_trgts + 1

    n_fanout = get_fanout_gate_by_custom_unitary(n_trgts)
    two_n_fanout = get_fanout_gate_by_custom_unitary(2*n_trgts)

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_toffoli)]
    qubit_cregs = ClassicalRegister(n_toffoli, f'data_cregs')

    qc = QuantumCircuit(*qubits, qubit_cregs)

    ctrl1_idx = [qc.qubits[0]]
    ctrl2_idx = qc.qubits[1:n_trgts+1]
    targ_idx = qc.qubits[n_trgts+1:]

    if input_bitstr is not None:
        for i, val in enumerate(input_bitstr[::-1]):
            if val == '1':
                qc.x(i)

    # tdg^n is periodic mod 8 so no need to aply more than 2 gates
    if n_trgts % 8 == 1:
        qc.tdg(ctrl1_idx)
    elif n_trgts % 8 == 2:
        qc.sdg(ctrl1_idx)
    elif n_trgts % 8 == 3:
        qc.tdg(ctrl1_idx)
        qc.sdg(ctrl1_idx)
    elif n_trgts % 8 == 4:
        qc.z(ctrl1_idx)
    elif n_trgts % 8 == 5:
        qc.t(ctrl1_idx)
        qc.s(ctrl1_idx)
    elif n_trgts % 8 == 6:
        qc.s(ctrl1_idx)
    else:
        qc.t(ctrl1_idx)

    qc.tdg(ctrl2_idx)
    qc.h(targ_idx)
    qc.cx(targ_idx, ctrl2_idx)

    qc.append(n_fanout, ctrl1_idx + targ_idx)
    if n_fanout_errors is not None:
        add_fanout_custom_error_injection(qc, ctrl1_idx + targ_idx, n_fanout_errors)

    qc.t(ctrl2_idx + targ_idx)
    qc.append(two_n_fanout, qubits)
    if two_n_fanout_errors is not None:
        add_fanout_custom_error_injection(qc, qubits, two_n_fanout_errors)

    qc.tdg(ctrl2_idx)
    qc.cx(targ_idx, ctrl2_idx)
    qc.t(ctrl2_idx)
    qc.tdg(targ_idx)

    qc.append(n_fanout, ctrl1_idx + ctrl2_idx)
    if n_fanout_errors is not None:
        add_fanout_custom_error_injection(qc, ctrl1_idx + ctrl2_idx, n_fanout_errors)

    qc.h(targ_idx)

    if meas_all:
        for i in range(n_toffoli):
            qc.measure(qubits[i], qubit_cregs[i])

    return qc