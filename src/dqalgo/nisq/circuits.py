from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.classical import expr
from qiskit.circuit.library import UnitaryGate
import numpy as np
from dqalgo.nisq.utils import add_custom_error_injection


def apply_GHZ_prep_circ_w_reset(
        circ: QuantumCircuit,
        qr: list[QuantumRegister],
        anc_cr: list[ClassicalRegister],
        n: int,
        meas_all: bool = False,
        telecnot_errors: list[tuple[str, float]] | None = None):
    """Ref: http://arxiv.org/abs/2206.15405, Fig. 1
    """
    assert n % 2 == 0
    half_n = n // 2

    for m in range(half_n):
        circ.h(qr[2 * m])

    for m in range(half_n):
        circ.cx(qr[2 * m], qr[2 * m + 1])
        if telecnot_errors is not None:
            add_custom_error_injection(circ, [qr[2 * m], qr[2 * m + 1]], telecnot_errors)


    for m in range(half_n - 1):
        circ.cx(qr[2 * m + 1], qr[2 * m + 2])
        if telecnot_errors is not None:
            add_custom_error_injection(circ, [qr[2 * m + 1], qr[2 * m + 2]], telecnot_errors)


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
        if telecnot_errors is not None:
            add_custom_error_injection(circ, [qr[2 * i + 1], qr[2 * i + 2]], telecnot_errors)

    circ.barrier()

    if meas_all:
        for i in range(n):
            circ.measure(qr[i], anc_cr[i])


def get_ideal_GHZ_prep_circ(n_parties: int) -> QuantumCircuit:
    circ = QuantumCircuit(n_parties)

    circ.h(0)
    for i in range(n_parties - 1):
        circ.cx(i, i + 1)

    return circ

def get_distributed_GHZ_prep_circ(
        n_parties: int,
        meas_all: bool = False,
        telecnot_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ) -> QuantumCircuit:

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_parties)]
    ancilla_cregs =  [ClassicalRegister(1, f'anc_cregs{i}') for i in range(n_parties)]

    qc = QuantumCircuit(*qubits, *ancilla_cregs)

    apply_GHZ_prep_circ_w_reset(qc, qubits, ancilla_cregs, n_parties, meas_all=meas_all, telecnot_errors=telecnot_errors)

    return qc



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

# helpers for CSWAP
def apply_parallel_toffoli_via_fanout(
        qc: QuantumCircuit,
        ctrl1_reg: QuantumRegister,
        ctrl2_regs: list[QuantumRegister],
        targ_regs: list[QuantumRegister],
        n_fanout_errors: list[tuple[str, float]] | None = None,
    ) -> None:
    """
    Red circuit from Fig. 6(e) of COMPAS
    """
    n_trgts = len(targ_regs)

    ctrl1_regs = [ctrl1_reg]
    qubits = ctrl1_regs + ctrl2_regs + targ_regs

    n_fanout = get_fanout_gate_by_custom_unitary(n_trgts)

    # tdg^n is periodic mod 8 so no need to aply more than 2 gates
    if n_trgts % 8 == 1:
        qc.tdg(ctrl1_regs)
    elif n_trgts % 8 == 2:
        qc.sdg(ctrl1_regs)
    elif n_trgts % 8 == 3:
        qc.tdg(ctrl1_regs)
        qc.sdg(ctrl1_regs)
    elif n_trgts % 8 == 4:
        qc.z(ctrl1_regs)
    elif n_trgts % 8 == 5:
        qc.t(ctrl1_regs)
        qc.s(ctrl1_regs)
    elif n_trgts % 8 == 6:
        qc.s(ctrl1_regs)
    else:
        qc.t(ctrl1_regs)

    qc.tdg(ctrl2_regs)
    qc.h(targ_regs)
    qc.cx(targ_regs, ctrl2_regs)

    qc.append(n_fanout, ctrl1_regs + targ_regs)
    if n_fanout_errors is not None:
        add_custom_error_injection(qc, ctrl1_regs + targ_regs, n_fanout_errors)

    qc.t(ctrl2_regs)

    qc.append(n_fanout, ctrl1_regs + ctrl2_regs)
    if n_fanout_errors is not None:
        add_custom_error_injection(qc, ctrl1_regs + ctrl2_regs, n_fanout_errors)

    qc.t(targ_regs)

    qc.append(n_fanout, ctrl1_regs + targ_regs)
    if n_fanout_errors is not None:
        add_custom_error_injection(qc, ctrl1_regs + targ_regs, n_fanout_errors)

    qc.tdg(ctrl2_regs)
    qc.cx(targ_regs, ctrl2_regs)
    qc.t(ctrl2_regs)
    qc.tdg(targ_regs)

    qc.append(n_fanout, ctrl1_regs + ctrl2_regs)
    if n_fanout_errors is not None:
        add_custom_error_injection(qc, ctrl1_regs + ctrl2_regs, n_fanout_errors)

    qc.h(targ_regs)

def get_parallel_toffoli_via_fanout_circ(
        input_bitstr: str,
        meas_all: bool = False,
        n_fanout_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ):

    n_qubits = len(input_bitstr)
    n_trgts = (n_qubits - 1)//2

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_qubits)]
    qubit_cregs = ClassicalRegister(n_qubits, f'data_cregs')
    qc = QuantumCircuit(*qubits, qubit_cregs)

    for i, val in enumerate(input_bitstr[::-1]):
        if val == '1':
            qc.x(i)

    ctrl1_reg = qc.qubits[0]
    ctrl2_regs = qc.qubits[1:n_trgts+1]
    targ_regs = qc.qubits[n_trgts+1:]

    apply_parallel_toffoli_via_fanout(
        qc=qc,
        ctrl1_reg=ctrl1_reg,
        ctrl2_regs=ctrl2_regs,
        targ_regs=targ_regs,
        n_fanout_errors=n_fanout_errors
    )

    if meas_all:
        for i in range(n_qubits):
            qc.measure(qubits[i], qubit_cregs[i])

    return qc

def apply_teleported_toffoli(
        qc: QuantumCircuit,
        ctrl1_reg: QuantumRegister,
        ctrl2_regs: list[QuantumRegister],
        targ_regs: list[QuantumRegister],
        party_A_bell_regs: list[QuantumRegister],
        party_B_bell_regs: list[QuantumRegister],
        party_A_bell_cregs: list[ClassicalRegister],
        party_B_bell_cregs: list[ClassicalRegister],
        n_fanout_errors: list[tuple[str, float]] | None = None,
    ) -> None:
        """
        Orange circuit from Fig. 6(d) of COMPAS.
        Control 1 and target are shared by part A, Control 2 is located on party B.

        ctrl1_reg: |phi> in the diagram (located in QPU controlled by party A)
        ctrl2_regs: rho_j in the diagram (located in QPU controlled by party B)
        targ_regs: rho_i in the diagram (located in QPU controlled by party A)
        party_A_bell_regs: party A's qubit of the shared bell pair, |Phi^+>
        party_B_bell_regs: party B's qubit of the shared bell pair, |Phi^+>
        party_A_bell_cregs: classical registers for measuring party A's bell pair qubit
        party_B_bell_cregs: classical registers for measuring party B's bell pair qubit
        """

        qc.cx(ctrl2_regs, party_B_bell_regs)

        for i in range(len(targ_regs)):
            qc.measure(party_B_bell_regs[i], party_B_bell_cregs[i])
            with qc.if_test((party_B_bell_cregs[i], 1)):
                qc.x(party_A_bell_regs[i])

        apply_parallel_toffoli_via_fanout(
            qc=qc,
            ctrl1_reg=ctrl1_reg,
            ctrl2_regs=party_A_bell_regs,
            targ_regs=targ_regs,
            n_fanout_errors=n_fanout_errors,
        )


        for i in range(len(targ_regs)):
            qc.measure(party_A_bell_regs[i], party_A_bell_cregs[i])
            with qc.if_test((party_A_bell_cregs[i], 1)):
                qc.z(ctrl2_regs[i])

def get_teleported_toffoli_circ(
        input_bitstr: str,
        meas_all: bool = False,
        n_fanout_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ):

    n_qubits = len(input_bitstr)
    n_trgts = (n_qubits - 1)//2

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_qubits)]
    bell_pair_ancilla_regs = [QuantumRegister(1, f'a{i}') for i in range(2*n_trgts)]
    qubit_cregs = ClassicalRegister(n_qubits, f'data_cregs')
    anc_cregs = [ClassicalRegister(1, f'anc_cregs{i}') for i in range(2*n_trgts)]

    all_regs = qubits + bell_pair_ancilla_regs
    qc = QuantumCircuit(*all_regs, *anc_cregs, qubit_cregs)

    ctrl1_reg = qc.qubits[0]
    ctrl2_regs = qc.qubits[1:n_trgts+1]
    targ_regs = qc.qubits[n_trgts+1:n_qubits]
    party_A_bell_regs = qc.qubits[n_qubits:n_qubits+n_trgts]
    party_B_bell_regs = qc.qubits[n_qubits+n_trgts:n_qubits+2*n_trgts]

    party_A_bell_cregs = qc.clbits[:n_trgts]
    party_B_bell_cregs = qc.clbits[n_trgts:2*n_trgts]

    # prepare initial state for data qubits
    for i, val in enumerate(input_bitstr[::-1]):
        if val == '1':
            qc.x(i)

    # prepare bell pairs
    qc.h(party_A_bell_regs)
    qc.cx(party_A_bell_regs, party_B_bell_regs)

    apply_teleported_toffoli(
        qc=qc,
        ctrl1_reg=ctrl1_reg,
        ctrl2_regs=ctrl2_regs,
        targ_regs=targ_regs,
        party_A_bell_regs=party_A_bell_regs,
        party_B_bell_regs=party_B_bell_regs,
        party_A_bell_cregs=party_A_bell_cregs,
        party_B_bell_cregs=party_B_bell_cregs,
        n_fanout_errors=n_fanout_errors
    )

    if meas_all:
        for i in range(n_qubits):
            qc.measure(qubits[i], qubit_cregs[i])

    return qc

def apply_teleported_cnot(
        qc: QuantumCircuit,
        ctrl_regs: list[QuantumRegister],
        targ_regs: list[QuantumRegister],
        party_A_bell_regs: list[QuantumRegister],
        party_B_bell_regs: list[QuantumRegister],
        party_A_bell_cregs: list[ClassicalRegister],
        party_B_bell_cregs: list[ClassicalRegister]
    ) -> None:
    """
    Teleported CNOT circuit from Fig. 1(b) of COMPAS. Applies
    a CNOT gate from ctrl_regs to targ_regs, measuring party_A_bell_regs and party_B_bell_regs.

    ctrl_regs: |phi> in the diagram (located in QPU controlled by party A)
    targ_regs: |psi> in the diagram (located in QPU controlled by party B)
    party_A_bell_regs: party A's qubit of the shared bell pair, |Phi^+>
    party_B_bell_regs: party B's qubit of the shared bell pair, |Phi^+>
    party_A_bell_cregs: classical registers for measuring party A's bell pair qubit
    party_B_bell_cregs: classical registers for measuring party B's bell pair qubit
    """

    qc.cx(ctrl_regs, party_A_bell_regs)
    qc.cx(party_B_bell_regs, targ_regs)
    qc.h(party_B_bell_regs)

    for i in range(len(targ_regs)):
        qc.measure(party_A_bell_regs[i], party_A_bell_cregs[i])
        with qc.if_test((party_A_bell_cregs[i], 1)):
            qc.x(targ_regs[i])

        qc.measure(party_B_bell_regs[i], party_B_bell_cregs[i])
        with qc.if_test((party_B_bell_cregs[i], 1)):
            qc.z(ctrl_regs)

def apply_state_teleportation(
        qc: QuantumCircuit,
        state_to_teleport: list[QuantumRegister],
        party_A_bell_regs: list[QuantumRegister],
        party_B_bell_regs: list[QuantumRegister],
        state_to_teleport_cregs: list[ClassicalRegister],
        bell_cregs: list[ClassicalRegister]
    ) -> None:
    """
    State teleportation circuit from Fig. 1(a) of COMPAS. Teleports state_to_teleport
    on part A to party_B_bell_regs on party B, measuring state_to_teleport and party_A_bell_regs.

    state_to_teleport: |psi> in the diagram (located in QPU controlled by party A)
    target_ancillas: |0> in the diagram (located in QPU controlled by party B)
    party_A_bell_regs: party A's qubit of the shared bell pair, |Phi^+>
    party_B_bell_regs: party B's qubit of the shared bell pair, |Phi^+>
    party_A_bell_cregs: classical registers for measuring party A's bell pair qubit
    party_B_bell_cregs: classical registers for measuring party B's bell pair qubit
    """

    qc.cx(state_to_teleport, party_A_bell_regs)
    qc.h(state_to_teleport)

    for i in range(len(state_to_teleport)):
        qc.measure(state_to_teleport[i], state_to_teleport_cregs[i])

        qc.measure(party_A_bell_regs[i], bell_cregs[i])
        with qc.if_test((bell_cregs[i], 1)):
            qc.x(party_B_bell_regs[i])

        with qc.if_test((state_to_teleport_cregs[i], 1)):
            qc.z(party_B_bell_regs[i])


# CSWAP circuits
def apply_CSWAP_teledata(
        qc: QuantumCircuit,
        ctrl_reg: QuantumRegister,
        state_A: list[QuantumRegister],
        state_B: list[QuantumRegister],
        party_A_bell_regs: list[QuantumRegister],
        party_B_bell_regs: list[QuantumRegister],
        state_to_teleport_cregs: list[ClassicalRegister],
        bell_cregs: list[ClassicalRegister],
        n_fanout_errors: list[tuple[str, float]] | None = None,
    ) -> None:
        """
        CSWAP circuit from Fig. 6(c) of COMPAS. Applies
        a CSWAP gate from ctrl_regs to state_A and state_B, measuring party_A_ancilla_regs,
        party_A_bell_regs and party_B_bell_regs.

        ctrl_reg: |phi> in the diagram (located in QPU controlled by party A)
        state_A: rho_i in the diagram (located in QPU controlled by party A)
        state_B: rho_j in the diagram (located in QPU controlled by party B)
        party_A_ancilla_regs: ancilla qubits in the diagram (located in QPU controlled by party A)
        party_A_bell_regs: party A's qubit of the shared bell pair, |Phi^+>
        party_B_bell_regs: party B's qubit of the shared bell pair, |Phi^+>
        party_A_bell_cregs: classical registers for measuring party A's bell pair qubit
        party_B_bell_cregs: classical registers for measuring party B's bell pair qubit
        """

        # first teleport state_B to party_A_ancilla_regs (note that in this direction, A and B are swapped)
        apply_state_teleportation(
            qc=qc,
            state_to_teleport=state_B,
            party_A_bell_regs=party_B_bell_regs,
            party_B_bell_regs=party_A_bell_regs,
            state_to_teleport_cregs=state_to_teleport_cregs,
            bell_cregs=bell_cregs
        )

        # apply an local controlled swap of ctrl_reg on state_A and party_A_ancilla_regs
        qc.cx(party_A_bell_regs, state_A)
        apply_parallel_toffoli_via_fanout(
            qc=qc,
            ctrl1_reg=ctrl_reg,
            ctrl2_regs=state_A,
            targ_regs=party_A_bell_regs,
            n_fanout_errors=n_fanout_errors,
        )
        qc.cx(party_A_bell_regs, state_A)


        # for the purposes of simulation we reuse the old register
        party_A_new_bell_regs = party_B_bell_regs
        party_B_new_bell_regs = state_B

        # reset and reshare bell pairs
        qc.reset(party_A_new_bell_regs + party_B_new_bell_regs)
        qc.h(party_A_new_bell_regs)
        qc.cx(party_A_new_bell_regs, party_B_new_bell_regs)


        # teleport the swapped state still in party_A_ancilla_regs back to state_B
        apply_state_teleportation(
            qc=qc,
            state_to_teleport=party_A_bell_regs,
            party_A_bell_regs=party_A_new_bell_regs,
            party_B_bell_regs=party_B_new_bell_regs,
            state_to_teleport_cregs=state_to_teleport_cregs,
            bell_cregs=bell_cregs
        )

def get_CSWAP_teledata_circ(
        input_bitstr: str,
        meas_all: bool = False,
        n_fanout_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ):
    n_qubits = len(input_bitstr)
    state_size = (n_qubits - 1)//2

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_qubits)]
    bell_pair_ancilla_regs = [QuantumRegister(1, f'a{i}') for i in range(2*state_size)]
    qubit_cregs = ClassicalRegister(n_qubits, f'data_cregs')
    anc_cregs = [ClassicalRegister(1, f'anc_cregs{i}') for i in range(2*state_size)]

    all_regs = qubits + bell_pair_ancilla_regs
    qc = QuantumCircuit(*all_regs, *anc_cregs, qubit_cregs)

    ctrl_reg = qc.qubits[0]
    state_A_regs = qc.qubits[1:state_size+1]
    state_B_regs = qc.qubits[state_size+1:n_qubits]
    party_A_bell_regs = qc.qubits[n_qubits:n_qubits+state_size]
    party_B_bell_regs = qc.qubits[n_qubits+state_size:n_qubits+2*state_size]

    state_to_teleport_cregs = qc.clbits[:state_size]
    bell_cregs = qc.clbits[state_size:2*state_size]

    # prepare initial state for data qubits
    for i, val in enumerate(input_bitstr[::-1]):
        if val == '1':
            qc.x(i)

    # prepare bell pairs
    qc.h(party_A_bell_regs)
    qc.cx(party_A_bell_regs, party_B_bell_regs)

    apply_CSWAP_teledata(
        qc=qc,
        ctrl_reg=ctrl_reg,
        state_A=state_A_regs,
        state_B=state_B_regs,
        party_A_bell_regs=party_A_bell_regs,
        party_B_bell_regs=party_B_bell_regs,
        state_to_teleport_cregs=state_to_teleport_cregs,
        bell_cregs=bell_cregs,
        n_fanout_errors=n_fanout_errors,
    )

    if meas_all:
        for i in range(n_qubits):
            qc.measure(qubits[i], qubit_cregs[i])

    return qc

def apply_CSWAP_telegate(
        qc: QuantumCircuit,
        ctrl_reg: QuantumRegister,
        state_A: list[QuantumRegister],
        state_B: list[QuantumRegister],
        party_A_bell_regs: list[QuantumRegister],
        party_B_bell_regs: list[QuantumRegister],
        party_A_bell_cregs: list[ClassicalRegister],
        party_B_bell_cregs: list[ClassicalRegister],
        n_fanout_errors: list[tuple[str, float]] | None = None,
    ) -> None:
        """
        CSWAP circuit from Fig. 6(b) of COMPAS. Applies
        a CSWAP gate from ctrl_regs to state_A and state_B, measuring party_A_ancilla_regs,
        party_A_bell_regs and party_B_bell_regs.

        ctrl_reg: |phi> in the diagram (located in QPU controlled by party A)
        state_A: rho_i in the diagram (located in QPU controlled by party A)
        state_B: rho_j in the diagram (located in QPU controlled by party B)
        party_A_bell_regs: party A's qubit of the shared bell pair, |Phi^+>
        party_B_bell_regs: party B's qubit of the shared bell pair, |Phi^+>
        party_A_bell_cregs: classical registers for measuring party A's bell pair qubit
        party_B_bell_cregs: classical registers for measuring party B's bell pair qubit
        """

        apply_teleported_cnot(
            qc=qc,
            ctrl_regs=state_A,
            targ_regs=state_B,
            party_A_bell_regs=party_A_bell_regs,
            party_B_bell_regs=party_B_bell_regs,
            party_A_bell_cregs=party_A_bell_cregs,
            party_B_bell_cregs=party_B_bell_cregs
        )

        # reset and reshare bell pairs
        qc.reset(party_A_bell_regs + party_B_bell_regs)
        qc.h(party_A_bell_regs)
        qc.cx(party_A_bell_regs, party_B_bell_regs)

        # note: A and B are swapped because state_A is the target
        apply_teleported_toffoli(
            qc=qc,
            ctrl1_reg=ctrl_reg,
            ctrl2_regs=state_B,
            targ_regs=state_A,
            party_A_bell_regs=party_A_bell_regs,
            party_B_bell_regs=party_B_bell_regs,
            party_A_bell_cregs=party_A_bell_cregs,
            party_B_bell_cregs=party_B_bell_cregs,
            n_fanout_errors=n_fanout_errors
        )

        # reset and reshare bell pairs
        qc.reset(party_A_bell_regs + party_B_bell_regs)
        qc.h(party_A_bell_regs)
        qc.cx(party_A_bell_regs, party_B_bell_regs)

        apply_teleported_cnot(
            qc=qc,
            ctrl_regs=state_A,
            targ_regs=state_B,
            party_A_bell_regs=party_A_bell_regs,
            party_B_bell_regs=party_B_bell_regs,
            party_A_bell_cregs=party_A_bell_cregs,
            party_B_bell_cregs=party_B_bell_cregs
        )

def get_CSWAP_telegate_circ(
        input_bitstr: str,
        meas_all: bool = False,
        n_fanout_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ):
    n_qubits = len(input_bitstr)
    state_size = (n_qubits - 1)//2

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_qubits)]
    bell_pair_ancilla_regs = [QuantumRegister(1, f'a{i}') for i in range(2*state_size)]
    qubit_cregs = ClassicalRegister(n_qubits, f'data_cregs')
    anc_cregs = [ClassicalRegister(1, f'anc_cregs{i}') for i in range(2*state_size)]

    all_regs = qubits + bell_pair_ancilla_regs
    qc = QuantumCircuit(*all_regs, *anc_cregs, qubit_cregs)

    ctrl_reg = qc.qubits[0]
    state_A_regs = qc.qubits[1:state_size+1]
    state_B_regs = qc.qubits[state_size+1:n_qubits]
    party_A_bell_regs = qc.qubits[n_qubits:n_qubits+state_size]
    party_B_bell_regs = qc.qubits[n_qubits+state_size:n_qubits+2*state_size]

    party_A_bell_cregs = qc.clbits[:state_size]
    party_B_bell_cregs = qc.clbits[state_size:2*state_size]
    # prepare initial state for data qubits
    for i, val in enumerate(input_bitstr[::-1]):
        if val == '1':
            qc.x(i)

    # prepare bell pairs
    qc.h(party_A_bell_regs)
    qc.cx(party_A_bell_regs, party_B_bell_regs)

    apply_CSWAP_telegate(
        qc=qc,
        ctrl_reg=ctrl_reg,
        state_A=state_A_regs,
        state_B=state_B_regs,
        party_A_bell_regs=party_A_bell_regs,
        party_B_bell_regs=party_B_bell_regs,
        party_A_bell_cregs=party_A_bell_cregs,
        party_B_bell_cregs=party_B_bell_cregs,
        n_fanout_errors=n_fanout_errors
    )

    if meas_all:
        for i in range(n_qubits):
            qc.measure(qubits[i], qubit_cregs[i])

    return qc

# Helpers for CSWAP with fewer ancillas
def apply_teleported_toffoli_fewer_ancillas(
        qc: QuantumCircuit,
        ctrl1_reg: QuantumRegister,
        ctrl2_regs: list[QuantumRegister],
        targ_regs: list[QuantumRegister],
        party_A_anc_regs: list[QuantumRegister],
        party_A_anc_cregs: list[ClassicalRegister],
        n_fanout_errors: list[tuple[str, float]] | None = None,
        pre_teletoffoli_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ) -> None:
        """
        Teleported toffoli cirtcuit  from Fig. 6(d) of COMPAS (in orange)

        ctrl1_reg: |phi> in the diagram (located in QPU controlled by party A)
        ctrl2_regs: rho_j in the diagram (located in QPU controlled by party A)
        targ_regs: rho_i in the diagram (located in QPU controlled by party B)
        party_A_ancilla_regs: party A's qubit of the shared bell pair, |Phi^+>
        """
        qc.cx(ctrl2_regs, party_A_anc_regs)

        if pre_teletoffoli_errors is not None:
            for i in range(len(ctrl2_regs)):
                add_custom_error_injection(qc, [party_A_anc_regs[i], ctrl2_regs[i]], pre_teletoffoli_errors)

        apply_parallel_toffoli_via_fanout(
            qc=qc,
            ctrl1_reg=ctrl1_reg,
            ctrl2_regs=party_A_anc_regs,
            targ_regs=targ_regs,
            n_fanout_errors=n_fanout_errors
        )

        for i in range(len(targ_regs)):
            qc.measure(party_A_anc_regs[i], party_A_anc_cregs[i])
            with qc.if_test((party_A_anc_cregs[i], 1)): # type: ignore
                qc.z(ctrl2_regs[i])


# CSWAPs with fewer ancillas
def apply_CSWAP_teledata_fewer_ancillas(
        qc: QuantumCircuit,
        ctrl_reg: QuantumRegister,
        state_A: list[QuantumRegister],
        state_B: list[QuantumRegister],
        n_fanout_errors: list[tuple[str, float]] | None = None,
        teledata_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ) -> None:
        """
        CSWAP circuit from Fig. 6(c) of COMPAS (purple). Applies
        a CSWAP gate from ctrl_regs to state_A and state_B, measuring party_A_ancilla_regs,
        party_A_bell_regs and party_B_bell_regs. In this implementation, we skip the teleportations
        and inject noise into the circuit based on other experiments

        ctrl_reg: |phi> in the diagram (located in QPU controlled by party A)
        state_A: rho_i in the diagram (located in QPU controlled by party A)
        state_B: rho_j in the diagram (located in QPU controlled by party B)
        """
        if teledata_errors is not None:
            for qubit in state_B:
                add_custom_error_injection(qc, [qubit], teledata_errors)

        qc.cx(state_B, state_A)
        apply_parallel_toffoli_via_fanout(
            qc=qc,
            ctrl1_reg=ctrl_reg,
            ctrl2_regs=state_A,
            targ_regs=state_B,
            n_fanout_errors=n_fanout_errors,
        )
        qc.cx(state_B, state_A)

        if teledata_errors is not None:
            for qubit in state_B:
                add_custom_error_injection(qc, [qubit], teledata_errors)

def get_CSWAP_teledata_fewer_ancillas_circ(
        input_bitstr: str,
        meas_all: bool = False,
        n_fanout_errors: list[tuple[str, float]] | None = None,
        teledata_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ):

    n_qubits = len(input_bitstr)
    state_size = (n_qubits - 1)//2

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_qubits)]
    qubit_cregs = ClassicalRegister(n_qubits, f'data_cregs')

    qc = QuantumCircuit(*qubits, qubit_cregs)

    ctrl_reg = qc.qubits[0]
    state_A_regs = qc.qubits[1:state_size+1]
    state_B_regs = qc.qubits[state_size+1:n_qubits]

    # prepare initial state for data qubits
    for i, val in enumerate(input_bitstr[::-1]):
        if val == '1':
            qc.x(i)


    apply_CSWAP_teledata_fewer_ancillas(
        qc=qc,
        ctrl_reg=ctrl_reg,
        state_A=state_A_regs,
        state_B=state_B_regs,
        n_fanout_errors=n_fanout_errors,
        teledata_errors=teledata_errors
    )

    if meas_all:
        for i in range(n_qubits):
            qc.measure(qubits[i], qubit_cregs[i])

    return qc

def apply_CSWAP_telegate_fewer_ancillas(
        qc: QuantumCircuit,
        ctrl_reg: QuantumRegister,
        state_A: list[QuantumRegister],
        state_B: list[QuantumRegister],
        party_A_anc_regs: list[QuantumRegister],
        party_A_anc_cregs: list[ClassicalRegister],
        n_fanout_errors: list[tuple[str, float]] | None = None,
        telecnot_errors: list[tuple[str, float]] | None = None,
        pre_teletoffoli_errors: list[tuple[str, float]] | None = None,
    ) -> None:
        """
        CSWAP circuit from Fig. 6(b) of COMPAS. Applies
        a CSWAP gate from ctrl_regs to state_A and state_B, measuring party_A_ancilla_regs,
        party_A_bell_regs and party_B_bell_regs.

        ctrl_reg: |phi> in the diagram (located in QPU controlled by party A)
        state_A: rho_i in the diagram (located in QPU controlled by party A)
        state_B: rho_j in the diagram (located in QPU controlled by party B)
        party_A_anc_regs: party A's ancilla qubits, |0>
        """

        qc.cx(state_A, state_B)

        if telecnot_errors is not None:
            for i in range(len(state_B)):
                add_custom_error_injection(qc, [state_A[i], state_B[i]], telecnot_errors)

        # note: A and B are swapped because state_A is the target
        apply_teleported_toffoli_fewer_ancillas(
            qc=qc,
            ctrl1_reg=ctrl_reg,
            ctrl2_regs=state_B,
            targ_regs=state_A,
            party_A_anc_regs=party_A_anc_regs,
            party_A_anc_cregs=party_A_anc_cregs,
            n_fanout_errors=n_fanout_errors,
            pre_teletoffoli_errors=pre_teletoffoli_errors,
        )

        qc.cx(state_A, state_B)
        if telecnot_errors is not None:
            for i in range(len(state_B)):
                add_custom_error_injection(qc, [state_A[i], state_B[i]], telecnot_errors)

def get_CSWAP_telegate_fewer_ancillas_circ(
        input_bitstr: str,
        meas_all: bool = False,
        n_fanout_errors: list[tuple[str, float]] | None = None,
        telecnot_errors: list[tuple[str, float]] | None = None,
        pre_teletoffoli_errors: list[tuple[str, float]] | None = None,
        **kwargs
    ):
    n_qubits = len(input_bitstr)
    state_size = (n_qubits - 1)//2

    qubits = [QuantumRegister(1, f't{i}') for i in range(n_qubits)]
    ancilla_regs = [QuantumRegister(1, f'a{i}') for i in range(state_size)]
    qubit_cregs = ClassicalRegister(n_qubits, f'data_cregs')
    anc_cregs = [ClassicalRegister(1, f'anc_cregs{i}') for i in range(state_size)]


    all_regs = qubits + ancilla_regs
    qc = QuantumCircuit(*all_regs, *anc_cregs, qubit_cregs)

    ctrl_reg = qc.qubits[0]
    state_A_regs = qc.qubits[1:state_size+1]
    state_B_regs = qc.qubits[state_size+1:n_qubits]
    party_A_anc_regs = qc.qubits[n_qubits:n_qubits+state_size]

    # prepare initial state for data qubits
    for i, val in enumerate(input_bitstr[::-1]):
        if val == '1':
            qc.x(i)

    apply_CSWAP_telegate_fewer_ancillas(
        qc=qc,
        ctrl_reg=ctrl_reg,
        state_A=state_A_regs,
        state_B=state_B_regs,
        party_A_anc_regs=party_A_anc_regs,
        party_A_anc_cregs=anc_cregs,
        n_fanout_errors=n_fanout_errors,
        telecnot_errors=telecnot_errors,
        pre_teletoffoli_errors=pre_teletoffoli_errors,
    )

    if meas_all:
        for i in range(n_qubits):
            qc.measure(qubits[i], qubit_cregs[i])

    return qc