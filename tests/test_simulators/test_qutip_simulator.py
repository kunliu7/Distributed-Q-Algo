from qutip import *

from dqalgo.simulators.qutip_simulator import QTCircuit


def test_Bell_pair():
    # Setup circuit with 4 qubits and 2 classical registers
    circ = QTCircuit(n_qubits=4, n_clbits=2)

    # Bell pair prep on (0,1) and (2,3)
    circ.M_XX(0, 1, creg_idx=0)
    circ.M_XX(2, 3, creg_idx=1)

    # Apply conditional Z corrections
    circ.c_Z(1, 0)
    circ.c_Z(3, 1)

    # Retrieve final state
    final_state = circ.get_state()

    # Define target Bell pairs for fidelity checking
    bell01 = (tensor(basis(2, 0), basis(2, 0)) + tensor(basis(2, 1), basis(2, 1))).unit()
    bell23 = bell01  # same structure

    # Full Bell tensor product state: (|00⟩+|11⟩)⊗(|00⟩+|11⟩)
    bell_target = ket2dm(tensor(bell01, bell23))
    fidelity_val = fidelity(final_state, bell_target)

    print(f"Final state fidelity with target Bell pairs: {fidelity_val:.6f}")


def test_GHZ4():
    # Setup circuit with 4 qubits and 2 classical registers
    circ = QTCircuit(n_qubits=4, n_clbits=3)

    # Bell pair prep on (0,1) and (2,3)
    circ.M_XX(0, 1, creg_idx=0)
    circ.M_XX(2, 3, creg_idx=1)

    # Apply conditional Z corrections
    circ.c_Z(1, 0)
    circ.c_Z(3, 1)

    circ.M_ZZ(1, 2, 2)
    circ.c_X(2, 2)
    circ.c_X(3, 2)

    # Retrieve final state
    final_state = circ.get_state()

    # Define target GHZ state: (|0000⟩ + |1111⟩)/√2
    ghz_target = (tensor(basis(2, 0), basis(2, 0), basis(2, 0), basis(2, 0)) +
                  tensor(basis(2, 1), basis(2, 1), basis(2, 1), basis(2, 1))).unit()
    ghz_target = ket2dm(ghz_target)

    fidelity_val = fidelity(final_state, ghz_target)
    print(f"Final state fidelity with target GHZ state: {fidelity_val:.6f}")
    assert fidelity_val > 0.99999  # Should be very close to 1


def test_Fanout():
    n_trgts = 2
    circ = QTCircuit(n_qubits=2*n_trgts + 1, n_clbits=2*n_trgts+2)

    circ.M_ZZ(0, 1, 1)
    for i in range(n_trgts):
        circ.M_XX(2*i + 1, 2*i + 2, 2*i + 2)
        circ.M_Z(2*i + 1, 2*i + 3)
        circ.c_Z(2*i + 2, 2*i + 2)

    def cond_func(cregs):
        s = 0
        for i in range(n_trgts):
            if cregs[2*i + 1] == 1:
                s += 1
        return s % 2 == 1

    circ.c_X(0, cond_func)
    final_state = circ.get_state()
    print(final_state)
