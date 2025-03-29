
from qutip import *

from dqalgo.simulators.qutip_simulator import QTCircuit


def test_Bell_pair():
    # Setup circuit with 4 qubits and 2 classical registers
    for i in range(1000):
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
        assert fidelity_val > 0.99999  # Should be very close to 1

        # print(f"Final state fidelity with target Bell pairs: {fidelity_val:.6f}")


def test_GHZ4():
    for i in range(1000):
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
        # print(f"Final state fidelity with target GHZ state: {fidelity_val:.6f}")
        assert fidelity_val > 0.99999  # Should be very close to 1


def test_CNOT():
    print("Testing CNOT")
    n_trgts = 1

    q1 = (basis(2, 0) + basis(2, 1)).unit()  # |+⟩ state
    basis_states = [basis(2, 0), basis(2, 1)]

    for b0, q0 in enumerate(basis_states):
        for b2, q2 in enumerate(basis_states):
            print(f"======================= Testing {b0}, {b2} ============================== ")
            init_state = tensor([q0, q1, q2])

            circ = QTCircuit(n_qubits=2*n_trgts + 1, n_clbits=3*n_trgts+1, init_state=init_state)
            circ.M_ZZ(0, 1, 1)
            for i in range(n_trgts):
                circ.M_XX(2*i + 1, 2*i + 2, 2*i + 2)
                circ.M_Z(2*i + 1, 2*i + 3)
                circ.c_Z(2*i + 2, 2*i + 2)

            circ.c_X(0, lambda cregs: (cregs[1] + cregs[3]) % 2 == 1)
            circ.draw()
            final_state = circ.get_state()
            final_state = final_state.ptrace([0, 2])

            target_qubit = b2 if b0 == 0 else (b0 + b2) % 2
            print(f"{b0}, {b2} -> {b0}, {target_qubit}")
            target_state = tensor([basis_states[b0], basis_states[target_qubit]]).unit()
            target_state = ket2dm(target_state)
            fidelity_val = fidelity(final_state, target_state)
            print(f"Fidelity with target state {b0}, {target_qubit}: {fidelity_val:.6f}")
            print(f"msmt result: {circ.cregs}")
            print(final_state)
