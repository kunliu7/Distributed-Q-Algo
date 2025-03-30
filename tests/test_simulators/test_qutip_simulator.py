
import itertools

from qutip import *
from tqdm import tqdm

from dqalgo.simulators.qutip_simulator import QTCircuit
from dqalgo.simulators.utils import get_GHZ_state


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


def test_FANOUT():
    print("Testing FANOUT")
    n_trgts = 3
    n_shots = 100
    # plus = (basis(2, 0) + basis(2, 1)).unit()  # |+⟩ state
    ghz_state = get_GHZ_state(n_trgts)
    base01 = [basis(2, 0), basis(2, 1)]
    all_trgt_bitstrings = list(itertools.product([0, 1], repeat=n_trgts))

    for ctrl_bit, ctrl_state in enumerate(base01):
        for bits in all_trgt_bitstrings:
            print(f"======================= Testing {ctrl_bit=}, target={bits} ============================== ")
            # all_basis_states = [[base01[bit]] for bit in bits]
            perm = []
            for i in range(n_trgts):
                perm.append(i)          # GHZ_i
                perm.append(i + n_trgts)  # Q_i

            # print(f"perm: {perm}")
            all_basis_states = tensor([ghz_state] + [base01[bit] for bit in bits])
            # print(f"all_basis_states: {all_basis_states}")
            all_basis_states = all_basis_states.permute(perm)

            # print(f"all_basis_states: {all_basis_states}")
            for _ in tqdm(range(n_shots)):
                init_state = tensor([ctrl_state, all_basis_states])
                # print(f"\n{init_state=}\n")

                circ = QTCircuit(n_qubits=2*n_trgts + 1, n_clbits=3*n_trgts+1, init_state=init_state)
                circ.M_ZZ(0, 1, 1)  # m1
                for i in range(n_trgts):
                    m2_idx = 3*i + 2
                    m3_idx = 3*i + 3
                    circ.M_XX(2*i + 1, 2*i + 2, m2_idx)  # m2 = 2*i + 2
                    circ.M_Z(2*i + 1, m3_idx)  # m3 = 2*i + 3
                    circ.c_X(2*i + 2, lambda cregs: (cregs[1] + cregs[m3_idx]) % 2 == 1)  # m1 + m3

                def _correction_condition_on_ctrl(cregs: list[int]) -> bool:
                    s = 0
                    for k in range(n_trgts):
                        s += cregs[3*k + 2]  # m2
                        # print(f"cregs[{3*k + 2}]: {cregs[3*k + 2]}")
                    return s % 2 == 1

                circ.c_Z(0, _correction_condition_on_ctrl)  # m2
                # circ.draw()
                final_state = circ.get_state()
                selected_qubits = [2*j for j in range(n_trgts+1)]
                # print(f"selected_qubits: {selected_qubits}")
                final_state = final_state.ptrace(selected_qubits)
                # print(f"\n{final_state=}\n")
                target_bits = [(ctrl_bit + origin_trgt_bit) % 2 for origin_trgt_bit in bits]
                # print(f"original: {ctrl_bit}{bits} -> target: {ctrl_bit}{target_bits}")
                target_state = tensor([ctrl_state] + [base01[target_bit]
                                      for target_bit in target_bits]).unit()
                # print(f"\n{target_state=}\n")
                target_state = ket2dm(target_state)
                fidelity_val = fidelity(final_state, target_state)
                assert fidelity_val > 0.99999  # Should be very close to 1
                # print(f"Fidelity with target state {b0}, {target_qubit}: {fidelity_val:.6f}")
                # print(f"msmt result: {circ.cregs}")
                # print(final_state)


def test_CNOT2():
    print("Testing CNOT 2, deprecated")
    n_trgts = 1

    q1 = basis(2, 0)
    basis_states = [basis(2, 0), basis(2, 1)]

    for b0, q0 in enumerate(basis_states):
        for b2, q2 in enumerate(basis_states):
            print(f"======================= Testing {b0}, {b2} ============================== ")
            init_state = tensor([q0, q1, q2])

            circ = QTCircuit(3, 4, init_state=init_state)
            circ.M_XX(1, 2, 1)
            circ.c_Z(0, 1)
            circ.M_ZZ(0, 1, 2)
            circ.c_X(0, 2)
            circ.M_X(1, 3)
            circ.c_Z(0, 3)
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


def test_CNOT3():
    """
    Test CNOT with 3 qubits from https://arxiv.org/pdf/1709.02318, Figure 3
    """
    print("Testing CNOT 3")
    n_trgts = 1
    n_shots = 1

    q1 = basis(2, 0)
    basis_states = [basis(2, 0), basis(2, 1)]

    for b0, q0 in enumerate(basis_states):
        for b2, q2 in enumerate(basis_states):
            print(f"======================= Testing {b0}, {b2} ============================== ")
            for _ in range(n_shots):
                init_state = tensor([q0, q1, q2])

                circ = QTCircuit(3, 4, init_state=init_state)
                circ.M_ZZ(0, 1, 1)  # m1
                circ.M_XX(1, 2, 2)  # m2
                circ.M_Z(1, 3)
                circ.c_Z(0, 2)  # m2
                circ.c_X(2, lambda cregs: (cregs[1] + cregs[3]) % 2)  # m1 + m3
                circ.draw()

                final_state = circ.get_state()
                final_state = final_state.ptrace([0, 2])

                target_qubit = b2 if b0 == 0 else (b0 + b2) % 2
                # print(f"{b0}, {b2} -> {b0}, {target_qubit}")
                target_state = tensor([basis_states[b0], basis_states[target_qubit]]).unit()
                target_state = ket2dm(target_state)
                fidelity_val = fidelity(final_state, target_state)

                assert fidelity_val > 0.99999  # Should be very close to 1
                # print(f"Fidelity with target state {b0}, {target_qubit}: {fidelity_val:.6f}")
                # print(f"msmt result: {circ.cregs}")
                # print(final_state)
