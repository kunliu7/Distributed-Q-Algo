
from qutip import *


def get_GHZ_state(n_qubits: int) -> Qobj:
    zero_state = tensor([basis(2, 0) for _ in range(n_qubits)])
    one_state = tensor([basis(2, 1) for _ in range(n_qubits)])
    ghz_state = (zero_state + one_state).unit()
    return ghz_state
