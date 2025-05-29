import stim
from tqdm import tqdm


def build_parallel_CNOTs(n: int, p2: float = 0.0) -> stim.TableauSimulator:
    circ = stim.TableauSimulator()
    for i in range(n):
        q_in = i
        q_out = i + n
        circ.cx(q_in, q_out)
        if p2 > 0:
            circ.depolarize2(q_in, q_out, p=p2)
    return circ

def eval_parallel_CNOTs(n: int, p2: float, n_shots: int, verbose: bool = False) -> dict[str, int]:
    """Evaluate the error distribution of the parallel CNOTs circuit.

    Args:
        n (int): the number of CNOTs
        p2 (float): the error rate of the 2-qubit gate
        n_shots (int): the number of shots
        verbose (bool, optional): whether to print the progress. Defaults to False.

    Returns:
        dict[str, int]: the error distribution of the parallel CNOTs circuit. E.g., {'XZ': 100, 'YZ': 200, 'ZZ': 300}
    """
    ideal_sim = build_parallel_CNOTs(n)
    ideal_inv_tableau = ideal_sim.current_inverse_tableau()
    error_counts = {}
    for i in tqdm(range(n_shots), desc=f"{n=}, {p2=}", disable=not verbose):
        noisy_sim = build_parallel_CNOTs(n, p2)
        noisy_inv_tableau = noisy_sim.current_inverse_tableau()
        pauli_error = (ideal_inv_tableau.inverse() * noisy_inv_tableau).to_pauli_string()
        remaining_pauli_error = pauli_error

        if remaining_pauli_error != stim.PauliString("I"*2*n):
            key = str(remaining_pauli_error)
            error_counts[key] = error_counts.get(key, 0) + 1
        else:
            assert remaining_pauli_error == stim.PauliString("I"*2*n)
    
    return error_counts
