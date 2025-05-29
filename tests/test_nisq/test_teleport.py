
import numpy as np
import pytest
import stim
from tqdm import tqdm

from dqalgo.nisq.teleports import TeleportCircBuilder


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("p2", [0.000, 0.005])
def test_teleportation_circuit(n: int, p2: float):
    n_shots = 10000
    p1 = p2/10
    pm = p2
    ideal_sim = TeleportCircBuilder(n).sim
    ideal_inv_tableau = ideal_sim.current_inverse_tableau()
    error_counts = {}
    for i in tqdm(range(n_shots), desc=f"{n=}, {p2=}"):
        builder = TeleportCircBuilder(n, p1, p2, pm)
        noisy_sim = builder.sim
        noisy_inv_tableau = noisy_sim.current_inverse_tableau()
        pauli_error = (ideal_inv_tableau.inverse() * noisy_inv_tableau).to_pauli_string()
        remaining_pauli_error = pauli_error[3*n:4*n]
        if np.isclose(p2, 0.0):
            assert remaining_pauli_error == stim.PauliString("I"*n)

        if remaining_pauli_error != stim.PauliString("I"*n):
            key = str(remaining_pauli_error)
            error_counts[key] = error_counts.get(key, 0) + 1
        else:
            assert remaining_pauli_error == stim.PauliString("I"*n)
    
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 errors:")
    for error, count in top_errors:
        print(f"  {error}: {count}/{n_shots}")
