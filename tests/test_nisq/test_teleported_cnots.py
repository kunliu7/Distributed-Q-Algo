import numpy as np
import pytest

from dqalgo.nisq.teleported_cnots import eval_teleported_CNOT_circ


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("p2", [0.000, 0.005])
def test_teleported_CNOTs(n: int, p2: float):
    n_shots = 1000
    error_counts = eval_teleported_CNOT_circ(n, p2/10, p2, p2, n_shots=n_shots)
    if np.isclose(p2, 0.0):
        assert error_counts == {}
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 errors:")
    for error, count in top_errors:
        print(f"  {error}: {count}/{n_shots}")