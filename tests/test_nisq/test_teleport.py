
import numpy as np
import pytest

from dqalgo.nisq.teleports import eval_teleport_circ


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("p2", [0.000, 0.005])
def test_teleport_circuit(n: int, p2: float):
    n_shots = 1000
    p1 = p2/10
    pm = p2
    error_counts = eval_teleport_circ(n, p1, p2, pm, n_shots)
    if np.isclose(p2, 0.0):
        assert error_counts == {}
    top_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 errors:")
    for error, count in top_errors:
        print(f"  {error}: {count}/{n_shots}")
