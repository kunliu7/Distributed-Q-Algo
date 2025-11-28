from dqalgo.nisq.circuits import get_teleported_toffoli_circ
from dqalgo.nisq.utils import (
    classically_compute_toffoli
)

from test_circuit import get_test_ideal, get_truth_table_tomography

test_teleported_toffoli_ideal = get_test_ideal(
    classical_eval=classically_compute_toffoli,
    circuit_builder=get_teleported_toffoli_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    max_test_size=3
)

test_teleported_toffoli_truth_table_tomography = get_truth_table_tomography(
    classical_eval=classically_compute_toffoli,
    circuit_builder=get_teleported_toffoli_circ,
    n_trgts=3,  # Adjust as needed for your tests
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
)