from dqalgo.nisq.circuits import (
    get_CSWAP_teledata_circ,
    get_CSWAP_teledata_fewer_ancillas_circ,
    get_CSWAP_telegate_circ,
    get_CSWAP_telegate_fewer_ancillas_circ
)

from dqalgo.nisq.utils import classically_compute_CSWAP

from test_circuit import get_test_ideal, get_truth_table_tomography

test_CSWAP_teledata_ideal = get_test_ideal(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_teledata_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    max_test_size=2
)

test_CSWAP_telegate_ideal = get_test_ideal(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_telegate_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    max_test_size=2
)

test_cswap_teledata_truth_table_tomography = get_truth_table_tomography(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_teledata_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    n_trgts=2,  # Adjust as needed for your tests
)

test_cswap_telegate_truth_table_tomography = get_truth_table_tomography(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_telegate_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    n_trgts=2,  # Adjust as needed for your tests
)

test_CSWAP_teledata_fewer_ancillas_ideal = get_test_ideal(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_teledata_fewer_ancillas_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    max_test_size=4,
)

test_cswap_teledata_fewer_ancillas_truth_table_tomography = get_truth_table_tomography(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_teledata_fewer_ancillas_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    n_trgts=3,
    error_types=('fanout', 'teledata'),
    shots_per_circ=512, # crshes with anything higher than 512
    circs_per_input=2
)

test_CSWAP_telegate_fewer_ancillas_ideal = get_test_ideal(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_telegate_fewer_ancillas_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    max_test_size=4,
)

test_cswap_telegate_fewer_ancillas_truth_table_tomography = get_truth_table_tomography(
    classical_eval=classically_compute_CSWAP,
    circuit_builder=get_CSWAP_telegate_fewer_ancillas_circ,
    get_data_qubits=lambda n_trgts: 2*n_trgts + 1,
    n_trgts=3,
    error_types=('fanout', 'telegate'),
    shots_per_circ=1024,
    circs_per_input=1
)
