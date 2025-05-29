import itertools

import numpy as np

from dqalgo.nisq.eval import compute_classical_fidelity, normalize_counts
from dqalgo.nisq.experimental_noise import (
    get_fanout_error_probs,
    get_pre_teletoffoli_error_probs,
    get_post_teletoffoli_error_probs,
    get_telecnot_error_probs,
    get_teledata_error_probs,
)

from dqalgo.nisq.utils import (
    get_counts_of_first_n_regs,
    get_depolarizing_noise_model,
    sample_bitstrings,
    update_total_counts,
)

from qiskit_aer import AerSimulator

from tqdm import tqdm

def get_test_ideal(classical_eval, circuit_builder, get_data_qubits, max_test_size=5):
    def test_ideal():
        for n_trgts in range(1, max_test_size+1):
            print(f"Testing {n_trgts} targets")
            n_data_qubits = get_data_qubits(n_trgts)

            # test all possible initial basis states
            for input_bitstr in itertools.product([0, 1], repeat=n_data_qubits):
                expected_trgt_bitstr = classical_eval(input_bitstr)

                input_bitstr = "".join(map(str, input_bitstr))

                qc = circuit_builder(input_bitstr=input_bitstr, meas_all=True)
                counts = AerSimulator().run(qc).result().get_counts()
                reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
                assert len(reg_counts) == 1
                assert list(reg_counts.keys())[0] == expected_trgt_bitstr

    return test_ideal

def get_truth_table_tomography(
        classical_eval,
        circuit_builder,
        n_trgts,
        get_data_qubits,
        p_err=0.0001,
        shots_per_circ=128, # Crashes with anything higher than 128
        circs_per_input=10, # Repeat 10 times to compensate for low shots
        samples=150, # Sample space gets large so choose 150 random input bitstrings
        error_types=('fanout',)
    ):

    def test_truth_table_tomography():
        n_data_qubits = get_data_qubits(n_trgts)
        n_samples = min(samples, 2**n_data_qubits)

        n_fanout_errors = None
        two_n_fanout_errors = None
        teledata_errors = None
        pre_teletoffoli_errors= None
        post_teletoffoli_errors= None
        telecnot_errors = None

        if 'fanout' in error_types:
            n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=10*p_err)
            two_n_fanout_errors = get_fanout_error_probs(n_trgts=2*n_trgts, p2=10*p_err)

        if 'teledata' in error_types:
            teledata_errors = get_teledata_error_probs(n_trgts=n_trgts, p2=10*p_err)

        if 'telegate' in error_types:
            pre_teletoffoli_errors = get_pre_teletoffoli_error_probs(n_trgts=n_trgts, p2=10*p_err)
            post_teletoffoli_errors = get_post_teletoffoli_error_probs(n_trgts=n_trgts, p2=10*p_err)
            telecnot_errors = get_telecnot_error_probs(n_trgts=n_trgts, p2=10*p_err)

        fids = []
        noise_model = get_depolarizing_noise_model(p_1q=p_err, p_2q=p_err*10, p_meas=p_err)
        sim = AerSimulator(noise_model=noise_model)

        print('Constructing circuits')

        for input_bitstr in tqdm(sample_bitstrings(n_data_qubits, n_samples), total=n_samples):
            expected_output_bitstr = classical_eval(input_bitstr)
            ideal_counts = {expected_output_bitstr: 1.0}

            total_counts = {}
            for _ in range(circs_per_input):
                qc = circuit_builder(
                    input_bitstr=input_bitstr,
                    meas_all=True,
                    n_fanout_errors=n_fanout_errors,
                    two_n_fanout_errors=two_n_fanout_errors,
                    teledata_errors=teledata_errors,
                    pre_teletoffoli_errors=pre_teletoffoli_errors,
                    post_teletoffoli_errors=post_teletoffoli_errors,
                    telecnot_errors=telecnot_errors
                )

                results = sim.run(qc, shots=shots_per_circ).result()
                counts = results.get_counts()
                reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
                update_total_counts(total_counts, reg_counts)

            normed_noisy_counts = normalize_counts(total_counts)
            fid = compute_classical_fidelity(ideal_counts, normed_noisy_counts)
            fids.append(fid)

        mean_fid = np.mean(fids)
        stddev_fid = np.std(fids)

        print(f"mean fidelity: {mean_fid}")
        print(f"stddev fidelity: {stddev_fid}")

    return test_truth_table_tomography

