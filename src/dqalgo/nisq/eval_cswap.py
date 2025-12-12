raise DeprecationWarning("This module is deprecated. Please use the eval module instead.")
import random
from multiprocessing import Pool, cpu_count

import numpy as np
from qiskit_aer import AerSimulator
from tqdm import tqdm

from dqalgo.nisq.circuits import (get_CSWAP_teledata_fewer_ancillas_circ,
                                  get_CSWAP_telegate_fewer_ancillas_circ)
from dqalgo.nisq.experimental_noise import (get_fanout_error_probs,
                                            get_pre_teletoffoli_error_probs,
                                            get_telecnot_error_probs,
                                            get_teledata_error_probs)
from dqalgo.nisq.utils import (classically_compute_CSWAP,
                               get_counts_of_first_n_regs,
                               get_depolarizing_noise_model, sample_bitstrings,
                               update_total_counts)

from .eval import compute_classical_fidelity, normalize_counts


def eval_CSWAP_teledata_single_thread(
        n_trgts: int,
        p2: float,
        circs_per_input: int,
        shots_per_circ: int,
    ):
    n_data_qubits = 2*n_trgts + 1

    n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=p2)
    teledata_errors = get_teledata_error_probs(n_trgts=1, p2=p2)

    noise_model = get_depolarizing_noise_model(p_1q=p2/10, p_2q=p2, p_meas=p2)
    sim = AerSimulator(noise_model=noise_model)

    print('Constructing circuit')

    random_index = random.randint(0, 2**n_data_qubits)
    input_bitstr = bin(random_index)[2:].zfill(n_data_qubits)

    expected_output_bitstr = classically_compute_CSWAP(input_bitstr)
    ideal_counts = {expected_output_bitstr: 1.0}

    total_counts = {}
    for _ in range(circs_per_input):
        qc = get_CSWAP_teledata_fewer_ancillas_circ(
            input_bitstr=input_bitstr,
            meas_all=True,
            n_fanout_errors=n_fanout_errors,
            teledata_errors=teledata_errors,
        )

        results = sim.run(qc, shots=shots_per_circ).result()
        counts = results.get_counts()
        reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
        update_total_counts(total_counts, reg_counts)

    normed_noisy_counts = normalize_counts(total_counts)
    return compute_classical_fidelity(ideal_counts, normed_noisy_counts)


def eval_CSWAP_telegate_single_thread(
        n_trgts: int,
        p2: float,
        circs_per_input: int,
        shots_per_circ: int,
    ):
    n_data_qubits = 2*n_trgts + 1

    n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=p2)
    telecnot_errors = get_telecnot_error_probs(n_trgts=1, p2=p2)
    pre_teletoffoli_errors = get_pre_teletoffoli_error_probs(n_trgts=1, p2=p2)

    noise_model = get_depolarizing_noise_model(p_1q=p2/10, p_2q=p2, p_meas=p2)
    sim = AerSimulator(noise_model=noise_model)

    print('Constructing circuit')

    random_index = random.randint(0, 2**n_data_qubits)
    input_bitstr = bin(random_index)[2:].zfill(n_data_qubits)

    expected_output_bitstr = classically_compute_CSWAP(input_bitstr)
    ideal_counts = {expected_output_bitstr: 1.0}

    total_counts = {}
    for _ in range(circs_per_input):
        qc = get_CSWAP_telegate_fewer_ancillas_circ(
            input_bitstr=input_bitstr,
            meas_all=True,
            n_fanout_errors=n_fanout_errors,
            telecnot_errors=telecnot_errors,
            pre_teletoffoli_errors=pre_teletoffoli_errors
        )

        results = sim.run(qc, shots=shots_per_circ).result()
        counts = results.get_counts()
        reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
        update_total_counts(total_counts, reg_counts)

    normed_noisy_counts = normalize_counts(total_counts)
    return compute_classical_fidelity(ideal_counts, normed_noisy_counts)


def evaluate_single_input_teledata(args):
    (
        input_bitstr,
        n_trgts,
        p_err,
        shots_per_circ,
        circs_per_input,
        n_fanout_errors,
        teledata_errors,
    ) = args
    n_data_qubits = 2*n_trgts + 1

    expected_output_bitstr = classically_compute_CSWAP(input_bitstr)
    ideal_counts = {expected_output_bitstr: 1.0}

    noise_model = get_depolarizing_noise_model(p_1q=p_err, p_2q=p_err*10, p_meas=p_err)
    sim = AerSimulator(noise_model=noise_model)

    total_counts = {}
    for _ in range(circs_per_input):
        qc = get_CSWAP_teledata_fewer_ancillas_circ(
            input_bitstr=input_bitstr,
            meas_all=True,
            n_fanout_errors=n_fanout_errors,
            teledata_errors=teledata_errors,
        )

        results = sim.run(qc, shots=shots_per_circ).result()
        counts = results.get_counts()
        reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
        update_total_counts(total_counts, reg_counts)

    normed_noisy_counts = normalize_counts(total_counts)
    return compute_classical_fidelity(ideal_counts, normed_noisy_counts)


def eval_CSWAP_teledata_parallel(n_trgts: int, p_err: float, shots_per_circ=1024, circs_per_input=1, n_samples=150, n_processes=None) -> tuple[float, float]:
    """Parallelized version of eval_CSWAP_teledata that processes input bitstrings in parallel.

    Args:
        n_trgts: Number of target qubits
        p_err: Error probability
        shots_per_circ: Number of shots per circuit
        circs_per_input: Number of circuits per input
        n_samples: Number of input samples
        n_processes: Number of processes to use (defaults to CPU count)

    Returns:
        tuple[float, float]: Mean and standard deviation of fidelity
    """
    n_data_qubits = 2*n_trgts + 1
    n_samples = min(n_samples, 2**n_data_qubits)

    n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=10*p_err)
    teledata_errors = get_teledata_error_probs(n_trgts=1, p2=10*p_err)

    print('Constructing circuits')

    # Prepare arguments for parallel processing
    eval_args = [
        (
            input_bitstr,
            n_trgts,
            p_err,
            shots_per_circ,
            circs_per_input,
            n_fanout_errors,
            teledata_errors,
        )
        for input_bitstr in sample_bitstrings(n_data_qubits, n_samples)
    ]

    # Use number of processes specified or default to CPU count
    n_processes = n_processes if n_processes is not None else cpu_count()
    print(f'Using {n_processes} processes')

    # Process evaluations in parallel with progress bar
    fids = []
    with Pool(processes=n_processes) as pool:
        for fid in tqdm(pool.imap(evaluate_single_input_teledata, eval_args), total=n_samples, desc="Processing inputs"):
            fids.append(fid)

    mean_fid = float(np.mean(fids))
    stddev_fid = float(np.std(fids))

    return mean_fid, stddev_fid


def evaluate_single_input_telegate(args):
    (
        input_bitstr,
        n_trgts,
        p_err,
        shots_per_circ,
        circs_per_input,
        n_fanout_errors,
        telecnot_errors,
        pre_teletoffoli_errors,
    ) = args
    n_data_qubits = 2*n_trgts + 1

    expected_output_bitstr = classically_compute_CSWAP(input_bitstr)
    ideal_counts = {expected_output_bitstr: 1.0}

    noise_model = get_depolarizing_noise_model(p_1q=p_err, p_2q=p_err*10, p_meas=p_err)
    sim = AerSimulator(noise_model=noise_model)

    total_counts = {}
    for _ in range(circs_per_input):
        qc = get_CSWAP_telegate_fewer_ancillas_circ(
            input_bitstr=input_bitstr,
            meas_all=True,
            n_fanout_errors=n_fanout_errors,
            telecnot_errors=telecnot_errors,
            pre_teletoffoli_errors=pre_teletoffoli_errors,
        )

        results = sim.run(qc, shots=shots_per_circ).result()
        counts = results.get_counts()
        reg_counts = get_counts_of_first_n_regs(counts, n_data_qubits)
        update_total_counts(total_counts, reg_counts)

    normed_noisy_counts = normalize_counts(total_counts)
    return compute_classical_fidelity(ideal_counts, normed_noisy_counts)


def eval_CSWAP_telegate_parallel(n_trgts: int, p_err: float, shots_per_circ=128, circs_per_input=10, n_samples=150, n_processes=None) -> tuple[float, float]:
    """Parallelized version of eval_CSWAP_telegate that processes input bitstrings in parallel.

    Args:
        n_trgts: Number of target qubits
        p_err: Error probability
        shots_per_circ: Number of shots per circuit
        circs_per_input: Number of circuits per input
        n_samples: Number of input samples
        n_processes: Number of processes to use (defaults to CPU count)

    Returns:
        tuple[float, float]: Mean and standard deviation of fidelity
    """
    n_data_qubits = 2*n_trgts + 1
    n_samples = min(n_samples, 2**n_data_qubits)

    n_fanout_errors = get_fanout_error_probs(n_trgts=n_trgts, p2=10*p_err)
    telecnot_errors = get_telecnot_error_probs(n_trgts=1, p2=10*p_err)
    pre_teletoffoli_errors = get_pre_teletoffoli_error_probs(n_trgts=1, p2=10*p_err)

    print('Constructing circuits')

    # Prepare arguments for parallel processing
    eval_args = [
        (
            input_bitstr,
            n_trgts,
            p_err,
            shots_per_circ,
            circs_per_input,
            n_fanout_errors,
            telecnot_errors,
            pre_teletoffoli_errors,
        )
        for input_bitstr in sample_bitstrings(n_data_qubits, n_samples)
    ]

    # Use number of processes specified or default to CPU count
    n_processes = n_processes if n_processes is not None else cpu_count()
    print(f'Using {n_processes} processes')

    # Process evaluations in parallel with progress bar
    fids = []
    with Pool(processes=n_processes) as pool:
        for fid in tqdm(pool.imap(evaluate_single_input_telegate, eval_args), total=n_samples, desc="Processing inputs"):
            fids.append(fid)

    mean_fid = float(np.mean(fids))
    stddev_fid = float(np.std(fids))

    return mean_fid, stddev_fid
