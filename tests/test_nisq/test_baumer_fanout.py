import itertools

import numpy as np
from qiskit_aer import AerSimulator

from dqalgo.nisq.fanouts import BaumerFanoutBuilder


def test_baumer_fanout():
    for n_trgts in [4, 6, 8]:
        for ctrl_bit in [0, 1]:
            print(f"n_trgts: {n_trgts}, ctrl_bit: {ctrl_bit}")
            for init_trgt_bits in itertools.product([0, 1], repeat=n_trgts):
                builder = BaumerFanoutBuilder(n_trgts=n_trgts, ctrl_bit=ctrl_bit, init_trgt_bits=list(init_trgt_bits))
                expected_trgt_bits = [(_bit + ctrl_bit) % 2 for _bit in init_trgt_bits]
                expected_trgt_bitstr = "".join(map(str, expected_trgt_bits))
                circ = builder.build_w_fewer_cnots()
                rst = AerSimulator().run(circ).result()
                counts = rst.get_counts()
                for measurement in counts:
                    measurement = measurement.replace(" ", "")
                    # Assuming the bit string format is "ctrl trgts ancs"
                    rst_trgt_bits = measurement[:builder.n_trgts]  # middle bits are targets
                    rst_ctrl_bit = measurement[-1]  # last bit is ctrl
                    assert rst_trgt_bits == expected_trgt_bitstr, \
                        f"trgt_bits: {rst_trgt_bits}, expected_trgt_bits: {expected_trgt_bitstr}"
                    assert rst_ctrl_bit == str(ctrl_bit), \
                        f"ctrl_bit: {rst_ctrl_bit}, expected_ctrl_bit: {ctrl_bit}"


def test_baumer_fanout_stim():
    for n_trgts in [4, 6, 8]:
        for ctrl_bit in [0, 1]:
            print(f"n_trgts: {n_trgts}, ctrl_bit: {ctrl_bit}")
            for init_trgt_bits in itertools.product([0, 1], repeat=n_trgts):
                print(f"init_trgt_bits: {init_trgt_bits}")
                builder = BaumerFanoutBuilder(n_trgts=n_trgts, ctrl_bit=ctrl_bit, init_trgt_bits=list(init_trgt_bits))
                rst = builder.build_in_stim()
                rst_trgt_bits = rst["target_measurements"][::-1]
                rst_ctrl_bit = rst["control_measurement"]
                expected_trgt_bits = [(_bit + ctrl_bit) % 2 for _bit in init_trgt_bits]
                assert rst_ctrl_bit == ctrl_bit, \
                    f"ctrl_bit: {rst_ctrl_bit}, expected_ctrl_bit: {ctrl_bit}"
                assert np.all(rst_trgt_bits == expected_trgt_bits), \
                    f"trgt_bits: {rst_trgt_bits}, expected_trgt_bits: {expected_trgt_bits}"
