import inspect
from typing import Callable

import numpy as np
from qutip import *


class QTCircuit:
    def __init__(self, n_qubits: int, n_clbits: int, init_state: Qobj | None = None):
        self.n: int = n_qubits
        if init_state is None:
            self.state: Qobj = tensor([basis(2, 0)] * n_qubits)
        else:
            assert init_state.shape == (2**n_qubits, 1), f"Initial state must be a {n_qubits}-qubit state vector"
            self.state: Qobj = init_state
        self.state = ket2dm(self.state)
        self.cregs: list = [0] * n_clbits
        self.timeline: list = []  # stores (str, list[int], any)

    def M_XX(self, q1: int, q2: int, creg_idx: int) -> None:
        self._proj_measure([q1, q2], creg_idx, 'XX')
        self.timeline.append(('MXX', [q1, q2], creg_idx))

    def M_ZZ(self, q1: int, q2: int, creg_idx: int) -> None:
        self._proj_measure([q1, q2], creg_idx, 'ZZ')
        self.timeline.append(('MZZ', [q1, q2], creg_idx))

    def _proj_measure(self, qubits: list[int], creg_idx: int, basis: str) -> None:
        assert len(basis) == len(qubits), f"Basis length {len(basis)} must match qubit length {len(qubits)}"
        I: list = [qeye(2)] * self.n
        op: list = [qeye(2)] * self.n
        paulis = [sigmax() if base == 'X' else sigmaz() for base in basis]
        for i, q in enumerate(qubits):
            op[q] = paulis[i]
        joint_op: Qobj = tensor(op)

        P_plus: Qobj = (tensor(I) + joint_op) / 2
        P_minus: Qobj = (tensor(I) - joint_op) / 2
        M0: Qobj = P_plus.sqrtm()
        M1: Qobj = P_minus.sqrtm()

        p0: float = (M0.dag() * M0 * self.state).tr().real
        p1: float = (M1.dag() * M1 * self.state).tr().real
        m: int = int(np.random.choice([0, 1], p=[p0, p1]))

        self.state = (M0 * self.state * M0.dag()) / p0 if m == 0 else (M1 * self.state * M1.dag()) / p1
        self.cregs[creg_idx] = m

    def M_X(self, q: int, creg_idx: int) -> None:
        self._proj_measure([q], creg_idx, 'X')
        self.timeline.append(('MX', [q], creg_idx))

    def M_Z(self, q: int, creg_idx: int) -> None:
        self._proj_measure([q], creg_idx, 'Z')
        self.timeline.append(('MZ', [q], creg_idx))

    def c_Z(self, q: int, condition: Callable | int) -> None:
        flag = self._process_condition(condition)
        if flag:
            print(f"c_Z({q}, {condition}) = {flag}")
            ops: list = [qeye(2)] * self.n
            ops[q] = sigmaz()
            Z_op: Qobj = tensor(ops)
            self.state = Z_op * self.state * Z_op.dag()
        self.timeline.append(('cZ', [q], condition))

    def c_X(self, q: int, condition: Callable | int) -> None:
        flag = self._process_condition(condition)
        if flag:
            print(f"c_X({q}, {condition}) = {flag}")
            ops: list = [qeye(2)] * self.n
            ops[q] = sigmax()
            X_op: Qobj = tensor(ops)
            self.state = X_op * self.state * X_op.dag()
        self.timeline.append(('cX', [q], condition))

    def _process_condition(self, condition: Callable | int) -> bool:
        if isinstance(condition, int):  # index of classical register
            return self.cregs[condition]
        elif isinstance(condition, Callable):  # lambda function
            return condition(self.cregs)
        else:
            raise ValueError(f"Invalid condition type: {type(condition)}")

    def get_state(self) -> Qobj:
        return self.state

    def draw(self) -> None:
        lines: list = [f"q[{i}]: " for i in range(self.n)]
        clines: list = [f"c[{i}]: " for i in range(len(self.cregs))]

        for op in self.timeline:
            kind: str
            qubits: list
            meta = op[2]
            kind, qubits, meta = op
            symbols: list = ['-----'] * self.n

            if kind in ('MXX', 'MZZ', 'MX', 'MZ'):
                if kind in ('MXX', 'MZZ'):
                    q1, q2 = qubits
                    gate = kind
                    symbols[q1] = f'─{gate}-'
                    symbols[q2] = f'-{gate}─'
                elif kind in ('MX', 'MZ'):
                    q = qubits[0]
                    label = 'M-X' if kind == 'MX' else 'M-Z'
                    symbols[q] = f'─{label}'
                for i in range(len(clines)):
                    clines[i] += f" {self.cregs[i]}  " if i == meta else "     "
            elif kind in ('cZ', 'cX'):
                q = qubits[0]
                label = 'c-Z' if kind == 'cZ' else 'c-X'
                symbols[q] = f'─{label}'
                # Try to extract source of condition if it's a lambda or named function
                try:
                    cond_str = inspect.getsource(meta).strip().replace('\n', '')
                except:
                    cond_str = 'cond'
                cond_str = cond_str.replace('lambda ', 'λ ')
                for i in range(len(clines)):
                    clines[i] += " ?  "  # Placeholder for condition logic

            for i in range(self.n):
                lines[i] += symbols[i]

        for l in lines:
            print(l)
        for c in clines:
            print(c)
