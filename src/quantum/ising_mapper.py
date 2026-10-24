from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp
from src.quantum.qubo_builder import QUBOModel


@dataclass
class IsingHamiltonian:
    """
    Contenedor del Hamiltoniano de Costo de Ising H_C:
    H_C = sum_q h_q Z_q + sum_{q < q'} J_{qq'} Z_q Z_{q'} + ising_offset * I
    """
    operator: SparsePauliOp
    ising_offset: float
    h_linear: np.ndarray
    J_quadratic: np.ndarray
    num_qubits: int


class IsingMapper:
    """
    Transforma un modelo QUBO E(z) = z^T Q z + const_offset en un Hamiltoniano de Ising
    mediante el mapeo afin z_q = (1 - s_q) / 2, donde s_q in {-1, +1} son los autovalores
    del operador de Pauli Z_q.
    """

    @staticmethod
    def qubo_to_ising(qubo: QUBOModel) -> IsingHamiltonian:
        """
        Realiza el cambio de base algebraico exacto desde Q (simetrica) hacia h_q y J_{qq'}:
        Para z_q = (I - Z_q)/2:
        - z_q^2 = z_q = 1/2 I - 1/2 Z_q
        - z_q z_{q'} = 1/4 I - 1/4 Z_q - 1/4 Z_{q'} + 1/4 Z_q Z_{q'} (para q != q')
        """
        N = qubo.num_qubits
        Q = qubo.Q

        ising_offset = float(qubo.const_offset)
        h = np.zeros(N, dtype=float)
        J = np.zeros((N, N), dtype=float)

        # Terminos diagonales Q_{qq} z_q
        for q in range(N):
            q_diag = float(Q[q, q])
            ising_offset += 0.5 * q_diag
            h[q] -= 0.5 * q_diag

        # Terminos fuera de la diagonal (Q_{qq'} + Q_{q'q}) z_q z_{q'} = 2 Q_{qq'} z_q z_{q'}
        for q1 in range(N):
            for q2 in range(q1 + 1, N):
                coeff_pair = float(Q[q1, q2] + Q[q2, q1])
                if abs(coeff_pair) > 1e-12:
                    ising_offset += 0.25 * coeff_pair
                    h[q1] -= 0.25 * coeff_pair
                    h[q2] -= 0.25 * coeff_pair
                    J[q1, q2] += 0.25 * coeff_pair

        # Construir lista de cadenas de Pauli para SparsePauliOp (convencion little-endian de Qiskit:
        # el qubit q corresponde al caracter en la posicion N - 1 - q de la cadena de Pauli)
        pauli_list: List[Tuple[str, float]] = []

        for q in range(N):
            if abs(h[q]) > 1e-12:
                label = ["I"] * N
                label[N - 1 - q] = "Z"
                pauli_list.append(("".join(label), float(h[q])))

        for q1 in range(N):
            for q2 in range(q1 + 1, N):
                if abs(J[q1, q2]) > 1e-12:
                    label = ["I"] * N
                    label[N - 1 - q1] = "Z"
                    label[N - 1 - q2] = "Z"
                    pauli_list.append(("".join(label), float(J[q1, q2])))

        if not pauli_list:
            pauli_list.append(("I" * N, 0.0))

        operator = SparsePauliOp.from_list(pauli_list).simplify()

        return IsingHamiltonian(
            operator=operator,
            ising_offset=ising_offset,
            h_linear=h,
            J_quadratic=J,
            num_qubits=N,
        )