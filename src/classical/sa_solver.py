import time
from typing import List
import numpy as np
from src.quantum.qubo_builder import QUBOModel
from src.classical.ilp_solver import ClassicalSolverResult


class SimulatedAnnealingSolver:
    """
    Metaheuristica clasica de Recocido Simulado (Simulated Annealing)
    con enfriamiento geometrico y criterio de aceptacion de Metropolis-Hastings.
    """

    def __init__(
        self,
        num_sweeps: int = 250,
        t_init: float = 25.0,
        t_final: float = 0.05,
        seed: int = 42,
    ):
        self.num_sweeps = num_sweeps
        self.t_init = t_init
        self.t_final = t_final
        self.seed = seed

    def solve(self, qubo: QUBOModel) -> ClassicalSolverResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        N = qubo.num_qubits
        n_j = qubo.spec.num_qubits_x

        current_z = rng.integers(0, 2, size=N)
        current_e = qubo.evaluate_bitstring(current_z)

        best_z = current_z.copy()
        best_e = current_e
        energy_trace: List[float] = [float(best_e)]

        temperatures = np.geomspace(self.t_init, self.t_final, self.num_sweeps)

        for temp in temperatures:
            # Proponer flip de 1 qubit aleatorio
            flip_q = int(rng.integers(0, N))
            candidate_z = current_z.copy()
            candidate_z[flip_q] = 1 - candidate_z[flip_q]

            candidate_e = qubo.evaluate_bitstring(candidate_z)
            delta_e = candidate_e - current_e

            if delta_e < 0 or rng.random() < np.exp(-delta_e / max(1e-6, temp)):
                current_z = candidate_z
                current_e = candidate_e
                if current_e < best_e:
                    best_e = current_e
                    best_z = current_z.copy()

            energy_trace.append(float(best_e))

        selected_x = [j for j in range(n_j) if best_z[j] == 1]
        elapsed = time.perf_counter() - t0

        return ClassicalSolverResult(
            algorithm="Simulated Annealing (Metropolis Clasico)",
            best_bitstring=best_z,
            best_qubo_energy=float(best_e),
            selected_x_indices=selected_x,
            execution_time_s=elapsed,
            metadata={"energy_trace": energy_trace},
        )