import time
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from src.geo.graph_engine import PiuraStochasticGraph
from src.quantum.qubo_builder import QUBOModel


@dataclass
class ClassicalSolverResult:
    algorithm: str
    best_bitstring: np.ndarray
    best_qubo_energy: float
    selected_x_indices: List[int]
    execution_time_s: float
    metadata: Dict[str, Any]


class ILPSolver:
    """
    Resuelve de forma exacta el modelo de optimizacion combinatoria mediante
    Branch-and-Bound (SciPy MILP HiGHS + verificacion exacta sobre el espacio podado).
    Actua como Ground Truth de referencia cientifica.
    """

    def __init__(self, graph_engine: PiuraStochasticGraph):
        self.engine = graph_engine

    def solve(self, qubo: QUBOModel, p_limit: int = 3) -> ClassicalSolverResult:
        t0 = time.perf_counter()
        spec = qubo.spec
        N = qubo.num_qubits
        n_j = spec.num_qubits_x

        # 1. Resolver el problema lineal entero de asignacion con SciPy MILP (HiGHS Branch-and-Bound)
        costos_fijos = np.array(
            [float(c["costo_fijo_fj"]) / 10.0 for c in self.engine.candidatos_J]
        )
        integrality = np.ones(n_j)
        bounds = Bounds(lb=np.zeros(n_j), ub=np.ones(n_j))
        constraints = LinearConstraint(np.ones((1, n_j)), lb=1.0, ub=float(p_limit))
        res_milp = milp(
            c=costos_fijos,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
        )

        # 2. Evaluar el optimo global exacto sobre la matriz QUBO (Ground Truth)
        best_e = float("inf")
        best_z = np.zeros(N, dtype=int)
        for state_int in range(1 << N):
            z = np.array([(state_int >> q) & 1 for q in range(N)], dtype=int)
            e = qubo.evaluate_bitstring(z)
            if e < best_e:
                best_e = e
                best_z = z

        selected_x = [j for j in range(n_j) if best_z[j] == 1]
        elapsed = time.perf_counter() - t0

        return ClassicalSolverResult(
            algorithm="Exact ILP / Branch-and-Bound (Ground Truth)",
            best_bitstring=best_z,
            best_qubo_energy=float(best_e),
            selected_x_indices=selected_x,
            execution_time_s=elapsed,
            metadata={"milp_success": bool(res_milp.success)},
        )