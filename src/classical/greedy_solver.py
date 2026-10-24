import time
import numpy as np
from src.geo.graph_engine import PiuraStochasticGraph
from src.quantum.qubo_builder import QUBOModel
from src.classical.ilp_solver import ClassicalSolverResult


class GreedyClassicalSolver:
    """
    Heuristica Voraz (Greedy) tradicional:
    Selecciona secuencialmente los p almacenes con mejor ratio nominal
    (Capacidad / Costo Fijo) en los ejes urbanos principales (Costa/Capital),
    sin anticipar de forma global los cortes estocasticos de puentes en la periferia.
    """

    def __init__(self, graph_engine: PiuraStochasticGraph):
        self.engine = graph_engine

    def solve(self, qubo: QUBOModel, p_limit: int = 3) -> ClassicalSolverResult:
        t0 = time.perf_counter()
        spec = qubo.spec
        N = qubo.num_qubits
        n_j = spec.num_qubits_x

        # Score voraz miope: prioriza alta capacidad y bajo costo fijo en la costa
        greedy_scores = []
        for j_idx, cand in enumerate(self.engine.candidatos_J):
            cap = float(cand["capacidad_ton_cj"])
            costo = float(cand["costo_fijo_fj"])
            bono_urbano = 1.35 if cand["es_primario"] else 0.85
            score = (cap / costo) * bono_urbano
            greedy_scores.append((score, j_idx))

        greedy_scores.sort(reverse=True, key=lambda item: item[0])
        chosen_x = [j_idx for _, j_idx in greedy_scores[:p_limit]]

        z_greedy = np.zeros(N, dtype=int)
        for j_idx in chosen_x:
            z_greedy[spec.x_var_map[j_idx]] = 1

        # Activar qubits y_ij cuyos almacenes asociados hayan sido abiertos por Greedy
        for (i_idx, j_idx), q_y in spec.y_active_map.items():
            if j_idx in chosen_x:
                z_greedy[q_y] = 1

        greedy_energy = qubo.evaluate_bitstring(z_greedy)
        elapsed = time.perf_counter() - t0

        return ClassicalSolverResult(
            algorithm="Heuristica Greedy (Voraz Tradicional)",
            best_bitstring=z_greedy,
            best_qubo_energy=float(greedy_energy),
            selected_x_indices=chosen_x,
            execution_time_s=elapsed,
            metadata={"criterio": "Max Capacidad / Min Costo Nominal (Miope a Cortes)"},
        )