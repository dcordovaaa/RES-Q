from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
from src.geo.graph_engine import PiuraStochasticGraph


@dataclass
class PrunedQuantumModelSpec:
    """
    Especificacion del espacio de variables reducido tras la poda topologica clasica.
    """
    num_qubits_x: int
    num_qubits_y_activos: int
    num_qubits_slack_p: int
    total_qubits: int
    total_vars_originales: int
    qubits_ahorrados: int
    porcentaje_reduccion: float
    x_var_map: Dict[int, int]                  # j_idx -> qubit_idx
    y_active_map: Dict[Tuple[int, int], int]   # (i_idx, j_idx) -> qubit_idx
    y_fixed_Fallback: Dict[int, List[int]]     # i_idx -> lista ordenada de j_idx viables
    matriz_mascara_viable: np.ndarray          # (m, n) booleana: True si j cubre a i en T_max
    t_tilde_esperado: np.ndarray               # (m, n) impedancias esperadas


class TopologicalPruner:
    """
    Implementa la Poda Topologica Clasica (Pre-procesamiento del Paso 1 de RES-Q).
    Elimina variables y_ij cuyos arcos violan T_max o colapsan bajo los escenarios activos,
    comprimiendo el circuito cuantico a los qubits estrictamente relevantes.
    """

    def __init__(self, graph_engine: PiuraStochasticGraph):
        self.engine = graph_engine

    def prune_network(
        self,
        probabilidades_pi: Dict[str, float] | None = None,
        t_max: float = 72.0,
        min_viabilidad_escenarios: float = 0.50,
        max_active_y_qubits: int = 5,
        include_slack_p: bool = True,
        p_limit: int = 3,
    ) -> PrunedQuantumModelSpec:
        """
        Ejecuta la poda topologica sobre el grafo G(S):
        1. Fija y_ij = 0 para todo par (i, j) cuya viabilidad estocastica sea menor al umbral
           o supere T_max.
        2. Identifica los pares (i, j) con mayor disputa estocastica (rutas criticas de
           contingencia entre Bajo Piura, Alto Piura y Sierra) para asignarlos a qubits explicitos,
           manteniendo el circuito en un rango tratable en tiempo real (10-12 qubits).
        """
        eval_res = self.engine.evaluate_multi_scenario_impedance(
            probabilidades_pi=probabilidades_pi, t_max=t_max
        )
        viabilidad = eval_res["viabilidad_acumulada"]
        t_tilde = eval_res["t_tilde_esperado"]

        m = len(self.engine.ids_I)
        n = len(self.engine.ids_J)

        # Total original sin poda: |J| + |I|*|J| + bits de holgura
        slack_p_bits = int(np.floor(np.log2(max(1, p_limit)))) + 1 if include_slack_p else 0
        total_original = n + (m * n) + slack_p_bits

        # Mascara de pares (i, j) viables dentro de la ventana de 72 horas
        mascara_viable = viabilidad >= min_viabilidad_escenarios

        # Garantizar que cada zona i in I tenga al menos su mejor candidato viable
        y_fixed_fallback: Dict[int, List[int]] = {}
        for i_idx in range(m):
            js_viables = [
                j_idx for j_idx in range(n) if mascara_viable[i_idx, j_idx]
            ]
            if not js_viables:
                # Si todos superan el umbral, rescatar el de menor tiempo esperado
                mejor_j = int(np.argmin(t_tilde[i_idx, :]))
                mascara_viable[i_idx, mejor_j] = True
                js_viables = [mejor_j]

            # Ordenar candidatos viables por menor tiempo esperado t_tilde
            js_viables.sort(key=lambda j: t_tilde[i_idx, j])
            y_fixed_fallback[i_idx] = js_viables

        # Asignar los primeros |J| qubits a las decisiones de apertura x_j
        x_var_map: Dict[int, int] = {j_idx: j_idx for j_idx in range(n)}
        next_qubit = n

        # Seleccionar las variables de asignacion y_ij criticas sobrevivientes a la poda
        # Priorizamos zonas con multiples candidatos viables en conflicto por rotura de puentes
        candidatos_y_criticos: List[Tuple[float, int, int]] = []
        for i_idx in range(m):
            js = y_fixed_fallback[i_idx]
            if len(js) >= 2:
                # Escoger el enlace alternativo resiliente frente al primario
                j_resiliente = js[0]
                demanda_i = float(self.engine.demandas_I[i_idx]["demanda_ton_di"])
                score_urgencia = demanda_i / max(1.0, t_tilde[i_idx, j_resiliente])
                candidatos_y_criticos.append((score_urgencia, i_idx, j_resiliente))

        candidatos_y_criticos.sort(reverse=True, key=lambda item: item[0])

        y_active_map: Dict[Tuple[int, int], int] = {}
        for _, i_idx, j_idx in candidatos_y_criticos[:max_active_y_qubits]:
            y_active_map[(i_idx, j_idx)] = next_qubit
            next_qubit += 1

        total_qubits = next_qubit + slack_p_bits
        qubits_ahorrados = total_original - total_qubits
        porcentaje_reduccion = (qubits_ahorrados / total_original) * 100.0

        return PrunedQuantumModelSpec(
            num_qubits_x=n,
            num_qubits_y_activos=len(y_active_map),
            num_qubits_slack_p=slack_p_bits,
            total_qubits=total_qubits,
            total_vars_originales=total_original,
            qubits_ahorrados=qubits_ahorrados,
            porcentaje_reduccion=porcentaje_reduccion,
            x_var_map=x_var_map,
            y_active_map=y_active_map,
            y_fixed_Fallback=y_fixed_fallback,
            matriz_mascara_viable=mascara_viable,
            t_tilde_esperado=t_tilde,
        )