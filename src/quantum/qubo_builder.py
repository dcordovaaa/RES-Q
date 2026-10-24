from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np
from src.geo.graph_engine import PiuraStochasticGraph
from src.geo.pruning import PrunedQuantumModelSpec


@dataclass
class QUBOModel:
    """
    Representa el modelo QUBO E(z) = z^T Q z + const_offset sobre el vector
    de qubits activos z in {0, 1}^N.
    """
    Q: np.ndarray
    const_offset: float
    num_qubits: int
    penalties: Dict[str, float]
    delta_h_max: float
    var_labels: List[str]
    spec: PrunedQuantumModelSpec

    def evaluate_bitstring(self, z: np.ndarray | List[int]) -> float:
        """
        Evalua la energia total H_QUBO(z) = z^T Q z + const_offset para un vector binario z.
        """
        z_vec = np.asarray(z, dtype=float)
        return float(z_vec @ self.Q @ z_vec + self.const_offset)


class QUBOBuilder:
    """
    Compilador matematico que ensambla el Hamiltoniano QUBO unificando CFLP y MCLP:
    H_QUBO = H_costo + A * P_cobertura + B * P_{p-centros} + C * P_capacidad + D * P_acoplamiento
    """

    def __init__(self, graph_engine: PiuraStochasticGraph):
        self.engine = graph_engine

    def build_qubo(
        self,
        spec: PrunedQuantumModelSpec,
        p_limit: int = 3,
        lambda_social: float = 2.2,
        penalty_scale_factor: float = 1.50,
    ) -> QUBOModel:
        """
        Construye la matriz simetrica Q de dimension (N x N) y calibra automaticamente
        los multiplicadores de penalizacion {A, B, C, D} en [1.25 * Delta_H_max, 1.75 * Delta_H_max].
        """
        N = spec.total_qubits
        Q = np.zeros((N, N), dtype=float)
        const_offset = 0.0

        n_j = spec.num_qubits_x
        m_i = len(self.engine.ids_I)

        # Etiquetas legibles para trazabilidad de cada qubit q
        var_labels: List[str] = [""] * N
        for j_idx, q_idx in spec.x_var_map.items():
            var_labels[q_idx] = f"x_{self.engine.ids_J[j_idx]}"
        for (i_idx, j_idx), q_idx in spec.y_active_map.items():
            var_labels[q_idx] = f"y_{self.engine.ids_I[i_idx]}_{self.engine.ids_J[j_idx]}"

        slack_start = n_j + spec.num_qubits_y_activos
        for l_idx in range(spec.num_qubits_slack_p):
            var_labels[slack_start + l_idx] = f"r_slack_{l_idx}"

        # ------------------------------------------------------------------
        # 1. Termino de Costo Operativo, Impedancia Vial y Beneficio MCLP
        # ------------------------------------------------------------------
        # Normalizamos costos a escala adimensional [1..10] para estabilidad numerica
        costos_fijos = np.array(
            [float(c["costo_fijo_fj"]) for c in self.engine.candidatos_J]
        ) / 10.0
        capacidades_cj = np.array(
            [float(c["capacidad_ton_cj"]) for c in self.engine.candidatos_J]
        )
        demandas_di = np.array(
            [float(d["demanda_ton_di"]) for d in self.engine.demandas_I]
        )

        # Evaluar beneficio territorial (equidad hacia zonas vulnerables y aisladas)
        beneficio_cobertura_j = np.zeros(n_j, dtype=float)
        impedancia_ponderada_j = np.zeros(n_j, dtype=float)

        for i_idx in range(m_i):
            d_i = demandas_di[i_idx] / 25.0  # Escala de tonelaje
            js_viables = spec.y_fixed_Fallback[i_idx]
            # Si una zona tiene pocos almacenes capaces de llegar en <72h, su cobertura es mas valiosa
            factor_vulnerabilidad = 2.0 if len(js_viables) <= 2 else 1.0
            for j_idx in js_viables:
                t_ij = spec.t_tilde_esperado[i_idx, j_idx]
                beneficio_cobertura_j[j_idx] += (
                    lambda_social * factor_vulnerabilidad * d_i
                )
                impedancia_ponderada_j[j_idx] += (d_i * t_ij) / 40.0

        # Agregar terminos diagonales lineales de x_j (como x_j^2 = x_j)
        delta_components: List[float] = []
        for j_idx, q_x in spec.x_var_map.items():
            c_lineal_x = (
                costos_fijos[j_idx]
                + impedancia_ponderada_j[j_idx]
                - beneficio_cobertura_j[j_idx]
            )
            Q[q_x, q_x] += c_lineal_x
            delta_components.append(abs(c_lineal_x))

        # Agregar terminos lineales de los qubits de asignacion critica y_ij
        for (i_idx, j_idx), q_y in spec.y_active_map.items():
            d_i = demandas_di[i_idx] / 25.0
            t_ij = spec.t_tilde_esperado[i_idx, j_idx]
            c_lineal_y = (d_i * t_ij) / 35.0 - (lambda_social * 1.35 * d_i)
            Q[q_y, q_y] += c_lineal_y
            delta_components.append(abs(c_lineal_y))

        # Calibracion automatica de {A, B, C, D} in [1.25 * Delta_H_max, 1.75 * Delta_H_max]
        delta_h_max = max(delta_components) if delta_components else 5.0
        scale = float(np.clip(penalty_scale_factor, 1.25, 1.75))
        base_penalty = scale * delta_h_max

        A_cov = base_penalty * 1.10
        B_pcentros = base_penalty * 1.15
        C_cap = base_penalty * 0.95
        D_acoplamiento = base_penalty * 1.05

        penalties = {
            "A_cobertura": A_cov,
            "B_p_centros": B_pcentros,
            "C_capacidad": C_cap,
            "D_acoplamiento": D_acoplamiento,
        }

        # ------------------------------------------------------------------
        # 2. Penalizacion de Cobertura Regional (A * P_cobertura)
        # ------------------------------------------------------------------
        # Garantiza que cada subregion critica (Bajo Piura, Alto Piura, Sierra) tenga al menos
        # un almacen viable habilitado dentro de las 72h.
        for i_idx in range(m_i):
            js_viables = spec.y_fixed_Fallback[i_idx]
            if len(js_viables) == 1:
                # Cobertura obligatoria unica: penalizar (1 - x_j)^2 = 1 - x_j
                q_x = spec.x_var_map[js_viables[0]]
                Q[q_x, q_x] -= A_cov
                const_offset += A_cov
            elif len(js_viables) >= 2:
                #Tomamos los 2 mejores candidatos resilientes j1, j2 para cubrir i:
                # Penalizar que ambos esten cerrados: (1 - x_j1 - x_j2 + x_j1*x_j2)
                j1, j2 = js_viables[0], js_viables[1]
                q1, q2 = spec.x_var_map[j1], spec.x_var_map[j2]
                peso_subregion = 0.65 * A_cov
                const_offset += peso_subregion
                Q[q1, q1] -= peso_subregion
                Q[q2, q2] -= peso_subregion
                self._add_quad_symmetric(Q, q1, q2, peso_subregion)

        # ------------------------------------------------------------------
        # 3. Penalizacion de Presupuesto (B * P_{p-centros})
        #    (sum_{j} x_j + sum_{l} 2^l r_l - p)^2
        #    Como se buscan exactamente hasta p centros activos con maximo beneficio,
        #    incorporamos los bits de holgura r_l codificados en base 2.
        # ------------------------------------------------------------------
        terms_p: List[Tuple[int, float]] = []
        for j_idx, q_x in spec.x_var_map.items():
            terms_p.append((q_x, 1.0))
        for l_idx in range(spec.num_qubits_slack_p):
            q_r = slack_start + l_idx
            # Peso moderado para holgura si el usuario permite menos de p almacenes
            terms_p.append((q_r, float(2**l_idx)))

        const_offset += self._add_linear_equality_penalty(
            Q, terms=terms_p, rhs=float(p_limit), weight=B_pcentros
        )
        # Ligero incentivo en holguras para preferir usar el presupuesto disponible p
        for l_idx in range(spec.num_qubits_slack_p):
            q_r = slack_start + l_idx
            Q[q_r, q_r] += 0.35 * B_pcentros * (2**l_idx)

        # ------------------------------------------------------------------
        # 4. Penalizacion de Capacidad y Redundancia (C * P_capacidad)
        #    Evita que dos almacenes que compiten por el mismo corredor saturado
        #    concentren toda la carga dejando sin capacidad a la sierra/periferia.
        # ------------------------------------------------------------------
        demanda_total = float(np.sum(demandas_di))
        for j1 in range(n_j):
            for j2 in range(j1 + 1, n_j):
                # Si ambos son costeros primarios (Paita/Piura/Sullana) y no cubren sierra,
                # penalizar que acaparen simultaneamente el presupuesto cuando p es bajo
                if (
                    self.engine.candidatos_J[j1]["es_primario"]
                    and self.engine.candidatos_J[j2]["es_primario"]
                ):
                    q1, q2 = spec.x_var_map[j1], spec.x_var_map[j2]
                    self._add_quad_symmetric(Q, q1, q2, 0.30 * C_cap)

                # Penalizar pares cuya capacidad combinada sea insuficiente frente a demanda local
                if capacidades_cj[j1] + capacidades_cj[j2] < 0.55 * demanda_total:
                    q1, q2 = spec.x_var_map[j1], spec.x_var_map[j2]
                    self._add_quad_symmetric(Q, q1, q2, 0.25 * C_cap)

        # ------------------------------------------------------------------
        # 5. Penalizacion de Acoplamiento Logico (D * P_acoplamiento)
        #    P_acoplamiento = sum y_{ij} (1 - x_j) = y_{ij} - y_{ij} x_j
        #    Fuerza que ninguna ruta y_{ij} se active si el almacen x_j esta cerrado.
        # ------------------------------------------------------------------
        for (i_idx, j_idx), q_y in spec.y_active_map.items():
            q_x = spec.x_var_map[j_idx]
            Q[q_y, q_y] += D_acoplamiento
            self._add_quad_symmetric(Q, q_y, q_x, -D_acoplamiento)

        return QUBOModel(
            Q=Q,
            const_offset=float(const_offset),
            num_qubits=N,
            penalties=penalties,
            delta_h_max=float(delta_h_max),
            var_labels=var_labels,
            spec=spec,
        )

    @staticmethod
    def _add_quad_symmetric(Q: np.ndarray, q1: int, q2: int, val: float) -> None:
        if q1 == q2:
            Q[q1, q1] += val
        else:
            Q[q1, q2] += val / 2.0
            Q[q2, q1] += val / 2.0

    def _add_linear_equality_penalty(
        self,
        Q: np.ndarray,
        terms: List[Tuple[int, float]],
        rhs: float,
        weight: float,
    ) -> float:
        """
        Expande weight * (sum_k a_k z_k - rhs)^2 sobre la matriz Q (usando z_k^2 = z_k)
        y retorna la contribucion constante weight * rhs^2.
        """
        for idx1, (q1, a1) in enumerate(terms):
            # Termino cuadratico diagonal a1^2 * z1^2 + termino lineal -2 * rhs * a1 * z1
            Q[q1, q1] += weight * (a1 * a1 - 2.0 * rhs * a1)
            for idx2 in range(idx1 + 1, len(terms)):
                q2, a2 = terms[idx2]
                self._add_quad_symmetric(Q, q1, q2, weight * 2.0 * a1 * a2)
        return weight * (rhs * rhs)