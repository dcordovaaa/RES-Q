import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import networkx as nx
import numpy as np


class PiuraStochasticGraph:
    """
    Motor de Grafo Dinamico Estocastico Multi-Escenario G(S) = (V, E_s, W_s)
    para la red vial departamental de Piura (COER Piura / INDECI).
    """

    M_BARRERA = 999.0  # Sobrecarga de barrera para rutas colapsadas (t_ij^s > T_max)

    def __init__(self, config_path: str | Path = "config/piura_network.json"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        self.candidatos_J: List[Dict[str, Any]] = self.raw_data["candidatos_J"]
        self.demandas_I: List[Dict[str, Any]] = self.raw_data["demandas_I"]
        self.arcos_base: List[Dict[str, Any]] = self.raw_data["arcos_base"]
        self.escenarios_S: Dict[str, Any] = self.raw_data["escenarios_S"]

        self.ids_J: List[str] = [c["id"] for c in self.candidatos_J]
        self.ids_I: List[str] = [d["id"] for d in self.demandas_I]
        self.node_map: Dict[str, Dict[str, Any]] = {
            c["id"]: {**c, "tipo": "J"} for c in self.candidatos_J
        }
        for d in self.demandas_I:
            self.node_map[d["id"]] = {**d, "tipo": "I"}

    def build_scenario_graph(self, scenario_key: str | None = None) -> nx.Graph:
        """
        Construye el grafo G(s) para un escenario especifico s in S.
        Si scenario_key es None, construye el grafo nominal sin desastres.
        """
        G = nx.Graph()
        for node_id, attrs in self.node_map.items():
            G.add_node(node_id, **attrs)

        # Mapa de afectaciones del escenario s
        afectaciones: Dict[Tuple[str, str], Dict[str, float]] = {}
        if scenario_key and scenario_key in self.escenarios_S:
            for arco_af in self.escenarios_S[scenario_key]["arcos_afectados"]:
                par = (arco_af["u"], arco_af["v"])
                par_inv = (arco_af["v"], arco_af["u"])
                afectaciones[par] = arco_af
                afectaciones[par_inv] = arco_af

        for arco in self.arcos_base:
            u, v = arco["u"], arco["v"]
            t_base = float(arco["tiempo_base_h"])
            via = arco["via"]

            if (u, v) in afectaciones:
                af = afectaciones[(u, v)]
                alpha = int(af["alpha"])
                factor = float(af["factor_retraso"])
                tiempo_s = self.M_BARRERA if alpha == 0 else t_base * factor
            else:
                alpha = 1
                tiempo_s = t_base

            G.add_edge(
                u,
                v,
                via=via,
                tiempo_base_h=t_base,
                tiempo_s=tiempo_s,
                alpha=alpha,
                operativo=(alpha == 1 and tiempo_s < self.M_BARRERA),
            )
        return G

    def compute_shortest_times_for_scenario(
        self, scenario_key: str | None, t_max: float = 72.0
    ) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, int], List[str]]]:
        """
        Calcula la matriz de tiempos minimos t_{ij}^s y el estado de operatividad
        efectivo alpha_{ij}^s entre cada zona de demanda i in I (filas) y
        cada almacen candidato j in J (columnas).
        """
        G = self.build_scenario_graph(scenario_key)
        # Crear subgrafo solo con arcos físicamente operativos (alpha == 1)
        G_op = nx.Graph()
        for u, v, data in G.edges(data=True):
            if data["alpha"] == 1:
                G_op.add_edge(u, v, weight=data["tiempo_s"], via=data["via"])

        m = len(self.ids_I)
        n = len(self.ids_J)
        t_matrix = np.full((m, n), self.M_BARRERA, dtype=float)
        alpha_matrix = np.zeros((m, n), dtype=int)
        rutas_opt: Dict[Tuple[int, int], List[str]] = {}

        for i_idx, id_i in enumerate(self.ids_I):
            for j_idx, id_j in enumerate(self.ids_J):
                if G_op.has_node(id_j) and G_op.has_node(id_i):
                    try:
                        dist = nx.shortest_path_length(
                            G_op, source=id_j, target=id_i, weight="weight"
                        )
                        path = nx.shortest_path(
                            G_op, source=id_j, target=id_i, weight="weight"
                        )
                        t_matrix[i_idx, j_idx] = float(dist)
                        rutas_opt[(i_idx, j_idx)] = path
                        if dist <= t_max:
                            alpha_matrix[i_idx, j_idx] = 1
                    except nx.NetworkXNoPath:
                        t_matrix[i_idx, j_idx] = self.M_BARRERA
                        alpha_matrix[i_idx, j_idx] = 0
                        rutas_opt[(i_idx, j_idx)] = []

        return t_matrix, alpha_matrix, rutas_opt

    def evaluate_multi_scenario_impedance(
        self,
        probabilidades_pi: Dict[str, float] | None = None,
        t_max: float = 72.0,
        m_barrera: float = 180.0,
    ) -> Dict[str, Any]:
        """
        Evalua todos los escenarios s in S ponderados por pi_s (sum pi_s = 1).
        Retorna:
        - t_tilde_matrix: matriz de tiempos esperados \tilde{t}_{ij} con sobrecarga de barrera
        - cobertura_por_escenario: conjuntos N_i^s para cada escenario
        - matrices_por_escenario: detalle de t_{ij}^s, alpha_{ij}^s y rutas por escenario
        """
        if probabilidades_pi is None:
            probabilidades_pi = {
                k: v["probabilidad_default"] for k, v in self.escenarios_S.items()
            }

        # Normalizar probabilidades para asegurar sum_{s in S} pi_s = 1
        total_pi = sum(probabilidades_pi.values())
        if total_pi <= 0:
            probabilidades_pi = {k: 1.0 / len( self.escenarios_S ) for k in self.escenarios_S}
        else:
            probabilidades_pi = {k: v / total_pi for k, v in probabilidades_pi.items()}

        m = len(self.ids_I)
        n = len(self.ids_J)
        t_tilde_esperado = np.zeros((m, n), dtype=float)
        viabilidad_acumulada = np.zeros((m, n), dtype=float)
        detalle_escenarios: Dict[str, Any] = {}

        for s_key, pi_s in probabilidades_pi.items():
            if pi_s <= 0:
                continue
            t_s, alpha_s, rutas_s = self.compute_shortest_times_for_scenario(
                s_key, t_max=t_max
            )
            # Segun ec. (5)-(6) del informe: si t_ij^s <= T_max y alpha_ij^s == 1 usa t_ij^s,
            # de lo contrario aplica sobrecarga de barrera M_barrera
            t_efectivo_s = np.where(
                (alpha_s == 1) & (t_s <= t_max), t_s, m_barrera
            )
            t_tilde_esperado += pi_s * t_efectivo_s
            viabilidad_acumulada += pi_s * alpha_s

            # Conjunto N_i^s = {j in J | t_ij^s <= T_max and alpha_ij^s == 1}
            n_i_s: Dict[str, List[str]] = {}
            for i_idx, id_i in enumerate(self.ids_I):
                n_i_s[id_i] = [
                    self.ids_J[j_idx]
                    for j_idx in range(n)
                    if alpha_s[i_idx, j_idx] == 1
                ]

            detalle_escenarios[s_key] = {
                "pi_s": pi_s,
                "t_matrix": t_s,
                "alpha_matrix": alpha_s,
                "rutas": rutas_s,
                "N_i_s": n_i_s,
            }

        return {
            "probabilidades_pi": probabilidades_pi,
            "t_tilde_esperado": t_tilde_esperado,
            "viabilidad_acumulada": viabilidad_acumulada,
            "detalle_escenarios": detalle_escenarios,
            "t_max": t_max,
        }