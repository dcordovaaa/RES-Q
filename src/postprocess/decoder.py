from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
from src.geo.graph_engine import PiuraStochasticGraph
from src.geo.pruning import PrunedQuantumModelSpec


@dataclass
class DispatchAssignment:
    id_demanda: str
    nombre_demanda: str
    subregion: str
    demanda_ton: float
    poblacion: int
    id_almacen: str | None
    nombre_almacen: str
    tiempo_esperado_h: float
    cubierto_en_72h: bool


@dataclass
class LogisticsOperationalPlan:
    algorithm_name: str
    almacenes_activos_ids: List[str]
    almacenes_activos_nombres: List[str]
    asignaciones: List[DispatchAssignment]
    carga_por_almacen_ton: Dict[str, float]
    capacidad_por_almacen_ton: Dict[str, float]
    porcentaje_cobertura_demanda: float
    porcentaje_cobertura_poblacion: float
    distritos_cubiertos: int
    total_distritos: int
    distritos_desatendidos: List[str]
    costo_fijo_total: float
    tiempo_promedio_ponderado_h: float
    factible_capacidad: bool


class SolutionDecoder:
    """
    Decodifica un bitstring binario z in {0, 1}^N hacia decisiones operativas
    de localizacion (x_j) y asignacion de rutas humanitarias (y_{ij}) para el COER Piura.
    """

    def __init__(self, graph_engine: PiuraStochasticGraph):
        self.engine = graph_engine

    def decode(
        self,
        bitstring: np.ndarray | List[int],
        spec: PrunedQuantumModelSpec,
        algorithm_name: str = "QAOA",
        t_max: float = 72.0,
        p_limit: int = 3,
    ) -> LogisticsOperationalPlan:
        z = np.asarray(bitstring, dtype=int)
        n_j = spec.num_qubits_x
        m_i = len(self.engine.ids_I)

        x_vec = z[:n_j].copy()
        # Reparacion ligera post-medicion si el bitstring medido excede p_limit o es todo 0
        if int(np.sum(x_vec)) == 0:
            x_vec[0] = 1
        elif int(np.sum(x_vec)) > p_limit:
            # Conservar los p_limit activos con mayor cobertura territorial
            activos = [j for j in range(n_j) if x_vec[j] == 1]
            activos.sort(
                key=lambda j: int(np.sum(spec.matriz_mascara_viable[:, j])),
                reverse=True,
            )
            x_vec[:] = 0
            for j in activos[:p_limit]:
                x_vec[j] = 1

        almacenes_activos_idx = [j for j in range(n_j) if x_vec[j] == 1]
        almacenes_activos_ids = [self.engine.ids_J[j] for j in almacenes_activos_idx]
        almacenes_activos_nombres = [
            self.engine.candidatos_J[j]["nombre"] for j in almacenes_activos_idx
        ]

        carga_por_almacen: Dict[str, float] = {
            id_j: 0.0 for id_j in self.engine.ids_J
        }
        capacidad_por_almacen: Dict[str, float] = {
            c["id"]: float(c["capacidad_ton_cj"]) for c in self.engine.candidatos_J
        }

        asignaciones: List[DispatchAssignment] = []
        toneladas_cubiertas = 0.0
        toneladas_totales = 0.0
        poblacion_cubierta = 0
        poblacion_total = 0
        distritos_cubiertos = 0
        distritos_desatendidos: List[str] = []
        suma_tiempo_ton = 0.0

        for i_idx, dem in enumerate(self.engine.demandas_I):
            id_i = dem["id"]
            d_ton = float(dem["demanda_ton_di"])
            pob = int(dem["poblacion_critica"])
            toneladas_totales += d_ton
            poblacion_total += pob

            # Buscar almacenes activos que cubran al distrito i dentro de las 72h
            js_candidatos = [
                j
                for j in almacenes_activos_idx
                if spec.matriz_mascara_viable[i_idx, j]
                and spec.t_tilde_esperado[i_idx, j] <= t_max
            ]

            j_elegido: int | None = None
            if js_candidatos:
                # Priorizar el almacen activo con menor tiempo esperado y holgura de capacidad
                js_con_cap = [
                    j
                    for j in js_candidatos
                    if carga_por_almacen[self.engine.ids_J[j]] + d_ton
                    <= capacidad_por_almacen[self.engine.ids_J[j]]
                ]
                pool = js_con_cap if js_con_cap else js_candidatos
                pool.sort(key=lambda j: spec.t_tilde_esperado[i_idx, j])
                j_elegido = pool[0]

            if j_elegido is not None:
                id_j = self.engine.ids_J[j_elegido]
                nom_j = self.engine.candidatos_J[j_elegido]["nombre"]
                t_esp = float(spec.t_tilde_esperado[i_idx, j_elegido])
                carga_por_almacen[id_j] += d_ton
                toneladas_cubiertas += d_ton
                poblacion_cubierta += pob
                distritos_cubiertos += 1
                suma_tiempo_ton += d_ton * t_esp
                cubierto = True
            else:
                id_j = None
                nom_j = "SIN COBERTURA (<72h Violado / Ruta Cortada)"
                t_esp = 999.0
                cubierto = False
                distritos_desatendidos.append(dem["nombre"])

            asignaciones.append(
                DispatchAssignment(
                    id_demanda=id_i,
                    nombre_demanda=dem["nombre"],
                    subregion=dem["subregion"],
                    demanda_ton=d_ton,
                    poblacion=pob,
                    id_almacen=id_j,
                    nombre_almacen=nom_j,
                    tiempo_esperado_h=t_esp,
                    cubierto_en_72h=cubierto,
                )
            )

        costo_fijo_total = sum(
            float(self.engine.candidatos_J[j]["costo_fijo_fj"])
            for j in almacenes_activos_idx
        )
        pct_demanda = (toneladas_cubiertas / max(1.0, toneladas_totales)) * 100.0
        pct_pob = (poblacion_cubierta / max(1, poblacion_total)) * 100.0
        t_prom = (
            suma_tiempo_ton / toneladas_cubiertas if toneladas_cubiertas > 0 else 999.0
        )

        factible_cap = all(
            carga_por_almacen[id_j] <= capacidad_por_almacen[id_j] + 1e-6
            for id_j in almacenes_activos_ids
        )

        return LogisticsOperationalPlan(
            algorithm_name=algorithm_name,
            almacenes_activos_ids=almacenes_activos_ids,
            almacenes_activos_nombres=almacenes_activos_nombres,
            asignaciones=asignaciones,
            carga_por_almacen_ton=carga_por_almacen,
            capacidad_por_almacen_ton=capacidad_por_almacen,
            porcentaje_cobertura_demanda=pct_demanda,
            porcentaje_cobertura_poblacion=pct_pob,
            distritos_cubiertos=distritos_cubiertos,
            total_distritos=m_i,
            distritos_desatendidos=distritos_desatendidos,
            costo_fijo_total=costo_fijo_total,
            tiempo_promedio_ponderado_h=t_prom,
            factible_capacidad=factible_cap,
        )