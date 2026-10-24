from typing import Dict, List, Tuple, Any
import numpy as np
import folium
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from src.geo.graph_engine import PiuraStochasticGraph
from src.quantum.qubo_builder import QUBOModel
from src.quantum.qaoa_solver import QuantumSolverResult
from src.postprocess.decoder import LogisticsOperationalPlan


def build_piura_scenario_map(
    engine: PiuraStochasticGraph, scenario_key: str | None = None
) -> folium.Map:
    """
    Construye el mapa cartografico interactivo de Piura mostrando los almacenes candidatos J,
    los distritos de demanda I y el estado operativo de los corredores viales bajo el escenario activo.
    """
    m = folium.Map(
        location=[-5.08, -80.25],
        zoom_start=8,
        tiles="CartoDB positron",
        control_scale=True,
    )

    G = engine.build_scenario_graph(scenario_key)

    # 1. Dibujar corredores viales (Verde = Operativo, Naranja = Retraso severo, Rojo = Colapsado)
    for u, v, data in G.edges(data=True):
        node_u = engine.node_map[u]
        node_v = engine.node_map[v]
        coords = [(node_u["lat"], node_u["lon"]), (node_v["lat"], node_v["lon"])]

        if data["alpha"] == 0:
            color = "#E74C3C"  # Rojo: Puente caido / Ruta cortada
            weight = 4.5
            dash = "8, 8"
            estado_str = "COLAPSADO / INTRANSITABLE (alpha=0)"
        elif data["tiempo_s"] > data["tiempo_base_h"] * 1.1:
            color = "#F39C12"  # Naranja: Huaico / Sobrecarga vial
            weight = 3.5
            dash = None
            estado_str = f"AFECTADO (Tiempo: {data['tiempo_s']:.1f}h vs Base: {data['tiempo_base_h']:.1f}h)"
        else:
            color = "#2C3E50"  # Azul grisaceo: Operativo normal
            weight = 2.5
            dash = None
            estado_str = f"OPERATIVO ({data['tiempo_s']:.1f}h)"

        tooltip_txt = f"<b>{data['via']}</b><br>Estado: {estado_str}"
        folium.PolyLine(
            coords,
            color=color,
            weight=weight,
            opacity=0.85,
            dash_array=dash,
            tooltip=tooltip_txt,
        ).add_to(m)

    # 2. Dibujar Almacenes Candidatos J
    for cand in engine.candidatos_J:
        popup_html = (
            f"<b>{cand['nombre']}</b><br>"
            f"ID: {cand['id']}<br>"
            f"Capacidad BAH: {cand['capacidad_ton_cj']} Ton<br>"
            f"Costo Fijo: S/ {cand['costo_fijo_fj']}k"
        )
        folium.Marker(
            location=[cand["lat"], cand["lon"]],
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"Almacen Candidato: {cand['nombre']}",
            icon=folium.Icon(color="darkblue", icon="warehouse", prefix="fa"),
        ).add_to(m)

    # 3. Dibujar Nodos de Demanda Humanitaria I
    for dem in engine.demandas_I:
        folium.CircleMarker(
            location=[dem["lat"], dem["lon"]],
            radius=8,
            color="#C0392B",
            fill=True,
            fill_color="#E74C3C",
            fill_opacity=0.85,
            tooltip=(
                f"<b>Distrito: {dem['nombre']}</b><br>"
                f"Subregion: {dem['subregion']}<br>"
                f"Demanda BAH: {dem['demanda_ton_di']} Ton<br>"
                f"Poblacion Critica: {dem['poblacion_critica']:,} hab."
            ),
        ).add_to(m)

    return m


def build_operational_dispatch_map(
    engine: PiuraStochasticGraph, plan: LogisticsOperationalPlan
) -> folium.Map:
    """
    Construye el mapa de despacho humanitario (72h) proyectando almacenes habilitados (x_j=1),
    enlaces de distribucion activos (y_ij=1) y el estado de cobertura de cada distrito.
    """
    m = folium.Map(
        location=[-5.08, -80.25],
        zoom_start=8,
        tiles="CartoDB positron",
        control_scale=True,
    )

    activos_set = set(plan.almacenes_activos_ids)

    # 1. Trazar arcos de abastecimiento activos (y_ij = 1)
    for asig in plan.asignaciones:
        dem_node = engine.node_map[asig.id_demanda]
        if asig.cubierto_en_72h and asig.id_almacen is not None:
            alm_node = engine.node_map[asig.id_almacen]
            coords = [
                (alm_node["lat"], alm_node["lon"]),
                (dem_node["lat"], dem_node["lon"]),
            ]
            folium.PolyLine(
                coords,
                color="#1E8449",
                weight=4.0,
                opacity=0.9,
                tooltip=(
                    f"<b>Ruta Activa ({plan.algorithm_name})</b><br>"
                    f"Origen: {asig.nombre_almacen}<br>"
                    f"Destino: {asig.nombre_demanda}<br>"
                    f"Despacho: {asig.demanda_ton} Ton BAH<br>"
                    f"Tiempo Esperado: {asig.tiempo_esperado_h:.1f} h (<= 72h)"
                ),
            ).add_to(m)

    # 2. Dibujar Almacenes (Verde = Habilitado x_j=1, Gris = En reserva x_j=0)
    for cand in engine.candidatos_J:
        cid = cand["id"]
        es_activo = cid in activos_set
        carga = plan.carga_por_almacen_ton.get(cid, 0.0)
        cap = plan.capacidad_por_almacen_ton.get(cid, 1.0)
        color_icon = "green" if es_activo else "lightgray"
        estado_lbl = "HABILITADO (x_j = 1)" if es_activo else "CERRADO (x_j = 0)"

        folium.Marker(
            location=[cand["lat"], cand["lon"]],
            tooltip=(
                f"<b>{cand['nombre']}</b><br>"
                f"Estado: {estado_lbl}<br>"
                f"Despacho Asignado: {carga:.1f} / {cap:.1f} Ton ({(carga/cap)*100:.0f}%)"
            ),
            icon=folium.Icon(color=color_icon, icon="truck", prefix="fa"),
        ).add_to(m)

    # 3. Dibujar Distritos de Demanda (Verde = Cubierto en <=72h, Rojo = Desatendido)
    for asig in plan.asignaciones:
        dem_node = engine.node_map[asig.id_demanda]
        c_fill = "#27AE60" if asig.cubierto_en_72h else "#C0392B"
        estado_d = (
            f"CUBIERTO por {asig.nombre_almacen} ({asig.tiempo_esperado_h:.1f}h)"
            if asig.cubierto_en_72h
            else "DESATENDIDO (Violacion ventana 72h)"
        )
        folium.CircleMarker(
            location=[dem_node["lat"], dem_node["lon"]],
            radius=9,
            color="#1B2631",
            weight=2,
            fill=True,
            fill_color=c_fill,
            fill_opacity=0.9,
            tooltip=(
                f"<b>{asig.nombre_demanda}</b><br>"
                f"Estado 72h: {estado_d}<br>"
                f"Demanda: {asig.demanda_ton} Ton | Pob: {asig.poblacion:,} hab."
            ),
        ).add_to(m)

    return m


def plot_qubo_heatmap(qubo: QUBOModel) -> go.Figure:
    """
    Genera el mapa de calor interactivo de la matriz simetrica Q (QUBO).
    """
    fig = px.imshow(
        qubo.Q,
        x=qubo.var_labels,
        y=qubo.var_labels,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Matriz QUBO (H_costo + Penalizaciones A, B, C, D)",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=45, b=20))
    return fig


def plot_energy_convergence(
    q_result: QuantumSolverResult, exact_energy: float | None = None
) -> go.Figure:
    """
    Grafica la curva de convergencia de energia del bucle variacional (QAOA / VQE).
    """
    iters = list(range(1, len(q_result.expected_energy_history) + 1))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iters,
            y=q_result.expected_energy_history,
            mode="lines+markers",
            name=q_result.algorithm,
            line=dict(color="#2980B9", width=2.5),
        )
    )
    if exact_energy is not None:
        fig.add_hline(
            y=exact_energy,
            line_dash="dash",
            line_color="#27AE60",
            annotation_text=f"Ground State Exacto (ILP = {exact_energy:.2f})",
            annotation_position="bottom right",
        )
    fig.update_layout(
        title="Convergencia del Valor Esperado de Energia <H_C>",
        xaxis_title="Iteracion del Optimizador Clasico",
        yaxis_title="Energia QUBO / Ising",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def plot_top_bitstrings(q_result: QuantumSolverResult) -> go.Figure:
    """
    Histograma de los estados cuanticos (bitstrings) con mayor probabilidad de medicion.
    """
    labels = [f"|{b}> (E={e:.1f})" for b, _, e in q_result.top_measurements]
    probs = [p * 100.0 for _, p, _ in q_result.top_measurements]

    fig = go.Figure(
        go.Bar(
            x=labels,
            y=probs,
            marker_color="#8E44AD",
            text=[f"{p:.1f}%" for p in probs],
            textposition="auto",
        )
    )
    fig.update_layout(
        title=f"Distribucion de Bitstrings Medidos ({q_result.shots} shots)",
        xaxis_title="Estado Cuantico |z> y Energia Asociada",
        yaxis_title="Probabilidad de Muestreo (%)",
        height=380,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def plot_benchmark_comparison(plans: List[LogisticsOperationalPlan]) -> go.Figure:
    """
    Grafico comparativo de cobertura humanitaria (<72h) y tiempo de respuesta entre
    los motores Cuanticos (QAOA/VQE) y Clasicos (ILP, Simulated Annealing, Greedy).
    """
    algos = [p.algorithm_name for p in plans]
    cov_demanda = [p.porcentaje_cobertura_demanda for p in plans]
    cov_pob = [p.porcentaje_cobertura_poblacion for p in plans]

    fig = go.Figure(
        data=[
            go.Bar(
                name="% Demanda BAH Cubierta (<=72h)",
                x=algos,
                y=cov_demanda,
                marker_color="#27AE60",
                text=[f"{v:.1f}%" for v in cov_demanda],
                textposition="auto",
            ),
            go.Bar(
                name="% Poblacion Critica Atendida",
                x=algos,
                y=cov_pob,
                marker_color="#2980B9",
                text=[f"{v:.1f}%" for v in cov_pob],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Benchmarking: Equidad Territorial y Cobertura en Ventana de 72 Horas",
        yaxis=dict(title="Porcentaje de Cobertura (%)", range=[0, 108]),
        height=400,
        margin=dict(l=20, r=20, t=45, b=20),
    )
    return fig


def render_circuit_matplotlib(q_result: QuantumSolverResult):
    """
    Renderiza el diagrama del circuito cuantico Qiskit en una figura Matplotlib.
    """
    fig = q_result.circuit.draw(output="mpl", style="iqp", fold=28)
    return fig