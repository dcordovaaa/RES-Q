import streamlit as st
from streamlit_folium import st_folium
from src.geo.graph_engine import PiuraStochasticGraph
from src.geo.pruning import TopologicalPruner
from src.quantum.qubo_builder import QUBOBuilder
from src.quantum.ising_mapper import IsingMapper
from src.quantum.qaoa_solver import QAOASolver
from src.quantum.vqe_solver import VQESolver
from src.classical.ilp_solver import ILPSolver
from src.classical.sa_solver import SimulatedAnnealingSolver
from src.classical.greedy_solver import GreedyClassicalSolver
from src.postprocess.decoder import SolutionDecoder
from src.ui.plots import (
    build_piura_scenario_map,
    build_operational_dispatch_map,
    plot_qubo_heatmap,
    plot_energy_convergence,
    plot_top_bitstrings,
    plot_benchmark_comparison,
    render_circuit_matplotlib,
)


def run_dashboard() -> None:
    st.set_page_config(
        page_title="RES-Q | Optimizacion Logistica Cuantica COER Piura",
        page_icon="🌊",
        layout="wide",
    )

    st.title("🌊 RES-Q: Sistema Espacial y Optimización Logística Cuántica (COER Piura)")
    st.caption(
        "Pre-posicionamiento y Despacho de Bienes de Ayuda Humanitaria (BAH) en la Ventana Crítica de 72 Horas "
        "mediante Algoritmos Cuánticos Variacionales (QAOA & VQE sobre Qiskit)"
    )

    engine = PiuraStochasticGraph()
    pruner = TopologicalPruner(engine)
    builder = QUBOBuilder(engine)
    decoder = SolutionDecoder(engine)

    # =========================================================================
    # BARRA LATERAL: PARAMETROS ESTOCASTICOS Y HARDWARE CUANTICO
    # =========================================================================
    with st.sidebar:
        st.header("⚙️ Parámetros COER Piura")

        st.subheader("1. Escenarios Hidrometeorológicos")
        w_s1 = st.slider(
            "Peso s1: Ruptura Alto Piura (PE-1NR)", 0.0, 1.0, 0.35, 0.05
        )
        w_s2 = st.slider(
            "Peso s2: Aislamiento Sierra (PE-1NJ)", 0.0, 1.0, 0.35, 0.05
        )
        w_s3 = st.slider(
            "Peso s3: Inundación Bajo Piura (PE-1N)", 0.0, 1.0, 0.30, 0.05
        )
        probs_pi = {
            "s1_alto_piura": w_s1,
            "s2_sierra_piurana": w_s2,
            "s3_bajo_piura": w_s3,
        }

        st.subheader("2. Restricciones Logísticas (CFLP/MCLP)")
        p_limit = st.slider(
            "Techo Presupuestal de Almacenes (p)", min_value=1, max_value=4, value=3
        )
        t_max = st.slider(
            "Ventana Crítica de Supervivencia T_max (h)",
            min_value=24.0,
            max_value=72.0,
            value=48.0,
            step=4.0,
        )
        lambda_social = st.slider(
            "Prioridad de Equidad Territorial (λ)",
            min_value=1.0,
            max_value=4.0,
            value=2.5,
            step=0.25,
        )

        st.subheader("3. Configuración del Motor Cuántico")
        quantum_algo = st.selectbox(
            "Algoritmo Variacional", ["QAOA (Quantum Approximate Optimization)", "VQE (Variational Quantum Eigensolver)"]
        )
        reps_layers = st.slider("Capas del Ansatz (p / reps)", 1, 3, 2)
        mixer_or_ent = st.selectbox(
            "Mezclador QAOA / Entrelazamiento VQE",
            ["X-Mixer / Linear", "XY-Mixer / Circular"],
        )
        optimizer_name = st.selectbox("Optimizador Clásico Híbrido", ["COBYLA", "SPSA"])
        shots = st.select_slider("Número de Shots (AerSimulator)", options=[512, 1024, 2048], value=1024)
        include_slack = st.checkbox("Incluir Qubits de Holgura Logarítmica (r_l)", value=False)

        run_btn = st.button("⚛️ Ejecutar Optimización Cuántica y Benchmarking", type="primary", use_container_width=True)

    # =========================================================================
    # PRE-PROCESAMIENTO Y PODA TOPOLOGICA EN TIEMPO REAL
    # =========================================================================
    spec = pruner.prune_network(
        probabilidades_pi=probs_pi,
        t_max=t_max,
        max_active_y_qubits=3,
        include_slack_p=include_slack,
        p_limit=p_limit,
    )
    qubo = builder.build_qubo(spec, p_limit=p_limit, lambda_social=lambda_social)
    ising = IsingMapper.qubo_to_ising(qubo)

    # Ejecutar en el primer arranque o cuando el usuario presiona el boton
    if run_btn or "resq_results" not in st.session_state:
        with st.spinner("Ejecutando circuito cuántico en AerSimulator y benchmarking clásico..."):
            if quantum_algo.startswith("QAOA"):
                m_type = "XY" if "XY" in mixer_or_ent else "X"
                q_solver = QAOASolver(
                    reps_p=reps_layers,
                    mixer_type=m_type,
                    optimizer_name=optimizer_name,
                    maxiter=50,
                    shots=shots,
                )
            else:
                ent_type = "circular" if "Circular" in mixer_or_ent else "linear"
                q_solver = VQESolver(
                    reps=reps_layers,
                    entanglement=ent_type,
                    optimizer_name=optimizer_name,
                    maxiter=50,
                    shots=shots,
                )

            res_q = q_solver.solve(qubo, ising)
            res_ilp = ILPSolver(engine).solve(qubo, p_limit=p_limit)
            res_sa = SimulatedAnnealingSolver(num_sweeps=220).solve(qubo)
            res_greedy = GreedyClassicalSolver(engine).solve(qubo, p_limit=p_limit)

            plan_q = decoder.decode(
                res_q.best_bitstring, spec, algorithm_name=res_q.algorithm.split(" ")[0], t_max=t_max, p_limit=p_limit
            )
            plan_ilp = decoder.decode(
                res_ilp.best_bitstring, spec, algorithm_name="Exact ILP", t_max=t_max, p_limit=p_limit
            )
            plan_sa = decoder.decode(
                res_sa.best_bitstring, spec, algorithm_name="Simulated Annealing", t_max=t_max, p_limit=p_limit
            )
            plan_greedy = decoder.decode(
                res_greedy.best_bitstring, spec, algorithm_name="Heurística Greedy", t_max=t_max, p_limit=p_limit
            )

            st.session_state["resq_results"] = {
                "res_q": res_q,
                "res_ilp": res_ilp,
                "res_sa": res_sa,
                "res_greedy": res_greedy,
                "plan_q": plan_q,
                "plan_ilp": plan_ilp,
                "plan_sa": plan_sa,
                "plan_greedy": plan_greedy,
            }

    results = st.session_state["resq_results"]
    res_q = results["res_q"]
    res_ilp = results["res_ilp"]
    res_sa = results["res_sa"]
    res_greedy = results["res_greedy"]
    plan_q = results["plan_q"]
    plan_ilp = results["plan_ilp"]
    plan_sa = results["plan_sa"]
    plan_greedy = results["plan_greedy"]

    # =========================================================================
    # FILA SUPERIOR DE KPIS
    # =========================================================================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Qubits Lógicos Activos (Tras Poda)",
        f"{spec.total_qubits} Qubits",
        f"-{spec.porcentaje_reduccion:.1f}% reducción ({spec.qubits_ahorrados} ahorrados)",
    )
    k2.metric(
        f"Cobertura BAH en <={t_max:.0f}h ({plan_q.algorithm_name})",
        f"{plan_q.porcentaje_cobertura_demanda:.1f}%",
        f"{plan_q.porcentaje_cobertura_demanda - plan_greedy.porcentaje_cobertura_demanda:+.1f}% vs Greedy",
    )
    k3.metric(
        "Distritos Atendidos en Ventana Crítica",
        f"{plan_q.distritos_cubiertos} / {plan_q.total_distritos}",
        "Equidad Territorial Activa",
    )
    k4.metric(
        "Energía del Estado Fundamental",
        f"{res_q.best_qubo_energy:.2f}",
        f"ILP Ref: {res_ilp.best_qubo_energy:.2f}",
    )

    # =========================================================================
    # 4 PESTANAS OPERATIVAS DE RES-Q
    # =========================================================================
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🗺️ 1. Red Vial de Piura y Poda Topológica",
            "⚛️ 2. Laboratorio Cuántico (QUBO, Ising y Circuito)",
            "🚁 3. Despacho Humanitario 72h (COER / INDECI)",
            "⚖️ 4. Benchmarking: Cuántico vs. Clásico (ILP, SA, Greedy)",
        ]
    )

    # -------------------------------------------------------------------------
    # PESTANA 1: ESCENARIOS VIALES Y PODA TOPOLOGICA
    # -------------------------------------------------------------------------
    with tab1:
        col_m, col_info = st.columns([2, 1])
        with col_m:
            esc_vista = st.selectbox(
                "Inspeccionar Estado Físico del Grafo por Escenario:",
                [
                    ("s1_alto_piura", "Escenario s1: Ruptura Eje Alto Piura (PE-1NR - Puentes Independencia/Carrasquillo)"),
                    ("s2_sierra_piurana", "Escenario s2: Aislamiento Sierra Piurana (PE-1NJ - Pusmalca/Canchaque)"),
                    ("s3_bajo_piura", "Escenario s3: Inundación Bajo Piura (PE-1N - Desborde Río Piura)"),
                ],
                format_func=lambda x: x[1],
            )
            mapa_esc = build_piura_scenario_map(engine, esc_vista[0])
            st_folium(mapa_esc, width=None, height=460, returned_objects=[])

        with col_info:
            st.subheader("✂️ Poda Topológica Pre-QPU")
            st.markdown(
                f"""
                * **Variables Binarias Originales:** `{spec.total_vars_originales}` ($|J| + |I|\\times|J| + \\text{{slacks}}$)
                * **Qubits de Apertura ($x_j$):** `{spec.num_qubits_x}`
                * **Qubits de Enlaces Críticos ($y_{{ij}}$):** `{spec.num_qubits_y_activos}`
                * **Qubits de Holgura ($r_l$):** `{spec.num_qubits_slack_p}`
                * **Total Qubits en QPU/Simulador:** `{spec.total_qubits}`
                * **Espacio de Hilbert ($2^N$):** `{2**spec.total_qubits:,}` estados en superposición
                """
            )
            st.info(
                "Los arcos con α = 0 (puentes colapsados en rojo) o tiempo > T_max se fijan a "
                "y_ij = 0 antes de compilar el circuito cuántico, ahorrando más del 75% de qubits."
            )

    # -------------------------------------------------------------------------
    # PESTANA 2: LABORATORIO CUANTICO
    # -------------------------------------------------------------------------
    with tab2:
        c_q1, c_q2 = st.columns(2)
        with c_q1:
            st.plotly_chart(plot_qubo_heatmap(qubo), use_container_width=True)
        with c_q2:
            st.plotly_chart(
                plot_energy_convergence(res_q, exact_energy=res_ilp.best_qubo_energy),
                use_container_width=True,
            )

        c_q3, c_q4 = st.columns([1, 1])
        with c_q3:
            st.plotly_chart(plot_top_bitstrings(res_q), use_container_width=True)
        with c_q4:
            st.subheader("🔬 Calibración del Hamiltoniano de Ising")
            st.markdown(
                f"""
                * **Cota Superior $\\Delta H_{{\\text{{costo}}}}^{{\\max}}$:** `{qubo.delta_h_max:.2f}`
                * **Penalización Cobertura ($A$):** `{qubo.penalties['A_cobertura']:.2f}`
                * **Penalización $p$-Centros ($B$):** `{qubo.penalties['B_p_centros']:.2f}`
                * **Penalización Capacidad ($C$):** `{qubo.penalties['C_capacidad']:.2f}`
                * **Penalización Acoplamiento ($D$):** `{qubo.penalties['D_acoplamiento']:.2f}`
                * **Términos de Pauli ($Z_i, Z_i Z_j$):** `{len(ising.operator)}` operadores
                * **Profundidad Transpilada:** `{res_q.transpiled_depth}` | **Compuertas:** `{res_q.gate_counts}`
                """
            )

        with st.expander("📐 Ver Diagrama del Circuito Cuántico Ejecutado en Qiskit", expanded=True):
            st.pyplot(render_circuit_matplotlib(res_q))

    # -------------------------------------------------------------------------
    # PESTANA 3: PLAN DE DESPACHO HUMANITARIO 72H
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader(f"📦 Plan Operativo de Respuesta Humanitaria — {plan_q.algorithm_name}")
        st.success(
            f"**Almacenes Adelantados Habilitados ({len(plan_q.almacenes_activos_nombres)}/{p_limit}):** "
            + " | ".join(plan_q.almacenes_activos_nombres)
        )

        col_map_op, col_tbl_op = st.columns([3, 2])
        with col_map_op:
            mapa_op = build_operational_dispatch_map(engine, plan_q)
            st_folium(mapa_op, width=None, height=450, returned_objects=[])

        with col_tbl_op:
            st.markdown("#### Detalle de Asignación por Distrito Damnificado")
            rows_asig = []
            for a in plan_q.asignaciones:
                rows_asig.append(
                    {
                        "Distrito": a.nombre_demanda,
                        "Subregión": a.subregion,
                        "BAH (Ton)": a.demanda_ton,
                        "Almacén Asignado": a.nombre_almacen,
                        "Tiempo (h)": round(a.tiempo_esperado_h, 1) if a.cubierto_en_72h else "Cortado",
                        "Estado <=T_max": "✅ Cubierto" if a.cubierto_en_72h else "❌ Desatendido",
                    }
                )
            st.dataframe(rows_asig, use_container_width=True, hide_index=True)

    # -------------------------------------------------------------------------
    # PESTANA 4: BENCHMARKING CIENTIFICO (CUANTICO VS CLASICO)
    # -------------------------------------------------------------------------
    with tab4:
        st.subheader("⚖️ Evaluación Comparativa: Equidad Territorial vs. Miopía Voraz (Greedy)")
        all_plans = [plan_q, plan_ilp, plan_sa, plan_greedy]
        st.plotly_chart(plot_benchmark_comparison(all_plans), use_container_width=True)

        rows_bench = [
            {
                "Algoritmo": res_q.algorithm,
                "Energía QUBO": round(res_q.best_qubo_energy, 2),
                "Cobertura Demanda (%)": round(plan_q.porcentaje_cobertura_demanda, 1),
                "Cobertura Población (%)": round(plan_q.porcentaje_cobertura_poblacion, 1),
                "Distritos Cubiertos": f"{plan_q.distritos_cubiertos}/{plan_q.total_distritos}",
                "Almacenes Activos": ", ".join(plan_q.almacenes_activos_ids),
                "Tiempo Cómputo (s)": round(res_q.execution_time_s, 3),
            },
            {
                "Algoritmo": res_ilp.algorithm,
                "Energía QUBO": round(res_ilp.best_qubo_energy, 2),
                "Cobertura Demanda (%)": round(plan_ilp.porcentaje_cobertura_demanda, 1),
                "Cobertura Población (%)": round(plan_ilp.porcentaje_cobertura_poblacion, 1),
                "Distritos Cubiertos": f"{plan_ilp.distritos_cubiertos}/{plan_ilp.total_distritos}",
                "Almacenes Activos": ", ".join(plan_ilp.almacenes_activos_ids),
                "Tiempo Cómputo (s)": round(res_ilp.execution_time_s, 4),
            },
            {
                "Algoritmo": res_sa.algorithm,
                "Energía QUBO": round(res_sa.best_qubo_energy, 2),
                "Cobertura Demanda (%)": round(plan_sa.porcentaje_cobertura_demanda, 1),
                "Cobertura Población (%)": round(plan_sa.porcentaje_cobertura_poblacion, 1),
                "Distritos Cubiertos": f"{plan_sa.distritos_cubiertos}/{plan_sa.total_distritos}",
                "Almacenes Activos": ", ".join(plan_sa.almacenes_activos_ids),
                "Tiempo Cómputo (s)": round(res_sa.execution_time_s, 4),
            },
            {
                "Algoritmo": res_greedy.algorithm,
                "Energía QUBO": round(res_greedy.best_qubo_energy, 2),
                "Cobertura Demanda (%)": round(plan_greedy.porcentaje_cobertura_demanda, 1),
                "Cobertura Población (%)": round(plan_greedy.porcentaje_cobertura_poblacion, 1),
                "Distritos Cubiertos": f"{plan_greedy.distritos_cubiertos}/{plan_greedy.total_distritos}",
                "Almacenes Activos": ", ".join(plan_greedy.almacenes_activos_ids),
                "Tiempo Cómputo (s)": round(res_greedy.execution_time_s, 4),
            },
        ]
        st.dataframe(rows_bench, use_container_width=True, hide_index=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.markdown(f"#### 🟢 Mapa Cuántico ({plan_q.algorithm_name})")
            st_folium(
                build_operational_dispatch_map(engine, plan_q),
                width=None,
                height=380,
                key="map_bench_q",
                returned_objects=[],
            )
        with col_g2:
            st.markdown("#### 🔴 Mapa Clásico Voraz (Heurística Greedy)")
            st_folium(
                build_operational_dispatch_map(engine, plan_greedy),
                width=None,
                height=380,
                key="map_bench_greedy",
                returned_objects=[],
            )