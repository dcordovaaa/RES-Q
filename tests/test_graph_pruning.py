import numpy as np
from src.geo.graph_engine import PiuraStochasticGraph
from src.geo.pruning import TopologicalPruner


def test_carga_nodos_piura():
    engine = PiuraStochasticGraph()
    assert len(engine.candidatos_J) == 5
    assert len(engine.demandas_I) == 9
    assert "J0_PAITA" in engine.ids_J
    assert "J3_CHULUCANAS" in engine.ids_J
    assert "I1_CURA_MORI" in engine.ids_I
    assert "I6_CANCHAQUE_FAIQUE" in engine.ids_I


def test_escenario_s1_ruptura_eje_alto_piura():
    """
    Verifica que bajo s1 (colapso de puentes Independencia/Carrasquillo en PE-1NR),
    el arco directo Piura -> Chulucanas quede inhabilitado (alpha = 0).
    """
    engine = PiuraStochasticGraph()
    G_s1 = engine.build_scenario_graph("s1_alto_piura")
    edge_data = G_s1.get_edge_data("J1_PIURA", "J3_CHULUCANAS")
    assert edge_data is not None
    assert edge_data["alpha"] == 0
    assert not edge_data["operativo"]


def test_escenario_s2_aislamiento_sierra_canchaque():
    """
    Verifica que bajo s2 (aluvion en Pusmalca / Agua Blanca en PE-1NJ),
    la conexion desde la costa hacia Canchaque exceda T_max o quede bloqueada,
    mientras que la base adelantada de Paimas mantiene cobertura (< 72h).
    """
    engine = PiuraStochasticGraph()
    t_s2, alpha_s2, _ = engine.compute_shortest_times_for_scenario(
        "s2_sierra_piurana", t_max=72.0
    )
    i_canchaque = engine.ids_I.index("I6_CANCHAQUE_FAIQUE")
    j_paimas = engine.ids_J.index("J4_PAIMAS")

    # Paimas debe poder abastecer a Canchaque dentro de las 72 horas por corredor andino
    assert alpha_s2[i_canchaque, j_paimas] == 1
    assert t_s2[i_canchaque, j_paimas] <= 72.0


def test_escenario_s3_inundacion_bajo_piura():
    """
    Verifica que bajo s3 (desborde del rio Piura y rotura de diques en PE-1N),
    los arcos directos de Piura a Catacaos y Cura Mori colapsen (alpha = 0),
    exigiendo abastecimiento desde nodos perifericos no inundables (Paita / Chulucanas).
    """
    engine = PiuraStochasticGraph()
    G_s3 = engine.build_scenario_graph("s3_bajo_piura")
    assert G_s3.get_edge_data("J1_PIURA", "I0_CATACAOS")["alpha"] == 0
    assert G_s3.get_edge_data("J1_PIURA", "I1_CURA_MORI")["alpha"] == 0

    t_s3, alpha_s3, _ = engine.compute_shortest_times_for_scenario(
        "s3_bajo_piura", t_max=72.0
    )
    i_curamori = engine.ids_I.index("I1_CURA_MORI")
    j_paita = engine.ids_J.index("J0_PAITA")
    assert alpha_s3[i_curamori, j_paita] == 1


def test_poda_topologica_ahorro_qubits():
    """
    Verifica que la poda topologica reduzca las 52 variables originales
    a un rango compacto (10 a 12 qubits) apto para ejecucion en tiempo real.
    """
    engine = PiuraStochasticGraph()
    pruner = TopologicalPruner(engine)
    spec = pruner.prune_network(t_max=72.0, max_active_y_qubits=5, p_limit=3)

    assert spec.total_vars_originales == 52
    assert 10 <= spec.total_qubits <= 12
    assert spec.qubits_ahorrados >= 40
    assert spec.porcentaje_reduccion > 75.0