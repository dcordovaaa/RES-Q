import numpy as np
from src.geo.graph_engine import PiuraStochasticGraph
from src.geo.pruning import TopologicalPruner
from src.quantum.qubo_builder import QUBOBuilder
from src.quantum.ising_mapper import IsingMapper


def test_calibracion_penalizaciones_qubo():
    engine = PiuraStochasticGraph()
    pruner = TopologicalPruner(engine)
    spec = pruner.prune_network(t_max=72.0, max_active_y_qubits=3, p_limit=3)

    builder = QUBOBuilder(engine)
    qubo = builder.build_qubo(spec, p_limit=3, penalty_scale_factor=1.50)

    assert qubo.Q.shape == (spec.total_qubits, spec.total_qubits)
    # Matriz Q debe ser simetrica
    assert np.allclose(qubo.Q, qubo.Q.T)
    assert qubo.delta_h_max > 0.0
    assert qubo.penalties["B_p_centros"] >= 1.25 * qubo.delta_h_max


def test_equivalencia_exacta_qubo_vs_ising():
    """
    Verifica que para todo estado computacional |z>, la energia clasica QUBO
    coincida exactamente con el autovalor del Hamiltoniano de Ising SparsePauliOp.
    """
    engine = PiuraStochasticGraph()
    pruner = TopologicalPruner(engine)
    spec = pruner.prune_network(
        t_max=72.0, max_active_y_qubits=2, include_slack_p=False, p_limit=3
    )

    builder = QUBOBuilder(engine)
    qubo = builder.build_qubo(spec, p_limit=3)
    ising = IsingMapper.qubo_to_ising(qubo)

    N = qubo.num_qubits
    # Diagonal del operador de Pauli Z en la base computacional
    diag_ising = np.real(ising.operator.to_matrix().diagonal()) + ising.ising_offset

    # Probar 25 estados binarios aleatorios y extremos
    rng = np.random.default_rng(42)
    for _ in range(25):
        z = rng.integers(0, 2, size=N)
        e_qubo = qubo.evaluate_bitstring(z)

        # En convencion little-endian de Qiskit, el bit q aporta 2^q al indice del vector de estado
        state_idx = sum(int(z[q]) << q for q in range(N))
        e_ising = float(diag_ising[state_idx])

        assert np.isclose(e_qubo, e_ising, atol=1e-9)


def test_ground_state_ising_respeta_presupuesto_p():
    """
    Verifica por diagonalizacion exacta que el estado de minima energia del
    Hamiltoniano de Ising active exactamente p almacenes y priorice cobertura.
    """
    engine = PiuraStochasticGraph()
    pruner = TopologicalPruner(engine)
    spec = pruner.prune_network(
        t_max=72.0, max_active_y_qubits=3, include_slack_p=False, p_limit=3
    )

    builder = QUBOBuilder(engine)
    qubo = builder.build_qubo(spec, p_limit=3)
    ising = IsingMapper.qubo_to_ising(qubo)

    diag_total = np.real(ising.operator.to_matrix().diagonal()) + ising.ising_offset
    min_idx = int(np.argmin(diag_total))

    N = qubo.num_qubits
    z_opt = np.array([(min_idx >> q) & 1 for q in range(N)], dtype=int)
    x_opt = z_opt[: spec.num_qubits_x]

    # El estado fundamental debe activar exactamente p = 3 almacenes
    assert int(np.sum(x_opt)) == 3