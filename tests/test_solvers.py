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


def test_integracion_qaoa_supera_greedy_en_cobertura_territorial():
    """
    Demuestra que bajo una combinacion severa de los escenarios s1, s2 y s3 y
    ventana operativa exigente, el algoritmo Greedy concentra almacenes en la costa
    (Piura y Paita) dejando desatendida la sierra, mientras que QAOA encuentra
    menor energia QUBO y mayor o igual cobertura territorial.
    """
    engine = PiuraStochasticGraph()
    pruner = TopologicalPruner(engine)
    probs = {"s1_alto_piura": 0.35, "s2_sierra_piurana": 0.40, "s3_bajo_piura": 0.25}

    spec = pruner.prune_network(
        probabilidades_pi=probs,
        t_max=32.0,
        max_active_y_qubits=3,
        include_slack_p=False,
        p_limit=2,
    )
    qubo = QUBOBuilder(engine).build_qubo(spec, p_limit=2, lambda_social=2.5)
    ising = IsingMapper.qubo_to_ising(qubo)

    qaoa = QAOASolver(reps_p=2, mixer_type="X", optimizer_name="COBYLA", maxiter=45, shots=512)
    res_qaoa = qaoa.solve(qubo, ising)

    greedy = GreedyClassicalSolver(engine)
    res_greedy = greedy.solve(qubo, p_limit=2)

    decoder = SolutionDecoder(engine)
    plan_qaoa = decoder.decode(
        res_qaoa.best_bitstring, spec, algorithm_name="QAOA", t_max=32.0, p_limit=2
    )
    plan_greedy = decoder.decode(
        res_greedy.best_bitstring, spec, algorithm_name="Greedy", t_max=32.0, p_limit=2
    )

    assert len(res_qaoa.expected_energy_history) > 5
    assert res_qaoa.best_qubo_energy < res_greedy.best_qubo_energy
    assert plan_qaoa.porcentaje_cobertura_demanda >= plan_greedy.porcentaje_cobertura_demanda


def test_integracion_vqe_y_benchmark_clasico():
    """
    Verifica que VQE, ILP (Branch-and-Bound) y Simulated Annealing se ejecuten
    de extremo a extremo y generen planes de despacho validos.
    """
    engine = PiuraStochasticGraph()
    pruner = TopologicalPruner(engine)
    spec = pruner.prune_network(
        t_max=72.0, max_active_y_qubits=2, include_slack_p=False, p_limit=3
    )
    qubo = QUBOBuilder(engine).build_qubo(spec, p_limit=3)

    vqe = VQESolver(reps=1, optimizer_name="COBYLA", maxiter=35, shots=512)
    res_vqe = vqe.solve(qubo)

    ilp = ILPSolver(engine)
    res_ilp = ilp.solve(qubo, p_limit=3)

    sa = SimulatedAnnealingSolver(num_sweeps=150)
    res_sa = sa.solve(qubo)

    decoder = SolutionDecoder(engine)
    plan_vqe = decoder.decode(res_vqe.best_bitstring, spec, algorithm_name="VQE", p_limit=3)
    plan_ilp = decoder.decode(res_ilp.best_bitstring, spec, algorithm_name="ILP", p_limit=3)
    plan_sa = decoder.decode(res_sa.best_bitstring, spec, algorithm_name="SA", p_limit=3)

    assert plan_ilp.porcentaje_cobertura_demanda >= 85.0
    assert plan_vqe.distritos_cubiertos >= 7
    assert plan_sa.distritos_cubiertos >= 7