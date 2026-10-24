import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
# pyrefly: ignore[missing-import]
from scipy.optimize import minimize
# pyrefly: ignore[missing-import]
from qiskit import QuantumCircuit, transpile
# pyrefly: ignore[missing-import]
from qiskit_aer import AerSimulator
from src.quantum.qubo_builder import QUBOModel
from src.quantum.ising_mapper import IsingHamiltonian, IsingMapper


@dataclass
class QuantumSolverResult:
    """
    Resultado estandarizado de los motores cuanticos variacionales (QAOA y VQE).
    """
    algorithm: str
    best_bitstring: np.ndarray
    best_qubo_energy: float
    expected_energy_history: List[float]
    top_measurements: List[Tuple[str, float, float]]  # (bitstring_str, prob, energy)
    optimal_params: np.ndarray
    circuit: QuantumCircuit
    transpiled_depth: int
    gate_counts: Dict[str, int]
    execution_time_s: float
    shots: int


class QAOASolver:
    """
    Implementacion del Quantum Approximate Optimization Algorithm (QAOA)
    con soporte para X-Mixer estandar y XY-Mixer conservador de subespacio.
    """

    def __init__(
        self,
        reps_p: int = 2,
        mixer_type: str = "X",
        optimizer_name: str = "COBYLA",
        maxiter: int = 60,
        shots: int = 1024,
        seed: int = 42,
    ):
        self.reps_p = reps_p
        self.mixer_type = mixer_type.upper()
        self.optimizer_name = optimizer_name.upper()
        self.maxiter = maxiter
        self.shots = shots
        self.seed = seed
        self.simulator = AerSimulator(seed_simulator=seed)

    def build_qaoa_circuit(
        self, ising: IsingHamiltonian, gammas: np.ndarray, betas: np.ndarray
    ) -> QuantumCircuit:
        """
        Construye el circuito cuantico variacional QAOA de p capas:
        |psi(gamma, beta)> = prod_{l=1}^p [U_M(beta_l) U_C(gamma_l)] |+>^{otimes N}
        """
        N = ising.num_qubits
        qc = QuantumCircuit(N, N)

        # 1. Estado inicial de superposicion uniforme |+>^{otimes N}
        qc.h(range(N))
        qc.barrier()

        # Escala normalizada del Hamiltoniano para evitar alias de fase en [0, 2*pi]
        max_coeff = max(
            float(np.max(np.abs(ising.h_linear))) if N > 0 else 1.0,
            float(np.max(np.abs(ising.J_quadratic))) if N > 0 else 1.0,
            1.0,
        )

        for layer in range(self.reps_p):
            gamma = float(gammas[layer]) / max_coeff
            beta = float(betas[layer])

            # 2. Operador Unitario de Costo U_C(gamma) = exp(-i * gamma * H_C)
            for q in range(N):
                h_q = float(ising.h_linear[q])
                if abs(h_q) > 1e-9:
                    qc.rz(2.0 * gamma * h_q, q)

            for q1 in range(N):
                for q2 in range(q1 + 1, N):
                    j_val = float(ising.J_quadratic[q1, q2])
                    if abs(j_val) > 1e-9:
                        qc.cx(q1, q2)
                        qc.rz(2.0 * gamma * j_val, q2)
                        qc.cx(q1, q2)

            qc.barrier()

            # 3. Operador Unitario de Mezcla U_M(beta)
            if self.mixer_type == "XY":
                # Mezclador XY parcial entre qubits adyacentes de almacenes
                for q in range(N - 1):
                    qc.rxx(2.0 * beta, q, q + 1)
                    qc.ryy(2.0 * beta, q, q + 1)
                for q in range(N):
                    qc.rx(beta, q)
            else:
                # Mezclador transversal estandar X-Mixer: exp(-i * beta * sum X_q)
                for q in range(N):
                    qc.rx(2.0 * beta, q)

            qc.barrier()

        qc.measure(range(N), range(N))
        return qc

    def solve(
        self, qubo: QUBOModel, ising: Optional[IsingHamiltonian] = None
    ) -> QuantumSolverResult:
        t0 = time.perf_counter()
        if ising is None:
            ising = IsingMapper.qubo_to_ising(qubo)

        N = qubo.num_qubits
        energy_history: List[float] = []

        def evaluate_params(params: np.ndarray) -> float:
            gammas = params[: self.reps_p]
            betas = params[self.reps_p :]
            qc = self.build_qaoa_circuit(ising, gammas, betas)
            t_qc = transpile(qc, self.simulator, optimization_level=1)
            counts = (
                self.simulator.run(t_qc, shots=self.shots).result().get_counts()
            )

            # Usar CVaR (Conditional Value at Risk al 25% mejor) + media para acelerar convergencia
            energies: List[Tuple[float, int]] = []
            for bitstr, count in counts.items():
                # Qiskit bitstring es q_{N-1}...q_0; invertimos para indice q_0...q_{N-1}
                z = np.array([int(b) for b in bitstr[::-1]], dtype=int)
                e_val = qubo.evaluate_bitstring(z)
                energies.append((e_val, count))

            energies.sort(key=lambda item: item[0])
            mean_e = sum(e * c for e, c in energies) / self.shots

            # Promedio del cuartil inferior (CVaR alpha = 0.25)
            cutoff_shots = max(1, int(0.25 * self.shots))
            acc_shots = 0
            cvar_sum = 0.0
            for e_val, c in energies:
                take = min(c, cutoff_shots - acc_shots)
                cvar_sum += e_val * take
                acc_shots += take
                if acc_shots >= cutoff_shots:
                    break
            cvar_e = cvar_sum / cutoff_shots

            hybrid_obj = 0.65 * cvar_e + 0.35 * mean_e
            energy_history.append(float(cvar_e))
            return float(hybrid_obj)

        # Inicializacion lineal tipo rampa adiabatica (Trotterized Annealing initialization)
        init_gammas = np.linspace(0.35, 0.85, self.reps_p)
        init_betas = np.linspace(0.75, 0.25, self.reps_p)
        init_params = np.concatenate([init_gammas, init_betas])

        if self.optimizer_name == "SPSA":
            opt_params = self._run_spsa(evaluate_params, init_params)
        else:
            res = minimize(
                evaluate_params,
                init_params,
                method="COBYLA",
                options={"maxiter": self.maxiter, "rhobeg": 0.4},
            )
            opt_params = np.asarray(res.x, dtype=float)

        # Muestreo final de alta precision con los angulos optimos
        opt_gammas = opt_params[: self.reps_p]
        opt_betas = opt_params[self.reps_p :]
        final_qc = self.build_qaoa_circuit(ising, opt_gammas, opt_betas)
        t_final_qc = transpile(final_qc, self.simulator, optimization_level=2)
        final_counts = (
            self.simulator.run(t_final_qc, shots=self.shots * 2)
            .result()
            .get_counts()
        )
        total_final_shots = self.shots * 2

        scored_states: List[Tuple[str, float, float, np.ndarray]] = []
        for bitstr, count in final_counts.items():
            z = np.array([int(b) for b in bitstr[::-1]], dtype=int)
            e_val = qubo.evaluate_bitstring(z)
            prob = count / total_final_shots
            scored_states.append((bitstr[::-1], prob, e_val, z))

        # Ordenar por menor energia encontrada dentro de los estados medidos con prob significativa
        scored_states.sort(key=lambda item: (item[2], -item[1]))
        best_str, _, best_energy, best_z = scored_states[0]

        # Top 8 estados por probabilidad y energia para visualizacion
        top_by_prob = sorted(scored_states, key=lambda x: x[1], reverse=True)[:8]
        top_measurements = [(s, p, e) for s, p, e, _ in top_by_prob]

        gate_counts = {str(k): int(v) for k, v in t_final_qc.count_ops().items()}
        elapsed = time.perf_counter() - t0

        return QuantumSolverResult(
            algorithm=f"QAOA (p={self.reps_p}, mixer={self.mixer_type}, opt={self.optimizer_name})",
            best_bitstring=best_z,
            best_qubo_energy=float(best_energy),
            expected_energy_history=energy_history,
            top_measurements=top_measurements,
            optimal_params=opt_params,
            circuit=final_qc,
            transpiled_depth=int(t_final_qc.depth()),
            gate_counts=gate_counts,
            execution_time_s=elapsed,
            shots=total_final_shots,
        )

    def _run_spsa(self, obj_fn, init_params: np.ndarray) -> np.ndarray:
        """
        Optimizador estocastico SPSA (Simultaneous Perturbation Stochastic Approximation)
        resiliente a ruido de shot en hardware/simuladores cuanticos.
        """
        rng = np.random.default_rng(self.seed)
        theta = init_params.copy()
        d = len(theta)
        n_steps = max(10, self.maxiter // 2)
        a0, c0 = 0.18, 0.12

        for k in range(1, n_steps + 1):
            ak = a0 / ((k + 5) ** 0.602)
            ck = c0 / (k**0.101)
            delta = rng.choice([-1.0, 1.0], size=d)

            e_plus = obj_fn(theta + ck * delta)
            e_minus = obj_fn(theta - ck * delta)
            grad_est = (e_plus - e_minus) / (2.0 * ck * delta)
            theta = theta - ak * grad_est

        return theta