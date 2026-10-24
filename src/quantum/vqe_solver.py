import time
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from src.quantum.qubo_builder import QUBOModel
from src.quantum.ising_mapper import IsingHamiltonian, IsingMapper
from src.quantum.qaoa_solver import QuantumSolverResult


class VQESolver:
    """
    Implementacion del Variational Quantum Eigensolver (VQE) con ansatz
    Hardware-Efficient (TwoLocal / EfficientSU2: rotaciones RY-RZ y escaleras CNOT).
    """

    def __init__(
        self,
        reps: int = 2,
        entanglement: str = "linear",
        optimizer_name: str = "COBYLA",
        maxiter: int = 60,
        shots: int = 1024,
        seed: int = 42,
    ):
        self.reps = reps
        self.entanglement = entanglement
        self.optimizer_name = optimizer_name.upper()
        self.maxiter = maxiter
        self.shots = shots
        self.seed = seed
        self.simulator = AerSimulator(seed_simulator=seed)

    def build_vqe_ansatz(self, num_qubits: int, params: np.ndarray) -> QuantumCircuit:
        """
        Construye un ansatz Hardware-Efficient tipo EfficientSU2 con capas RY, RZ
        y entrelazamiento CNOT lineal o circular.
        """
        qc = QuantumCircuit(num_qubits, num_qubits)
        idx = 0

        # Capa inicial de rotaciones
        for q in range(num_qubits):
            qc.ry(float(params[idx]), q)
            idx += 1
            qc.rz(float(params[idx]), q)
            idx += 1
        qc.barrier()

        for _ in range(self.reps):
            # Escalera de entrelazamiento CNOT
            for q in range(num_qubits - 1):
                qc.cx(q, q + 1)
            if self.entanglement == "circular" and num_qubits > 2:
                qc.cx(num_qubits - 1, 0)
            qc.barrier()

            # Rotaciones parametrizadas por capa
            for q in range(num_qubits):
                qc.ry(float(params[idx]), q)
                idx += 1
                qc.rz(float(params[idx]), q)
                idx += 1
            qc.barrier()

        qc.measure(range(num_qubits), range(num_qubits))
        return qc

    def solve(
        self, qubo: QUBOModel, ising: Optional[IsingHamiltonian] = None
    ) -> QuantumSolverResult:
        t0 = time.perf_counter()
        if ising is None:
            ising = IsingMapper.qubo_to_ising(qubo)

        N = qubo.num_qubits
        num_params = 2 * N * (self.reps + 1)
        energy_history: List[float] = []

        def evaluate_params(params: np.ndarray) -> float:
            qc = self.build_vqe_ansatz(N, params)
            t_qc = transpile(qc, self.simulator, optimization_level=1)
            counts = (
                self.simulator.run(t_qc, shots=self.shots).result().get_counts()
            )

            energies: List[Tuple[float, int]] = []
            for bitstr, count in counts.items():
                z = np.array([int(b) for b in bitstr[::-1]], dtype=int)
                e_val = qubo.evaluate_bitstring(z)
                energies.append((e_val, count))

            energies.sort(key=lambda item: item[0])
            mean_e = sum(e * c for e, c in energies) / self.shots

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

            hybrid_obj = 0.70 * cvar_e + 0.30 * mean_e
            energy_history.append(float(cvar_e))
            return float(hybrid_obj)

        rng = np.random.default_rng(self.seed)
        # Inicializacion cercana a pi/2 para partir de superposicion amplia
        init_params = rng.uniform(0.3 * np.pi, 0.7 * np.pi, size=num_params)

        res = minimize(
            evaluate_params,
            init_params,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.5},
        )
        opt_params = np.asarray(res.x, dtype=float)

        final_qc = self.build_vqe_ansatz(N, opt_params)
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

        scored_states.sort(key=lambda item: (item[2], -item[1]))
        best_str, _, best_energy, best_z = scored_states[0]

        top_by_prob = sorted(scored_states, key=lambda x: x[1], reverse=True)[:8]
        top_measurements = [(s, p, e) for s, p, e, _ in top_by_prob]

        gate_counts = {str(k): int(v) for k, v in t_final_qc.count_ops().items()}
        elapsed = time.perf_counter() - t0

        return QuantumSolverResult(
            algorithm=f"VQE (EfficientSU2, reps={self.reps}, ent={self.entanglement})",
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