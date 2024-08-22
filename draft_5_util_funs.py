import math
from pathlib import PosixPath, Path
from typing import Optional, Any, Callable, Union
from collections import Counter
from itertools import combinations

import numpy as np
import networkx as nx
from scipy.optimize import minimize

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.circuit.library import RealAmplitudes
from qiskit_aer import StatevectorSimulator

from draft_4_util_funs import qemc_cost_fun, compute_cut


class QEMC:

    def __init__(
        self,
        graph: nx.Graph,
        num_blue_nodes: int,
        quantum_backend: Optional[Backend] = StatevectorSimulator
    ) -> None:

        self.graph = graph
        self.num_blue_nodes = num_blue_nodes
        self.quantum_backend = quantum_backend

        self.num_nodes = graph.number_of_nodes()


    def define_ansatz(self, ansatz: Optional[QuantumCircuit] = None) -> QuantumCircuit:

        self.num_qubits = math.ceil(np.log2(self.num_nodes))

        if ansatz is None:
            ansatz = RealAmplitudes(num_qubits=self.num_qubits, entanglement="circular", reps=5).decompose()

            if not self.quantum_backend.name == "statevector_simulator":
                ansatz.measure_all()

        self.ansatz = ansatz
        return ansatz


    def run(
        self,
        optimization_fun: Optional[Callable] = qemc_cost_fun,
        x0: Optional[np.ndarray] = None,
        optimization_args: Optional[tuple[Any]] = (),
        optimization_method: Optional[str] = "COBYLA",
        optimization_options: Optional[dict[str, Any]] = {"max_iter": 2_000},
    ) -> None: # TODO complete return type

        x0_random_flag = False
        if x0 is None:
            x0 = np.random.uniform(0, 2 * np.pi, size=self.ansatz.num_parameters)
            x0_random_flag = True

        self.cost_values = []
        self.cut_values = []
        self.best_cut_values = []

        optimizer_result = minimize(
            fun=self.compute_cost_single_iteration,
            x0=x0,
            args=(optimization_fun, optimization_args),
            method=optimization_method,
            options=optimization_options
        )

        metadata = dict(
            optimization_fun=optimization_fun.__name__,
            num_nodes=self.num_nodes,
            num_blue_nodes=self.num_blue_nodes,
            backend=self.quantum_backend.name,
            num_shots=self.quantum_backend.options.shots,
            edges=list(self.graph.edges()),
            x0_random=x0_random_flag,
            optimization_args=optimization_args,
            optimization_method=optimization_method,
            optimization_options=optimization_options
        )
        data = dict(
            optimizer_result=dict(optimizer_result),
            cost_values=self.cost_values,
            cut_values=self.cut_values,
            best_cut_values=self.best_cut_values,
            # final_probability_distribution=self.probability_distribution
        )

        return dict(metadata=metadata, data=data)
    

    def compute_cost_single_iteration(
        self,
        optimization_params: np.ndarray,
        optimization_fun: Callable,
        optimization_args: tuple[Any]
    ) -> float:
        
        counts = Counter(
            self.quantum_backend.run(
                self.ansatz.assign_parameters(optimization_params)
            ).result().get_counts()
        )

        if list(counts.values())[0] >= 1:
            self.probability_distribution = counts_to_prob_dist(counts)
        else:
            self.probability_distribution = counts

        cost_value = optimization_fun(
            self.probability_distribution,
            self.graph,
            self.num_blue_nodes,
            self.num_qubits,
            *optimization_args
        )

        self.cost_values.append(cost_value)

        cut_value = compute_cut(
            self.graph,
            partition=obtain_partition_from_distribution(
                probability_distribution=self.probability_distribution,
                num_nodes=self.num_nodes,
                num_blue_nodes=self.num_blue_nodes,
                classification_threshold=None
            ),
        )

        self.cut_values.append(cut_value)

        if len(self.best_cut_values) == 0:
            self.best_cut_values.append(cut_value)
        elif self.best_cut_values[-1] > cut_value:
            self.best_cut_values.append(self.best_cut_values[-1])
        else:
            self.best_cut_values.append(cut_value)

        return cost_value


    @staticmethod
    def export_qemc_result_data(
        qemc_result: dict[str, Any],
        export_path: Union[str, PosixPath]
    ) -> None:
        
        import h5py
        from datetime import datetime

        time_stamp = datetime.timestamp(datetime.now())

        export_path = Path(export_path, f"qemc_{time_stamp}.hdf5")
        with h5py.File(export_path, "w") as f:

            for key, value in qemc_result["metadata"].items():
                if key == "optimization_options":
                    f.attrs["maxiter"] = value["maxiter"]
                else:
                    f.attrs[key] = value

            for key, value in qemc_result["data"].items():
                
                if key == "optimizer_result":
                    f.create_dataset("optimizer_result", data=value.pop("x"))

                    for optimizer_result_key, optimizer_result_value in value.items():
                        f["optimizer_result"].attrs[optimizer_result_key] = optimizer_result_value

                else:
                    f.create_dataset(key, data=value)


def counts_to_prob_dist(counts: dict[str, int]) -> Counter[str, float]:
    """Converts `Counts` object to a probability distribution."""

    total_counts = sum(counts.values())

    return Counter(
        {basis_state: count / total_counts for basis_state, count in counts.items()}
    )


def qemc_c2(
    prob_dist: dict[str, float],
    graph: nx.Graph,
    B: int, # Number of blue nodes,
    num_bits: int,
    inverse: bool = False,
    addition: bool = False
) -> float:
        
    cost = 0
    for node_i, node_j in graph.edges():
        
        p_i = prob_dist[format(node_i, "b").zfill(num_bits)]
        p_j = prob_dist[format(node_j, "b").zfill(num_bits)]

        d_i = graph.degree[node_i]
        d_j = graph.degree[node_j]
        
        degree_prefactor = (d_i - d_j)**2
        if inverse and degree_prefactor != 0:
            degree_prefactor = 1 / degree_prefactor

        if addition:
            cost +=  degree_prefactor + (p_i + p_j - 1/B)**2 + (np.abs(p_i - p_j) - 1/B)**2
        else:    
            cost +=  degree_prefactor * ((p_i + p_j - 1/B)**2 + (np.abs(p_i - p_j) - 1/B)**2)
        
    return cost


def qemc_c3(
    prob_dist: dict[str, float],
    graph: nx.Graph,
    B: int, # Number of blue nodes,
    num_bits: int,
) -> float:
    
    return qemc_c2(
        prob_dist,
        graph,
        B,
        num_bits,
        inverse=True
    )


def qemc_c4(
    prob_dist: dict[str, float],
    graph: nx.Graph,
    B: int, # Number of blue nodes,
    num_bits: int,
) -> float:
    
    return qemc_c2(
        prob_dist,
        graph,
        B,
        num_bits,
        inverse=False,
        addition=True
    )


def generate_bitstrings(n, k):
    positions = list(combinations(range(n), k))
    for pos in positions:
        bitstring = ['0'] * n
        for p in pos:
            bitstring[p] = '1'
        yield ''.join(bitstring)


def brute_force_maxcut_efficient_ver2(
    graph: nx.Graph,
    num_blue_nodes: Optional[int] = None
) -> dict[str, Union[str, int]]:
    
    num_nodes = graph.number_of_nodes()
    maxcut = 0
    best_partition = ""
    
    for partition in generate_bitstrings(num_nodes, num_blue_nodes):

        cut = compute_cut(graph, partition)
        
        if cut > maxcut:
            maxcut = cut
            best_partition = partition
            
    return {"best_partition": best_partition, "maxcut": maxcut}


def obtain_partition_from_distribution(
    probability_distribution: dict[str, float],
    num_nodes: int,
    num_blue_nodes: int,
    classification_threshold: Optional[float] = None
) -> str:

    partition = ["0" for _ in range(num_nodes)]

    sorted_probability_distribution = sorted(
        probability_distribution.items(),
        key=lambda item: item[1],
        reverse=True
    )

    blue_nodes_counter = 0
    for blue_basis_state, probability in sorted_probability_distribution:

        if classification_threshold is None:
            if blue_nodes_counter >= num_blue_nodes:
                break
        
        else:
            if probability < classification_threshold:
                break
        
        partition[int(blue_basis_state, 2)] = "1"
        blue_nodes_counter += 1

    return "".join(list(partition))


if __name__ == "__main__":

    num_qubits = 4
    num_nodes = 2 ** num_qubits
    num_blue_nodes = int(num_nodes / 2)

    graph = nx.erdos_renyi_graph(n=num_nodes, p=0.4)

    qemc_executer = QEMC(
        graph=graph,
        num_blue_nodes=num_blue_nodes,
        quantum_backend=StatevectorSimulator()
    )

    ansatz = qemc_executer.define_ansatz()

    optimization_funs = [qemc_c2]
    for optimization_fun in optimization_funs:

        print()
        print(f"Executing QEMC with {optimization_fun.__name__}")

        qemc_result = qemc_executer.run(
            optimization_fun=optimization_fun,
            optimization_method="COBYLA",
            optimization_options={"maxiter": 1_000}
        )

        export_path = Path(
            "/home/ohadlev77/personal/research/large_scale_variational_quantum_optimization/draft_5_data"
        )
        QEMC.export_qemc_result_data(qemc_result, export_path)

        print()
        print(
            f"Done executing QEMC with {optimization_fun.__name__}. " \
            f"Exported data and metadata into {export_path}."
        )

    print(brute_force_maxcut_efficient_ver2(graph, num_blue_nodes=num_blue_nodes))