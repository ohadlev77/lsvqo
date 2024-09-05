from typing import Callable, Union
from pathlib import Path, PosixPath

import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator

from draft_6_util_funs import (
    obtain_high_degree_variance_graph,
    obtain_highest_degree_random_b_blue_nodes,
    obtain_random_partition
)
from draft_5_util_funs import QEMC
from draft_4_util_funs import qemc_cost_fun, compute_cut

def mine_data(
    num_nodes: list[int],
    num_blue_nodes: list[int],
    optimization_funs: dict[str, Callable],
    num_shots: list[int],
    num_optimizer_iterations: int,
    data_export_path: Union[str, PosixPath],
    style_map: dict[str, str]
) -> None: # TODO
    """TODO COMPLETE."""

    x_axis = range(num_optimizer_iterations)
    num_cols = len(num_nodes)
    num_rows = len(num_blue_nodes)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(150, 60))
    fig.suptitle(
        "Best cuts as a function of optimizer iterations "
        f"Optimization functions: {list(optimization_funs.keys())}, "
        "y-axes: cut values, x-axes: optimizer iterations \n"
        f"Legend = {style_map}"
    )

    for current_num_nodes in num_nodes:
    
        graph = obtain_high_degree_variance_graph(current_num_nodes)
        
        for current_num_blue_nodes in num_blue_nodes:

            for current_num_shots in num_shots:

                row_index = num_blue_nodes.index(current_num_blue_nodes)
                col_index = num_nodes.index(current_num_nodes)        
                cell = axes[row_index][col_index]

                best_random_cuts = []
                for _ in x_axis:
                    current_random_cut = compute_cut(
                        graph,
                        obtain_random_partition(
                            num_nodes=current_num_nodes,
                            num_blue_nodes=current_num_blue_nodes,
                            blue_nodes=obtain_highest_degree_random_b_blue_nodes(
                                graph, current_num_blue_nodes, current_num_shots
                            )
                        )
                    )
                
                    if len(best_random_cuts) == 0:
                        best_random_cuts.append(current_random_cut)
                    elif best_random_cuts[-1] > current_random_cut:
                        best_random_cuts.append(best_random_cuts[-1])
                    else:
                        best_random_cuts.append(current_random_cut)

                cell.plot(
                    x_axis,
                    best_random_cuts,
                    color=style_map["num_shots"][current_num_shots],
                    linestyle=style_map["type"]["random"]
                )
                
                for optimization_fun_name, optimization_fun in optimization_funs.items():

                    qemc_executer = QEMC(
                        graph=graph,
                        num_blue_nodes=current_num_blue_nodes,
                        quantum_backend=AerSimulator(shots=current_num_shots)
                    )

                    ansatz = qemc_executer.define_ansatz()

                    print()
                    print(
                        f"Executing QEMC with: "
                        f"num_nodes={current_num_nodes}, "
                        f"num_blue_nodes={current_num_blue_nodes}, "
                        f"num_shots={current_num_shots}, "
                        f"cost_function={optimization_fun_name}"
                    )

                    qemc_result = qemc_executer.run(
                        optimization_fun=optimization_fun,
                        optimization_method="COBYLA",
                        optimization_options={"maxiter": num_optimizer_iterations}
                    )

                    QEMC.export_qemc_result_data(qemc_result, data_export_path)
                    
                    cell.plot(
                        range(len(qemc_result["data"]["best_cut_values"])),
                        qemc_result["data"]["best_cut_values"],
                        color=style_map["num_shots"][current_num_shots],
                        linestyle=style_map["type"][optimization_fun_name]
                    )

                    cell.set_xlabel(
                        f"num_nodes={current_num_nodes}, "
                        f"num_blue_nodes={current_num_blue_nodes}"
                    )

                    # print(
                    #     f"DONE. final_cut_value={qemc_result['data']['cut_values'][-1]}"
                    # )

                    # The fig is saved in each iteration to see online progress
                    plt.savefig(Path(data_export_path, "fig.jpg"))


if __name__ == "__main__":
    
    # Settings
    maxiter = 2_000
    num_nodes = [32, 64, 128, 256, 512, 1024, 2048]
    num_blue_nodes = [2, 4, 8, 16, 32]
    num_shots = [25, 64, 121, 512]
    cost_functions = {"c1": qemc_cost_fun}
    data_export_path = Path(
        "/home/ohadlev77/personal/research/large_scale_variational_quantum_optimization/draft_8_data_start_from_all_zeros_param_vec_with_initial_h_layer"
    )

    style_map = {
        "num_shots": {
            25: "red",
            64: "green",
            121: "blue",
            512: "purple"

        },
        "type": {
            "random": "--",
            "c1": "-"
        }
    }

    mine_data(
        num_nodes=num_nodes,
        num_blue_nodes=num_blue_nodes,
        optimization_funs=cost_functions,
        num_shots=num_shots,
        num_optimizer_iterations=maxiter,
        data_export_path=data_export_path,
        style_map=style_map
    )