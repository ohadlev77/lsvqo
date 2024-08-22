from typing import Optional

import numpy as np
import networkx as nx


def obtain_random_partition(
    num_nodes: int,
    num_blue_nodes: int,
    blue_nodes: Optional[list[int]] = None
) -> str:
    
    partition = ["0" for _ in range(num_nodes)]

    if blue_nodes is None:
        blue_nodes = np.random.choice(num_nodes, num_blue_nodes, replace=False)
    
    for blue_node in blue_nodes:
        partition[blue_node] = "1"
        
    return "".join(partition)


def obtain_highest_degree_random_b_blue_nodes(
    graph: nx.Graph,
    num_blue_nodes: int,
    num_shots: int
) -> dict[str, int]:
    
    num_nodes = graph.number_of_nodes()
    random_nodes = np.random.choice(num_nodes, num_shots)

    blue_nodes = []
    lowest_degree = 0
    for node_id in random_nodes:

        if node_id in blue_nodes:
            continue

        if graph.degree(node_id) > lowest_degree:
            blue_nodes.append(node_id)
            
        if len(blue_nodes) > num_blue_nodes:
            blue_nodes.remove(min(blue_nodes, key=lambda x: graph.degree(x)))
            lowest_degree = min(graph.degree(node_id) for node_id in blue_nodes)

    return blue_nodes


def compute_cut_from_edges_list(edges_list: list[list[int]], partition: str) -> int:
    
    cut = 0
    for node_i, node_j in edges_list:
        if partition[node_i] != partition[node_j]:
            cut += 1
            
    return cut


def obtain_high_degree_variance_graph(num_nodes) -> nx.Graph:
    graph = nx.scale_free_graph(
        n=num_nodes,
        seed=np.random.randint(100),
        alpha=0.4,
        beta=0.2,
        gamma=0.4
    ).to_undirected()
    
    graph.remove_edges_from(nx.selfloop_edges(graph))
    
    return graph


if __name__ == "__main__":
    print(
        obtain_highest_degree_random_b_blue_nodes(
            graph=nx.erdos_renyi_graph(n=10, p=0.4),
            num_blue_nodes=5,
            num_shots=1_000
        )
    )