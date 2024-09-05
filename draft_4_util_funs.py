from typing import Union, Optional

import networkx as nx
import numpy as np


def qemc_cost_fun(
    prob_dist: dict[str, float],
    graph: nx.Graph,
    B: int, # Number of blue nodes,
    num_bits: int,
    mul_variance: Optional[bool] = False
) -> float:
       
    if mul_variance:
        prob_dist_array = np.array(list(prob_dist.values()))
        mul_variance = -1 * prob_dist_array.var()
    else:
        mul_variance = 1
        
    cost = 0
    for node_i, node_j in graph.edges():
        
        p_i = prob_dist[format(node_i, "b").zfill(num_bits)]
        p_j = prob_dist[format(node_j, "b").zfill(num_bits)]
        
        cost += (p_i + p_j - 1/B)**2 + (np.abs(p_i - p_j) - 1/B)**2
        
    cost *= mul_variance
        
    return cost


def brute_force_maxcut(
    graph: nx.Graph,
    num_blue_nodes: Optional[int] = None
) -> dict[str, Union[str, int]]:
    
    num_nodes = graph.number_of_nodes()
    maxcut = 0
    best_partition = ""
    
    for i in range(2**num_nodes):
        
        partition = format(i, "b").zfill(num_nodes)

        if num_blue_nodes is not None and partition.count("1") != num_blue_nodes:
            continue

        cut = compute_cut(graph, partition)
        
        if cut > maxcut:
            maxcut = cut
            best_partition = partition
            
    return {"best_partition": best_partition, "maxcut": maxcut}
        
        
def compute_cut(
    graph: nx.Graph,
    partition: str
) -> int:
    
    cut = 0
    for node_i, node_j in graph.edges():
        if partition[node_i] != partition[node_j]:
            cut += 1
            
    return cut


def obtain_uniform_distribution(num_nodes: int) -> np.ndarray:

    return np.ones(num_nodes) / num_nodes


def obtain_pseudo_ideal_distribution(num_nodes: int, B: int) -> np.ndarray:

    return np.array([1/B if i % (num_nodes/B) == 0 else 0 for i in range(num_nodes)])


def obtain_distribution_from_partition(partition: str) -> np.ndarray:

    num_nodes = len(partition)
    distribution = np.zeros(num_nodes)

    num_blue_nodes = 0
    for index, bit in enumerate(partition):
        bit = int(bit)
        distribution[index] = bit
        num_blue_nodes += bit

    distribution /= num_blue_nodes

    return distribution