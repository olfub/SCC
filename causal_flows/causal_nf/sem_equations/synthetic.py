import os

import torch
import torch.nn.functional as F

from causal_nf.sem_equations.sem_base import SEM


def load_adj(file):
    with open(file, "r") as f:
        lines = f.readlines()
    
    nodes = []
    edges = []
    reading_edges = False
    
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        if line == "<NODES>":
            reading_edges = False
            continue
        elif line == "<EDGES>":
            reading_edges = True
            continue
        
        if reading_edges:
            src, dst = line.split(" -> ")
            edges.append((src, dst))
        else:
            nodes.append(line)

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    adj_matrix = torch.zeros(len(nodes), len(nodes), dtype=torch.float32)
    for src, dst in edges:
        adj_matrix[node_to_idx[src], node_to_idx[dst]] = 1.0
    
    return adj_matrix


class Synthetic(SEM):
    def __init__(self, nr_nodes, nr_edges, data_seed, sem_name="dummy"):
        functions = None
        inverses = None
        self.adj = None

        if sem_name == "dummy":
            functions = []
            inverses = []
            for i in range(nr_nodes):
                functions.append(lambda *args, idx=i: args[idx])
                inverses.append(lambda *args, idx=i: args[idx])

        super().__init__(functions, inverses, sem_name)

        # TODO here, the root dir is fixed, should be read from the config ideally
        config_name = f"{nr_nodes}_{nr_edges}_{data_seed}"
        path = os.path.join("datasets", "large", f"{config_name}.cg")
        self.adj = load_adj(path).T


    def adjacency(self, add_diag=False):
        adj = self.adj

        if add_diag:
            adj += torch.eye(adj.size(0))

        return adj

    def intervention_index_list(self):
        nr_nodes = len(self.functions)
        return list(range(nr_nodes))
