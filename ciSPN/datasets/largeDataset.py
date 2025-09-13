import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_data(path, folder, nodes, edges, split_name, seed, include_exo=False):
    assert split_name in ["train", "val", "test"]
    base_path = Path(f"{path}/{folder}/{nodes}_{edges}_{seed}")
    
    graph = load_graph(str(base_path) + ".cg")
    # model = load_model(str(base_path) + ".pkl")
    data_orig = np.load(str(base_path) + f"_{split_name}_data_orig.npy")
    data_cf = np.load(str(base_path) + f"_{split_name}_data_cf.npy")
    intervention_indices = np.load(str(base_path) + f"_{split_name}_intervention_indices.npy")
    if include_exo:
        data_exo = np.load(str(base_path) + f"_{split_name}_data_exo.npy")

    class LargeDataset(Dataset):
        def __init__(self, X, Y, seed=None, exo=None):
            self.rng = None if seed is None else np.random.default_rng(seed=seed)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.X = torch.tensor(X, dtype=torch.float32).to(device)
            self.Y = torch.tensor(Y, dtype=torch.float32).to(device)
            self.exo = None
            if exo is not None:
                self.exo = torch.tensor(exo, dtype=torch.float32).to(device)

        def __len__(self):
            return len(self.X)
        
        def shuffle_data(self):
            if self.rng is None:
                return
            permutation = self.rng.permutation(self.X.shape[0])
            self.X = self.X[permutation, :]
            self.Y = self.Y[permutation, :]
            if self.exo is not None:
                self.exo = self.exo[permutation, :]

    # Concatenate intervention_indices and data_orig as X, and use data_cf as Y
    num_interventions = data_orig.shape[1]
    intervention_one_hot = np.zeros((data_cf.shape[0], num_interventions))
    intervention_values = np.zeros((data_cf.shape[0], 1))
    for i in range(intervention_indices.shape[0]):
        if intervention_indices[i, 0] != -1:
            index = intervention_indices[i, 0]
            intervention_one_hot[i, index] = 1
            intervention_values[i, 0] = data_cf[i, index]

    X = np.concatenate((intervention_one_hot, intervention_values, data_orig), axis=1)
    Y = data_cf

    # Create the PyTorch dataset
    data = LargeDataset(X, Y, seed=seed, exo=data_exo if include_exo else None)
    
    # Shuffle the data
    data.shuffle_data()

    return data, graph


def load_graph(file):
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
    
    return nodes, edges


def load_model(path, folder, nodes, edges, seed):
    base_path = Path(f"{path}/{folder}/{nodes}_{edges}_{seed}")
    file = str(base_path) + ".pkl"
    with open(file, "rb") as handle:
        model = pickle.load(handle)
    return model