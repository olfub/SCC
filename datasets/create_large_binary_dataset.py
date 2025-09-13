import argparse
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianModel
from util import number_to_identifier, save_model


class GenerateError(RuntimeError):
    pass


def generate_new_dag(nodes, edges, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # adjacency matrix
    adj = np.ones((nodes, nodes))
    # as many edges as possible in a DAG
    adj = np.tril(adj, -1)
    edge_indices = np.nonzero(adj)
    nr_edges = len(edge_indices[0])
    if edges > nr_edges:
        GenerateError(
            f"Too many edges specified, can not be satisfied (want {edges} but maximum DAG has {nr_edges})"
        )
    # how many edges to remove
    nr_edges_to_rm = nr_edges - edges
    edges_to_rm = rng.choice(
        np.arange(nr_edges), nr_edges_to_rm, replace=False
    )
    # remove edges
    adj[edge_indices[0][edges_to_rm], edge_indices[1][edges_to_rm]] = 0
    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    # relabel nodes to A, B, C, ...
    graph = nx.relabel_nodes(graph, {i: number_to_identifier(i) for i in range(nodes)})

    return graph


def init_bayesian_model(graph, nr_bins, rng, normal_bins_until=3):
    # normal_bins_until: until this many parents, use all possible exogenous bins (i.e., 2^n parents)
    # beyond that, use only nr_bins bins (i.e., possible exogenous states) 

    # create the Bayesian model
    model = BayesianModel()
    model.add_nodes_from(graph.nodes)
    model.add_edges_from(graph.edges)

    # add the counterfactual world variables
    model.add_nodes_from([f"{node}_cf" for node in graph.nodes])
    model.add_edges_from([(f"{u}_cf", f"{v}_cf") for u, v in graph.edges])

    max_parents = 0
    # iterate over all nodes, addings cpts, exogenous parents, and counterfactual cpts
    for node in graph.nodes:        
        parents = list(graph.predecessors(node))
        max_parents = max(max_parents, len(parents))
        
        # generate a random "cpt" with the "normal parents"
        # (these will not be used exactly later, but determine the probabilities for the exogenous parents;
        # as we want all randomness to come from the exogenous parents)
        if len(parents) == 0:
            probs = rng.random((1, ))
        else:
            parent_cards = [2 for _ in parents]
            nr_parent_states = np.prod(parent_cards)
            probs = rng.random((nr_parent_states, ))
        
        # sort these probabilities
        if len(parents) > normal_bins_until:
            # use only nr_bins random probs (-1 because a 1.0 bin will be added)
            # this means, that we simplify our problem a bit (could be more compley), as we need to ensure
            # all noise comes from the exogenous variables
            selected_probs = rng.choice(probs.tolist(), size=nr_bins - 1, replace=False).tolist()
            sorted_probs = np.sort(selected_probs)
        else:
            sorted_probs = np.sort(probs)  # sorted for the state 0 probabilities

        # create bins, assigning probabilities to all thresholds
        bin_thresholds = sorted_probs.tolist()  # contains max value for each bin
        bin_values = [] # contains the probability mass for each bin
        for i in range(len(bin_thresholds)):
            threshold = bin_thresholds[i]
            if i == 0:
                bin_probs = threshold
            else:
                bin_probs = threshold - bin_thresholds[i - 1]
            bin_values.append(bin_probs)
        bin_values.append(1.0 - bin_thresholds[-1])
        bin_thresholds.append(1.0)

        # create the exogenous variables and put the bins into the exogenous parent cpt
        exog_parent = f"{node}_exog"
        model.add_node(exog_parent)
        model.add_edge(exog_parent, node)
        values = np.array(bin_values).reshape(-1, 1)
        exog_cpd = TabularCPD(variable=exog_parent, variable_card=len(bin_values), values=values)
        model.add_cpds(exog_cpd)

        # create the cpt for the node with placeholder probabilities (zeros, will be filled right after)
        cpd = TabularCPD(variable=node, variable_card=2, 
                         evidence=[exog_parent] + parents,
                         evidence_card=[len(bin_values)] + [2 for _ in parents],
                         values=np.zeros((2, len(bin_values) * (2 ** len(parents)))))

        # iterate over bins
        values = np.zeros((2, len(bin_values) * (2 ** len(parents))))
        temp_counter = 0
        last = -1
        for bin_idx, threshold in enumerate(bin_thresholds):
            # assign 1 if bin threshold < cpt threshold, 0 otherwise
            # basically, the value is true, if the exogenous state sample
            # is below the conditional probability of the respective cpt
            # while this models only a subset of all possible functions,
            # the full set would be too large to handle and this still
            # allows for meaningful and interesting counterfactuals
            for parent_state in range(2 ** len(parents)):
                if threshold <= probs[parent_state]:
                    values[0, bin_idx * (2 ** len(parents)) + parent_state] = 0
                    values[1, bin_idx * (2 ** len(parents)) + parent_state] = 1
                else:
                    values[0, bin_idx * (2 ** len(parents)) + parent_state] = 1
                    values[1, bin_idx * (2 ** len(parents)) + parent_state] = 0
            # sanity check, ensuring that we have at least one more 1 than in the last bin
            # this checks that each exogenous state can make a difference, i.e., no 
            # exogenous state is redundant
            current = sum(values[0]) - temp_counter
            temp_counter += current
            assert (current > last)  # should always be true
            last = current
        
        cpd.values = values
        model.add_cpds(cpd)

        # add the counterfactual nodes with the same cpt logic
        cf_node = f"{node}_cf"
        model.add_edge(exog_parent, cf_node)
        cf_parents = [f"{p}_cf" for p in parents]
        cpd = TabularCPD(variable=cf_node, variable_card=2,
                         evidence=[exog_parent] + cf_parents,
                         evidence_card=[len(bin_values)] + [2 for _ in cf_parents],
                         values=np.zeros((2, len(bin_values) * (2 ** len(cf_parents)))))
        
        cpd.values = values  # same logic as above
        model.add_cpds(cpd)

    model.check_model()
    print(f"Max parents: {max_parents}")
    return model


def sample_bayesian_model(model, n_samples_per_intervention, rng):
    nspi = n_samples_per_intervention  # abbreviation
    # endogenous nodes (without exogenous and counterfactual nodes)
    endo_nodes = [node for node in model.nodes if not str(node).endswith("_exog") and not str(node).endswith("_cf")]
    assert(len(endo_nodes) * 3 == len(model.nodes))  # exogenous and cf nodes should be present for each endogenous node

    # prepare data containers
    data_orig = np.zeros((nspi * (len(endo_nodes) + 1), len(endo_nodes)))
    data_cf = np.zeros((nspi * (len(endo_nodes) + 1), len(endo_nodes)))
    intervention_indices = np.zeros((nspi * (len(endo_nodes) + 1), 1), dtype=np.int8)
    data_exo = np.zeros((nspi * (len(endo_nodes) + 1), len(endo_nodes)))

    # sample observational data
    no_intervention = model.simulate(n_samples=nspi, seed=rng.integers(0, 1e6))

    # fill in the observational data
    for i, node in enumerate(endo_nodes):
        data_orig[:nspi, i] = no_intervention[node].values
        data_cf[:nspi, i] = no_intervention[f"{node}_cf"].values
        data_exo[:nspi, i] = no_intervention[f"{node}_exog"].values
    intervention_indices[:nspi] = -1  # no intervention


    # sample counterfactual data
    for count, node in enumerate(endo_nodes):
        nr_do_0 = nspi // 2
        nr_do_1 = nspi - nr_do_0
        samples_0 = model.simulate(n_samples=nr_do_0, do={f"{node}_cf": 0}, seed=rng.integers(0, 1e6))
        samples_1 = model.simulate(n_samples=nr_do_1, do={f"{node}_cf": 1}, seed=rng.integers(0, 1e6))
        samples = pd.concat([samples_0, samples_1], ignore_index=True)

        # fill in the data
        for i, n in enumerate(endo_nodes):
            data_orig[nspi * (count + 1): nspi * (count + 2), i] = samples[n].values
            data_cf[nspi * (count + 1): nspi * (count + 2), i] = samples[f"{n}_cf"].values
            data_exo[nspi * (count + 1): nspi * (count + 2), i] = samples[f"{n}_exog"].values
        intervention_indices[nspi * (count + 1): nspi * (count + 2)] = count

    return data_orig, data_cf, data_exo, intervention_indices

def save_graph(file_name, graph):
    with open(file_name, "w") as f:
        f.write("<NODES>\n")
        f.write("\n".join([str(node) for node in graph.nodes()]) + "\n\n")
        f.write("<EDGES>\n")
        for edge in graph.edges():
            f.write(str(edge[0]) + " -> " + str(edge[1]) + "\n")

def main(args):
    nodes = args.nodes
    edges = args.edges
    nr_bins = 10
    rng = np.random.default_rng(args.seed)
    n_samples_per_intervention = args.samples_per_int
    path = args.path
    folder = args.folder

    # generate graph
    graph = generate_new_dag(nodes, edges, rng)
    # create model from graph (random cpts, counterfactual variables, exogenous parents)
    model = init_bayesian_model(graph, nr_bins, rng)
    start = time.time()
    # sample observational and counterfactual data
    data_orig, data_cf, data_exo, intervention_indices = sample_bayesian_model(model, n_samples_per_intervention=n_samples_per_intervention, rng=rng)
    end = time.time()
    print(f"Sampling took {end - start} seconds")

    path = Path(f"{path}/{folder}/{nodes}_{edges}_{args.seed}")
    path.parent.mkdir(parents=True, exist_ok=True)

    save_model(str(path) + ".pkl", model)

    # Shuffle and split the data into train, validation, and test sets using a shuffle mask
    num_samples = data_orig.shape[0]
    np.random.seed(args.seed)  # Ensure reproducibility
    shuffle_mask = np.random.permutation(num_samples)

    train_split_idx = int(0.8 * num_samples)
    val_split_idx = int(0.9 * num_samples)

    train_mask = shuffle_mask[:train_split_idx]
    val_mask = shuffle_mask[train_split_idx:val_split_idx]
    test_mask = shuffle_mask[val_split_idx:]

    train_data_orig = data_orig[train_mask]
    train_data_cf = data_cf[train_mask]
    train_data_exo = data_exo[train_mask]
    train_intervention_indices = intervention_indices[train_mask]

    val_data_orig = data_orig[val_mask]
    val_data_cf = data_cf[val_mask]
    val_data_exo = data_exo[val_mask]
    val_intervention_indices = intervention_indices[val_mask]

    test_data_orig = data_orig[test_mask]
    test_data_cf = data_cf[test_mask]
    test_data_exo = data_exo[test_mask]
    test_intervention_indices = intervention_indices[test_mask]

    np.save(str(path) + "_train_data_orig.npy", train_data_orig)
    np.save(str(path) + "_train_data_cf.npy", train_data_cf)
    np.save(str(path) + "_train_data_exo.npy", train_data_exo)
    np.save(str(path) + "_train_intervention_indices.npy", train_intervention_indices)

    np.save(str(path) + "_val_data_orig.npy", val_data_orig)
    np.save(str(path) + "_val_data_cf.npy", val_data_cf)
    np.save(str(path) + "_val_data_exo.npy", val_data_exo)
    np.save(str(path) + "_val_intervention_indices.npy", val_intervention_indices)

    np.save(str(path) + "_test_data_orig.npy", test_data_orig)
    np.save(str(path) + "_test_data_cf.npy", test_data_cf)
    np.save(str(path) + "_test_data_exo.npy", test_data_exo)
    np.save(str(path) + "_test_intervention_indices.npy", test_intervention_indices)

    save_graph(str(path) + ".cg", graph)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--nodes", default=5, type=int)
    parser.add_argument("--edges", default=5, type=int)
    parser.add_argument("--samples_per_int", default=10000, type=int)
    parser.add_argument("--path", default="datasets", type=str)
    parser.add_argument("--folder", default="large", type=str)

    args = parser.parse_args()
    main(args)