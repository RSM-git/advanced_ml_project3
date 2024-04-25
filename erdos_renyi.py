import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from torch_geometric.utils import to_dense_adj

import numpy as np
import networkx as nx
from graph_utils import draw_graph

import matplotlib.pyplot as plt


dataset = TUDataset(root='./data', name='MUTAG')

rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)


train_loader = DataLoader(train_dataset, batch_size=1)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)


def graph_density(n_nodes, n_edges):
    return n_edges / (n_nodes * (n_nodes - 1) / 2)


def node_stats(dataloader):
    node_distribution = {}
    graph_densities = {}
    for graph in dataloader:
        node_distribution[graph.num_nodes] = node_distribution.get(graph.num_nodes, 0) + 1

        if graph_densities.get(graph.num_nodes, None) is None:
            graph_densities[graph.num_nodes] = []

        graph_densities[graph.num_nodes].append(graph_density(graph.num_nodes, graph.num_edges / 2))

    for key, value in graph_densities.items():
        graph_densities[key] = np.mean(value)

    return node_distribution, graph_densities


def sample_empirical_graph(n_samples: int, node_distribution, densities):
    node_distribution = sorted(node_distribution.items(), key=lambda x: x[0])

    empirical_node_distribution = torch.tensor([x[1] / 100 for x in node_distribution], dtype=torch.float32)

    N = np.random.choice([x[0] for x in node_distribution], n_samples, p=empirical_node_distribution)

    density = [densities[n] for n in N]

    return N, density


def generate_graph_er(num_nodes, edge_probability):
    """
    Generates a random graph using the Erdos-Renyi model.
    """
    A = torch.rand(num_nodes, num_nodes) < edge_probability
    A = torch.triu(A, diagonal=1)
    A = A + A.T
    return A



if __name__ == '__main__':
    node_distribution, densities = node_stats(train_loader)
    N, edge_probability = sample_empirical_graph(n_samples=1, node_distribution=node_distribution, densities=densities)
    for n, prob in zip(N, edge_probability):
        print(f"Graph with {n} nodes and edge probability {prob}")
        A = generate_graph_er(n, prob)
        draw_graph(A)


    # Sample graph from the training set
    graph = next(iter(train_loader))

    A = to_dense_adj(graph.edge_index, graph.batch)

    draw_graph(A[0])

