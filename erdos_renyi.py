import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

dataset = TUDataset(root='./data/', name='MUTAG')

rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)


train_loader = DataLoader(train_dataset, batch_size=1)
validation_loader = DataLoader(validation_dataset, batch_size=44)
test_loader = DataLoader(test_dataset, batch_size=44)


def graph_density(n_nodes, n_edges):
    return n_edges / (n_nodes * (n_nodes - 1) / 2)


def sample_empirical_graph(dataloader):
    node_distribution = {}
    graph_densities = {}
    for graph in dataloader:
        node_distribution[graph.num_nodes] = node_distribution.get(graph.num_nodes, 0) + 1
        if graph_densities.get(graph.num_nodes, None) is None:
            graph_densities[graph.num_nodes] = []
        graph_densities[graph.num_nodes].append(graph_density(graph.num_nodes, graph.num_edges))

    for key, value in graph_densities.items():
        graph_densities[key] = np.mean(value)

    node_distribution = sorted(node_distribution.items(), key=lambda x: x[0])

    empirical_node_distribution = torch.tensor([x[1] / 100 for x in node_distribution], dtype=torch.float32)

    N = np.random.choice([x[0] for x in node_distribution], 1, p=empirical_node_distribution)[0]

    return N, graph_densities[N]


def generate_graph_er(num_nodes, edge_probability):
    """
    Generates a random graph using the Erdos-Renyi model.
    """
    A = torch.rand(num_nodes, num_nodes) < edge_probability
    A_triu = torch.triu(A, diagonal=1)
    A = torch.zeros(num_nodes, num_nodes) + A_triu + A_triu.T
    return A


def draw_graph(A):
    G = nx.from_numpy_array(A.numpy())
    nx.draw(G, with_labels=True)
    plt.show()


if __name__ == '__main__':
    N, edge_probability = sample_empirical_graph(train_loader)
    print(f"Graph with {N} nodes and edge probability {edge_probability}")
    A = generate_graph_er(N, edge_probability)
    draw_graph(A)
