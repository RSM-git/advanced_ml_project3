import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from torch_geometric.utils import to_dense_adj

import numpy as np
import networkx as nx
from graph_utils import draw_graph

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


dataset = TUDataset(root='./data', name='MUTAG')

rng = torch.Generator().manual_seed(0)

train_loader = DataLoader(dataset, batch_size=1)

NUM_GRAPHS = len(train_loader)


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


def plot_node_distribution(node_distribution):
    node_distribution = sorted(node_distribution.items(), key=lambda x: x[0])

    x = [x[0] for x in node_distribution]
    y = [x[1] for x in node_distribution]

    data = np.repeat(x, y)

    sns.histplot(data=data, bins=x, discrete=True, stat="probability")
    plt.xlabel("Number of nodes")
    plt.ylabel("Number of graphs")
    plt.xticks(range(10, 30, 2), range(10, 30, 2))
    plt.tight_layout()
    plt.show()


def sample_empirical_graph(n_samples: int, node_distribution, densities):
    node_distribution = sorted(node_distribution.items(), key=lambda x: x[0])

    empirical_node_distribution = torch.tensor([x[1] / NUM_GRAPHS for x in node_distribution], dtype=torch.float32)

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

    plot_node_distribution(node_distribution)

    for n, prob in zip(N, edge_probability):
        print(f"Graph with {n} nodes and edge probability {prob}")
        A = generate_graph_er(n, prob)
        draw_graph(A)

    # Sample graph from the training set
    graph = next(iter(train_loader))

    A = to_dense_adj(graph.edge_index, graph.batch)

    draw_graph(A[0])

