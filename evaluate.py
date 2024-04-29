import networkx as nx
import numpy as np
import data
import numpy.typing as npt
from collections import defaultdict
from torch_geometric.utils import to_networkx
from erdos_renyi import generate_graph_er, node_stats, sample_empirical_graph
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pickle

class Evaluator:
    """
    params:
        graphs: list of graphs to evaluate
    """
    def __init__(self, graphs: list[nx.Graph]):
        self.graphs = graphs
        self.train_graphs = data.get_datasets()
        self.convert_train_graphs()
        self.n_train = len(self.train_graphs)
        self.n_eval = len(graphs)
        self.set_hashes()
    
    def set_hashes(self) -> None:
        self.train_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in self.train_graphs]
        self.eval_hashes = [nx.weisfeiler_lehman_graph_hash(g) for g in self.graphs]
        self.train_hashes_set = set(self.train_hashes)
        self.eval_hashes_set = set(self.eval_hashes)
    
    def convert_train_graphs(self):
        # convert the torch_geometric train_graphs to nx.Graph
        self.train_graphs = [to_networkx(g, to_undirected=True) for g in self.train_graphs]
    
    def get_novelty(self) -> float:
        n_novel = 0
        for hash in self.eval_hashes:
            if hash not in self.train_hashes_set:
                n_novel += 1
        return n_novel / self.n_eval
    
    def get_uniqueness(self) -> float:
        return len(self.eval_hashes_set) / self.n_eval
    
    def get_novel_and_unique(self) -> float:
        eval_counts = defaultdict(lambda: 0)
        for hash in self.eval_hashes:
            eval_counts[hash] += 1

        novel_and_unique = 0
        for hash in self.eval_hashes:
            if hash not in self.train_hashes_set:
                if eval_counts[hash] == 1:
                    novel_and_unique += 1
        
        return novel_and_unique / self.n_eval

def get_degree_counts(graphs: list[nx.Graph]) -> list:
    degrees = []
    for g in graphs:
        d = g.degree()
        for t in d:
            degrees.append(t[1])
    return degrees

def get_ccoef_counts(graphs: list[nx.Graph]) -> list:
    ccoefs = []
    for g in graphs:
        g_coefs = nx.clustering(g)
        ccoefs.extend(g_coefs.values())
    return ccoefs

def get_centralities(graphs: list[nx.Graph]) -> list:
    centralities = []
    for g in graphs:
        centrality = nx.degree_centrality(g)
        for t in centrality.values():
            centralities.append(t)
    return centralities


def plot_degree_distribution(empirical_degrees: list, baseline_degrees: list, vae_degrees: list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_degrees = [empirical_degrees, baseline_degrees, vae_degrees]
    max_degree = max([max(d) for d in all_degrees])
    titles = ["MUTAG", "Baseline", "VAE"]
    for i, degrees in enumerate(all_degrees):
        axes[i].hist(degrees, bins=np.arange(0, max_degree+1, 1), rwidth=0.5, align="left")
        axes[i].set_title(titles[i])
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlabel("Degrees")
    plt.show()

def plot_ccoef_distribution(empirical_ccoefs: list, baseline_ccoefs: list, vae_ccoefs: list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_ccoefs = [empirical_ccoefs, baseline_ccoefs, vae_ccoefs]
    # max_ccoef = max([max(d) for d in all_ccoefs])
    titles = ["MUTAG", "Baseline", "VAE"]
    for i, ccoefs in enumerate(all_ccoefs):
        # axes[i].hist(ccoefs, bins=np.arange(0, max_ccoef+1, 1), rwidth=0.5, align="left")
        axes[i].hist(ccoefs, bins=np.arange(0, 1.1, 0.1), rwidth=0.5, align="left")
        axes[i].set_title(titles[i])
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlabel("Cluestering Coefficients")
    plt.show()

def plot_centrals_distribution(empirical_centrals: list, baseline_centrals: list, vae_centrals: list):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    all_centrals = [empirical_centrals, baseline_centrals, vae_centrals]
    # max_ccoef = max([max(d) for d in all_ccoefs])
    titles = ["MUTAG", "Baseline", "VAE"]
    for i, centrals in enumerate(all_centrals):
        # axes[i].hist(ccoefs, bins=np.arange(0, max_ccoef+1, 1), rwidth=0.5, align="left")
        axes[i].hist(centrals, bins=np.arange(0, 0.7, 0.1), rwidth=0.5, align="left")
        axes[i].set_title(titles[i])
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlabel("Eigenvector centralities")
    plt.show()

if __name__ == "__main__":
    # evaluate baseline
    train = data.get_dataset()
    # train_loader = DataLoader(train, batch_size=1)
    # node_distribution, densities = node_stats(train_loader)
    # N, edge_probability = sample_empirical_graph(n_samples=1000, node_distribution=node_distribution, densities=densities)
    # As = []
    #for n, prob in zip(N, edge_probability):
    #    A = generate_graph_er(n, prob)
    #    A = A.numpy()
    #   A = A.astype(int)
    #   As.append(A)
    #ith open("base_graphs.pkl", "wb") as f:
    #   pickle.dump(As, f)
    
    with open("base_graphs.pkl", "rb") as f:
        As = pickle.load(f)
    base_graphs = [nx.from_numpy_array(A) for A in As]

    # evaluator = Evaluator(base_graphs)
    # novelty = evaluator.get_novelty()
    # uniqueness = evaluator.get_uniqueness()
    # novel_and_unique = evaluator.get_novel_and_unique()
    # print("Baseline statistics:")
    # print(f"Novelty: {novelty}, Uniqueness: {uniqueness}, Novel and Unique: {novel_and_unique}")

    vae_graphs = np.load("vae_graphs2.npy", allow_pickle=True)
    new_vae_graphs = []
    for g in vae_graphs:
        new_g = np.delete(g, g.sum(axis=1) == 0, axis=0)
        new_g = np.delete(new_g, new_g.sum(axis=0) == 0, axis=1)
        new_vae_graphs.append(new_g)
        
    vae_graphs = [nx.from_numpy_array(A) for A in vae_graphs]
    vae_degrees = get_degree_counts(vae_graphs)
    vae_ccoefs = get_ccoef_counts(vae_graphs)
    vae_centralities = get_centralities(vae_graphs)

    base_degrees = get_degree_counts(base_graphs)
    base_ccoefs = get_ccoef_counts(base_graphs)
    base_centralities = get_centralities(base_graphs)

    train_graphs = [to_networkx(g, to_undirected=True) for g in train]
    train_degrees = get_degree_counts(train_graphs)
    train_ccoefs = get_ccoef_counts(train_graphs)
    train_centralities = get_centralities(train_graphs)

    #for g in train_graphs:
    #   nx.draw(g, with_labels=True)
    #   print(nx.clustering(g))
        # print(nx.is_directed(g))
    #   print(nx.adjacency_matrix(g).todense())
    #   plt.show()
    
    # vae_degrees = get_degree_counts(vae_graphs)
    # vae_ccoefs = get_ccoef_counts(vae_graphs)
    # vae_centralities = get_centralities(vae_graphs)

    plot_degree_distribution(train_degrees, base_degrees, vae_degrees)

    plot_ccoef_distribution(train_ccoefs, base_ccoefs, vae_ccoefs)

    plot_centrals_distribution(train_centralities, base_centralities, vae_centralities)


