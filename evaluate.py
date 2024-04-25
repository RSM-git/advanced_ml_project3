import networkx as nx
import data
import numpy.typing as npt
from collections import defaultdict
from torch_geometric.utils import to_networkx
from erdos_renyi import generate_graph_er


class Evaluator:
    """
    params:
        graphs: list of graphs to evaluate
    """
    def __init__(self, graphs: list[nx.Graph]):
        self.graphs = graphs
        self.train_graphs, _, _ = data.get_datasets()
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
            if hash in self.train_hashes_set:
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
            novel_and_unique += (hash in self.train_hashes) and (eval_counts[hash] == 1)
        
        return novel_and_unique / self.n_eval

if __name__ == "__main__":
    # evaluate baseline
    base_graphs = [nx.from_numpy_array(A) for A in base_A]

    evaluator = Evaluator(base_graphs)
    novelty = evaluator.get_novelty()
    uniqueness = evaluator.get_uniqueness()
    novel_and_unique = evaluator.get_novel_and_unique()
    print("Baseline statistics:")
    print(f"Novelty: {novelty}, Uniqueness: {uniqueness}, Novel and Unique: {novel_and_unique}")
