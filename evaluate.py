import networkx as nx
import data

def is_isomorphic(G1, G2) -> bool:
    G1_hash = nx.weisfeiler_lehman_graph_hash(G1)
    G2_hash = nx.weisfeiler_lehmen_graph_hash(G2)
    if G1_hash == G2_hash:
        return True
    return False

class Evaluator:
    def __init__(self, graphs):
        self.graphs = self.graphs
        self.train_graphs, _, _ = data.get_datasets()