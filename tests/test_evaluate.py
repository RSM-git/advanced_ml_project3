import sys
sys.path.extend("..")
import evaluate
import numpy as np
import networkx as nx

def test_Evaluator():
    A1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    A2 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    A3 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    eval_graphs = [nx.from_numpy_array(A1), nx.from_numpy_array(A2), nx.from_numpy_array(A3)]

    evaluator = evaluate.Evaluator(eval_graphs)
    novelty = evaluator.get_novelty()
    uniqueness = evaluator.get_uniqueness()
    novel_and_unique = evaluator.get_novel_and_unique()
