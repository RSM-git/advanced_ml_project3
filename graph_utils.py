import networkx as nx
import numpy as np
import torch

def indices_no_diagonal(matrix_indices):
    """
    matrix_indices: (2, N) 
    """
    index_r, index_c = matrix_indices
    diagonal_mask = index_r == index_c
    index_r = index_r[~diagonal_mask]
    index_c = index_c[~diagonal_mask]
    return index_r, index_c

def tri_to_adj_matrix(tris, n_nodes):
    """
    ** tris DOES NOT INCLUDE the diagonal (should be of dim (matrix_dim - 1) * (matrix_dim - 2) / 2)
    tris: (B, N) tensor containing the upper or lower triangular values of the matrix (corresponding to edges)
    n_nodes: side length of adj matrix
    """
    B, N = tris.shape
    
    M = torch.zeros((B, n_nodes, n_nodes), device = tris.device)
    lower_r, lower_c = indices_no_diagonal(torch.tril_indices(n_nodes, n_nodes))
    
    
    M[:, lower_r, lower_c] = tris
    M = M + M.mT

    assert (M.mT == M).all()

    return M

def draw_graph(A):
    G = nx.from_numpy_array(A.numpy())
    fig = nx.draw(G, with_labels=True)
    return fig    

def save_graph(A):
    np.save('graphs.npy', A.numpy())