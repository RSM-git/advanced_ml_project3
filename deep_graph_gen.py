from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader

import matplotlib.pyplot as plt

from graph_convolution import SimpleGraphConv
from vae import VAE, BernoulliDecoder, GraphGaussianEncoder, GaussianPrior
from graph_utils import draw_graph, indices_no_diagonal, tri_to_adj_matrix, save_graph

MAX_GRAPH_NODES = 28  # maximum graph size in the training set


def single_depth_graph_generator(node_feature_dim, max_graph_nodes, latent_dim, filter_length, intermediate_dim=8,
                                 device='cuda'):
    """
    Create a vae with a 1-deep graph convolutional network
    - node_feature_dim is dimension of node features
    - latent_dim is latent dimension of VAE
    - max_graph_nodes is the length of the adjacency matrix to compute 
    - filter_length is length of convolutional filter (how long are the paths each node should pay attention to)
    """

    lower_triangle_length = (max_graph_nodes) * (max_graph_nodes - 1) >> 1  # does not include the diagonal

    encoder_net = SimpleGraphConv(node_feature_dim, 2 * latent_dim, filter_length)
    decoder_net = nn.Sequential(
        nn.Linear(latent_dim, intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, intermediate_dim),
        nn.ReLU(),
        nn.Linear(intermediate_dim, lower_triangle_length)
    )

    encoder = GraphGaussianEncoder(encoder_net).to(device)
    decoder = BernoulliDecoder(decoder_net).to(device)

    prior = GaussianPrior(latent_dim).to(device)
    model = VAE(prior=prior, encoder=encoder, decoder=decoder)

    return model


def get_MUTAG_dataloader(root):
    dataset = TUDataset(root=root, name='MUTAG')

    train_loader = DataLoader(dataset, batch_size=100)

    return train_loader


def padded_lower_triangular(edge_index, batch, max_graph_nodes):
    adj = to_dense_adj(edge_index, batch)
    _, h, w = adj.shape
    adj = F.pad(adj, (0, max_graph_nodes - h, 0, max_graph_nodes - h))
    _, h, w = adj.shape
    index_r, index_c = indices_no_diagonal(torch.triu_indices(h, w))

    adj_tri = adj[:, index_r, index_c]

    return adj_tri


def train(model, optimizer, data_loader, epochs, device, save_path=None):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()
    epoch = 0

    training_losses = []

    with tqdm(range(epochs)) as pbar:
        for step in pbar:
            running_loss = 0.0
            for data in data_loader:
                # x = next(iter(data_loader))

                data.adj = padded_lower_triangular(data.edge_index, data.batch, MAX_GRAPH_NODES)
                data = data.to(device)
                optimizer.zero_grad()
                loss = model(data)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(data_loader)
            pbar.set_description(f"epoch={epoch}, loss={epoch_loss:.1f}")
            training_losses.append(epoch_loss)

    plt.plot(training_losses)
    plt.savefig('train_loss.png')
    if save_path is not None:
        torch.save(model.state_dict(), save_path)


def sample(model_weight_path, n_samples, node_feature_dim, max_graph_nodes, latent_dim, convolutional_filter_length,
           save_path='vae_graph.png', device='cuda'):
    model = single_depth_graph_generator(node_feature_dim, max_graph_nodes, latent_dim, convolutional_filter_length)

    state_dict = torch.load(model_weight_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    with torch.no_grad():
        samples = model.sample(n_samples).cpu()

    B, _ = samples.shape
    adjs = tri_to_adj_matrix(samples, max_graph_nodes)

    match n_samples:
        case 1:
            draw_graph(adjs[0])
            plt.savefig(save_path)
        case _:
            save_graph(adjs, save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'],
                        help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--weights', type=str, default='models/vae_basic.pt')
    parser.add_argument("--n_samples", type=int, default=1, help="number of samples to generate")
    parser.add_argument("--fname", type=str, default="graphs.npy", help="filename for saving graph samples")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="device to use (default: %(default)s)")

    args = parser.parse_args()

    MUTAG_DATASET_ROOT = './data'

    NODE_FEATURE_DIM = 7
    LATENT_DIM = 8
    CONVOLUTIONAL_FILTER_LENGTH = 3

    epochs = 5000
    device = 'cuda'

    if args.mode == 'train':
        train_loader = get_MUTAG_dataloader(MUTAG_DATASET_ROOT)
        model = single_depth_graph_generator(NODE_FEATURE_DIM, MAX_GRAPH_NODES, LATENT_DIM, CONVOLUTIONAL_FILTER_LENGTH)

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

        train(model, optimizer, train_loader, epochs, device=device, save_path=args.weights)
    elif args.mode == 'sample':
        weight_path = args.weights
        sample(weight_path, args.n_samples, NODE_FEATURE_DIM, MAX_GRAPH_NODES, LATENT_DIM, CONVOLUTIONAL_FILTER_LENGTH,
               save_path=args.fname)


if __name__ == '__main__':
    main()
