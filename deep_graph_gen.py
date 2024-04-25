from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader

from graph_convolution import SimpleGraphConv
from vae import VAE, BernoulliDecoder, GraphGaussianEncoder, GaussianPrior

MAX_GRAPH_NODES = 29 # maximum graph size in the training set

def single_depth_graph_generator(node_feature_dim, max_graph_nodes, latent_dim, filter_length, intermediate_dim=8, device='cuda'):
    """
    Create a vae with a 1-deep graph convolutional network
    - node_feature_dim is dimension of node features
    - latent_dim is latent dimension of VAE
    - max_graph_nodes is the length of the adjacency matrix to compute 
    - filter_length is length of convolutional filter (how long are the paths each node should pay attention to)
    """
    encoder_net = SimpleGraphConv(node_feature_dim, 2*latent_dim, filter_length)
    decoder_net = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, max_graph_nodes ** 2 )
            )
    
    encoder = GraphGaussianEncoder(encoder_net).to(device)
    decoder = BernoulliDecoder(decoder_net).to(device)
    
    prior = GaussianPrior(latent_dim).to(device)
    model = VAE(prior=prior, encoder=encoder, decoder=decoder)

    return model


def get_MUTAG_train_test(root):
    dataset = TUDataset(root=root, name='MUTAG')

    rng = torch.Generator().manual_seed(0)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)

    train_loader = DataLoader(train_dataset, batch_size=100)
    validation_loader = DataLoader(validation_dataset, batch_size=44)
    test_loader = DataLoader(test_dataset, batch_size=44)

    return train_loader, validation_loader, test_loader

def train(model, optimizer, data_loader, epochs, device):
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
                x = next(iter(data_loader))
                adj = to_dense_adj(x.edge_index, x.batch)

                target_size = MAX_GRAPH_NODES - adj.shape[1]
                adj = F.pad(adj, (0,  target_size, 0, target_size))
                x.adj = adj
                x = x.to(device)
                optimizer.zero_grad()
                loss = model(x)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
                if step % 100 == 99:    # every 100 mini-batches...
                    running_loss = 0 

            epoch_loss = running_loss / len(data_loader)
            pbar.set_description(f"epoch={epoch}, loss={epoch_loss:.1f}")
            training_losses.append(epoch_loss)

def main():
    MUTAG_DATASET_ROOT= '/zhome/c7/2/208212/classes/adv_ml/advanced_ml_e8/data' # './data'
    NODE_FEATURE_DIM = 7
    LATENT_DIM = 4
    convolutional_filter_length = 3
    epochs = 500
    device = 'cuda'
    

    train_loader, val_loader, test_loader = get_MUTAG_train_test(MUTAG_DATASET_ROOT)
    model = single_depth_graph_generator(NODE_FEATURE_DIM, MAX_GRAPH_NODES,LATENT_DIM, convolutional_filter_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    
    train(model, optimizer, train_loader, epochs, device)

if __name__=='__main__':
    main()