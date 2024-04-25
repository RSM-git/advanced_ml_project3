from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

from graph_convolution import SimpleGraphConv
from vae import VAE, BernoulliDecoder, GaussianEncoder, GaussianPrior

def single_depth_graph_generator(node_feature_dim, max_graph_nodes, latent_dim, filter_length, intermediate_dim=8):
    """
    Create a vae with a 1-deep graph convolutional network
    - node_feature_dim is dimension of node features
    - latent_dim is latent dimension of VAE
    - max_graph_nodes is the length of the adjacency matrix to compute 
    - filter_length is length of convolutional filter (how long are the paths each node should pay attention to)
    """
    encoder_net = SimpleGraphConv(node_feature_dim, latent_dim, filter_length)
    decoder_net = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, max_graph_nodes)
            ),
    
    encoder = GaussianEncoder(encoder_net)
    decoder = BernoulliDecoder(decoder_net)
    
    prior = GaussianPrior(latent_dim)
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
    num_steps = len(data_loader)*epochs
    epoch = 0

    
    running_loss = 0.0
    training_losses = []

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            
            x = next(iter(data_loader))[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 100 == 99:    # every 100 mini-batches...
                training_losses.append(running_loss)
                running_loss = 0 

            # Report
            if step % 5 ==0 :
                loss = loss.detach().cpu()
                pbar.set_description(f"epoch={epoch}, step={step}, loss={loss:.1f}")

            if (step+1) % len(data_loader) == 0:
                epoch += 1


def main():
    MUTAG_DATASET_ROOT= '/zhome/c7/2/208212/classes/adv_ml/advanced_ml_e8/data' # './data'
    NODE_FEATURE_DIM = 7
    MAX_GRAPH_NODES = 20
    LATENT_DIM = 4
    convolutional_filter_length = 3
    epochs = 100
    device = 'cuda'
    

    train_loader, val_loader, test_loader = get_MUTAG_train_test(MUTAG_DATASET_ROOT)
    model = single_depth_graph_generator(NODE_FEATURE_DIM, MAX_GRAPH_NODES,LATENT_DIM, convolutional_filter_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.)
    
    train(model, optimizer, train_loader, epochs, device)

if __name__=='__main__':
    main()