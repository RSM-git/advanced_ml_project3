# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.1 (2024-01-29)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
import os


import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.decomposition import PCA as PCA
import matplotlib.pyplot as plt

import numpy as np



class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=torch.exp(self.std)), 1)

class MoGPrior(nn.Module):

    def __init__(self, M, K):
        """
        Define a Mixture of Gaussians prior distribution.

        Parameters:
        M: [int] 
           Dimension of the latent space.
        K: [int]
           Number of components in the mixture.
        """
        super(MoGPrior, self).__init__()
        self.M = M
        self.K = K
        self.mean = nn.Parameter(torch.randn(self.K, self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.K, self.M), requires_grad=False)
        self.weights = nn.Parameter(torch.ones(self.K), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.MixtureSameFamily(
            td.Categorical(self.weights), 
            td.Independent(td.Normal(loc=self.mean, scale=torch.exp(self.std)), 1)
            )



class GraphGaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]             
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GraphGaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x.x, x.edge_index, x.batch), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Multivariate Gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):  
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        mean = self.decoder_net(z)
        ok = td.Normal.arg_constraints["scale"].check(self.std)

        return td.Independent(td.Normal(loc=mean, scale=torch.exp(self.std)), 2)
        
class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)

    def get_p(self, z):
        """
        Returns the most likely value of the distribution 
        """
        return self.decoder_net(z)
    
    def get_manifold_curve_length(self, c, N):
        """
        Returns the length of the curve c in the manifold defined by the decoder
        c is defined in latent space`
        N is the number of points to sample on the curve
        """
        all_t = np.linspace(0, 1, N)
        z = c(all_t)
        x = self.decoder_net(z).flatten()

        return np.sum(np.linalg.norm(np.diff(x), axis=0))
        

def monte_carlo_kl(q, p, n_samples):
    """
    Compute a Monte Carlo estimate of the KL divergence between two distributions.

    Parameters:
    q: [torch.distributions.Distribution] 
       The first distribution.
    p: [torch.distributions.Distribution] 
       The second distribution.
    n_samples: [int]
       Number of samples to use for the Monte Carlo estimate.
    """
    z = q.rsample(torch.Size([n_samples]))
    return torch.mean(q.log_prob(z) - p.log_prob(z), dim=0)

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """
    def __init__(self, prior, encoder, decoder):
        """
        Parameters:
        prior: [torch.nn.Module] 
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """
            
        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """

        q = self.encoder(x)
        z = q.rsample()

        targets = x.adj.flatten(start_dim=1)
        elbo = torch.mean(self.decoder(z).log_prob(targets) - monte_carlo_kl(q, self.prior(), 100), dim=0)
        return elbo, z

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        elbo, z = self.elbo(x)
        return -elbo


def test(model, data_loader, device):
    """
    Evaluate a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    device: [torch.device]
        The device to use for evaluation.
    """
    model.eval()
    zs = []
    ys = []
    
    with torch.no_grad():
        loss = 0
        for x, y in data_loader:
            x = x.to(device)
            elbo = -model(x)
            loss += elbo.item()
        loss /= len(data_loader)
    print(f'{loss:.1f}')
   # zs_np = torch.cat(zs, dim=0).cpu().numpy()
   # ys_np = torch.cat(ys, dim=0).cpu().numpy()
   #plot_pca(zs_np, ys_np)
    return loss

def plot_pca(zs, ys):
    """
    Plots the first two principal components of the latent space represented by zs against their labels
    zs: [np.ndarray], shape (N, C)
    ys: [np.ndarray], shape (N,)
    """
    _, C = zs.shape
    pca = PCA(2)

    if C > 2:
        pca.fit(zs)
        zs_pca = pca.transform(zs)
        scatter_x = zs_pca[:,0]
        scatter_y = zs_pca[:,1]
    else:
        scatter_x = zs[:,0]
        scatter_y = zs[:,1]

    _, ax = plt.subplots()
    for g in np.unique(ys):
        i = np.nonzero(ys == g)
        ax.scatter(scatter_x[i], scatter_y[i], label=g)
    ax.legend()
    plt.show()
