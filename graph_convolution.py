# %%
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch



# %% Define a simple graph convolution for graph classification
class SimpleGraphConv(torch.nn.Module):
    """Simple graph convolution for graph classification

    Keyword Arguments
    -----------------
        node_feature_dim : Dimension of the node features
        filter_length : Length of convolution filter
    """

    def __init__(self, node_feature_dim, output_dim, filter_length):
        super().__init__()

        # Define dimensions and other hyperparameters
        self.node_feature_dim = node_feature_dim
        self.filter_length = filter_length

        # Define graph filter
        self.h = torch.nn.Parameter(1e-5*torch.randn(filter_length))
        self.h.data[0] = 1.

        # State output network
        self.output_net = torch.nn.Linear(self.node_feature_dim, output_dim)

        self.cached = False


    def forward(self, x, edge_index, batch, eigen = True):
        """Evaluate neural network on a batch of graphs.

        Parameters
        ----------
        x : torch.tensor (num_nodes x num_features)
            Node features.
        edge_index : torch.tensor (2 x num_edges)
            Edges (to-node, from-node) in all graphs.
        batch : torch.tensor (num_nodes)
            Index of which graph each node belongs to.

        Returns
        -------
        out : torch tensor (num_graphs)
            Neural network output for each graph.

        """
        # Extract number of nodes and graphs
        num_graphs = batch.max()+1

        # Compute adjacency matrices and node features per graph
        A = to_dense_adj(edge_index, batch)
        X, idx = to_dense_batch(x, batch)
 
        # ---------------------------------------------------------------------------------------------------------

        # Implementation in vertex domain
        

        if eigen:
            eigenvalues, eigenvectors = torch.linalg.eigh(A)
            
            spectral_node_state = torch.zeros_like(eigenvalues)
            for k in range(self.filter_length):
                spectral_node_state += self.h[k] * eigenvalues ** k

            node_state = eigenvectors @ torch.diag_embed(spectral_node_state) @ eigenvectors.mT @ X
        else:
            node_state = torch.zeros_like(X)
            for k in range(self.filter_length):
                node_state += self.h[k] * torch.linalg.matrix_power(A, k) @ X

        # TODO: Comment out the above three lines, and re-implement the graph
        #       convolution in the spectral domain. Check that the two 
        #       implementations yield identical results

        # ---------------------------------------------------------------------------------------------------------

        # Aggregate the node states
        graph_state = node_state.sum(1)

        # Output
        out = self.output_net(graph_state)
        return out
    

def train():
    # %% Set up the model, loss, and optimizer etc.
    # Instantiate the model
    filter_length = 3
    torch.manual_seed(1)
    model = SimpleGraphConv(node_feature_dim, filter_length).to(device)

    # Loss function
    cross_entropy = torch.nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1.)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

    # %% Lists to store accuracy and loss
    train_accuracies = []
    train_losses = []
    validation_accuracies = []
    validation_losses = []

    # %% Fit the model
    # Number of epochs
    epochs = 500

    for epoch in range(epochs):
        # Loop over training batches
        model.train()
        train_accuracy = 0.
        train_loss = 0.
        for data in train_loader:
            out = model(data.x, data.edge_index, batch=data.batch, eigen=True)
            loss = cross_entropy(out, data.y.float())

            # Gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training loss and accuracy
            train_accuracy += sum((out>0) == data.y).detach().cpu() / len(train_loader.dataset)
            train_loss += loss.detach().cpu().item() * data.batch_size / len(train_loader.dataset)
        
        # Learning rate scheduler step
        scheduler.step()

        # Validation, print and plots
        with torch.no_grad():    
            model.eval()
            # Compute validation loss and accuracy
            validation_loss = 0.
            validation_accuracy = 0.
            for data in validation_loader:
                out = model(data.x, data.edge_index, data.batch, eigen=True)
                validation_accuracy += sum((out>0) == data.y).cpu() / len(validation_loader.dataset)
                validation_loss += cross_entropy(out, data.y.float()).cpu().item() * data.batch_size / len(validation_loader.dataset)

            # Store the training and validation accuracy and loss for plotting
            train_accuracies.append(train_accuracy)
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)

            # Print stats and update plots
            if (epoch+1)%50 == 0:
                print(f'Epoch {epoch+1}')
                print(f'- Learning rate   = {scheduler.get_last_lr()[0]:.1e}')
                print(f'- Train. accuracy = {train_accuracy:.3f}')
                print(f'         loss     = {train_loss:.3f}')
                print(f'- Valid. accuracy = {validation_accuracy:.3f}')
                print(f'         loss     = {validation_loss:.3f}')

                plt.figure('Loss').clf()
                plt.plot(train_losses, label='Train')
                plt.plot(validation_losses, label='Validation')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Cross entropy')
                plt.yscale('log')
                plt.tight_layout()
                plt.savefig('loss.png')
                plt.clf()
                

                plt.figure('Accuracy').clf()
                plt.plot(train_accuracies, label='Train')
                plt.plot(validation_accuracies, label='Validation')
                plt.legend()
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.tight_layout()
                plt.savefig('acc.png')
                plt.clf()

    # %% Save final predictions.
    with torch.no_grad():
        data = next(iter(test_loader))
        out = model(data.x, data.edge_index, data.batch).cpu()
        torch.save(out, 'test_predictions.pt')
