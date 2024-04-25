import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def get_datasets(seed: int = 0):
    dataset = TUDataset(root='./data/', name='MUTAG')
    rng = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)
    return train_dataset, validation_dataset, test_dataset
    