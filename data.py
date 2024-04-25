import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def get_dataset():
    dataset = TUDataset(root='./data/', name='MUTAG')
    return dataset
    