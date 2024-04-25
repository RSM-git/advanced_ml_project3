import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


dataset = TUDataset(root='./data/', name='MUTAG')

rng = torch.Generator().manual_seed(0)
train_dataset, validation_dataset, test_dataset = random_split(dataset, (100, 44, 44), generator=rng)


train_loader = DataLoader(train_dataset, batch_size=100)

validation_loader = DataLoader(validation_dataset, batch_size=44)

test_loader = DataLoader(test_dataset, batch_size=44)


