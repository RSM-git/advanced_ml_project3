from torch_geometric.datasets import TUDataset

def get_dataset():
    dataset = TUDataset(root='./data/', name='MUTAG')
    return dataset
    
