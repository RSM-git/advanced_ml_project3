import sys
sys.path.extend("..")
from data import get_datasets
import torch

def test_get_datasets():
    train, val, test = get_datasets()
    assert type(train) == torch.utils.data.dataset.Subset