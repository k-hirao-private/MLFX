from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import numpy as np


class ExchangeDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.labels = data["labels"]
        self.params = data["params"]

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        input = torch.FloatTensor(self.params[idx])
        label = int(self.labels[idx])
        return input, label
