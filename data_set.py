from torchvision.io import read_image
from torch.utils.data import Dataset
import torch
import numpy as np
import collections


class ExchangeDataset(Dataset):
    def __init__(self, data_path, noise):
        data = np.load(data_path)
        self.labels = data["labels"]
        self.params = data["params"]
        self.noise = noise
        self.label_kinds = len(collections.Counter(self.labels).keys())

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        input = self.params[idx]
        label = int(self.labels[idx])
        if self.noise:
            input += np.random.normal(loc=0, scale=0.01, size=(len(self.params[idx])))
            if np.random.rand() < 0.20:
                label = int(np.random.rand() * self.label_kinds)

        input = torch.FloatTensor(input)
        return input, label

    def label_distribution(self):
        c = collections.Counter(self.labels)
        return torch.tensor([c[l] for l in range(len(c.keys()))])
