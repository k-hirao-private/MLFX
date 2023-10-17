import os
import json
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch


class ExchangeDataset(Dataset):
    def __init__(self, data_path):
        self.data = json.load(open(data_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input = torch.FloatTensor(self.data[idx]["params"])
        label = self.data[idx]["label"]
        return input, label
