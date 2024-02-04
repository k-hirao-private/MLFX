import datetime
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from plot_thread import PlotThread


class LearningBase:
    def __init__(self, device="auto"):
        if device == "auto":
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        elif device in ["cuda", "mps", "cpu"]:
            self.device = device
        print(f"Using {device} device")

    def setModel(self, model):
        self.model = model().to(self.device)

    def model_init(self, pth_path):
        if self.model in locals():
            self.model.load_state_dict(torch.load(pth_path))
        else:
            "First, Please set a model."

    def setDataLoader(self, train, test):
        self.train_dataloader = train
        self.test_dataloader = test

    def setLossFn(self, loss):
        self.loss_fn = loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, correct_fn):
        size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        self.model.train()
        train_loss, correct = 0, 0
        for batch, (X, y) in enumerate(
            tqdm(
                self.train_dataloader,
                desc=f"Train:",
                disable=False,
                leave=False,
                ncols=80,
                unit="batch",
                mininterval=0.5,
                # dynamic_ncols=True,
            )
        ):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.model(X)
            loss = self.loss_fn(pred.squeeze(), y)

            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            train_loss += loss
            correct += correct_fn(pred, y)
            # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        train_loss /= num_batches
        correct /= size
        return correct, train_loss.item()

    def test(self, correct_fn, statistics_fn, statistics):
        size = len(self.test_dataloader.dataset)
        num_batches = len(self.test_dataloader)
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(
                tqdm(
                    self.test_dataloader,
                    desc=f"Test:",
                    disable=False,
                    leave=False,
                    ncols=80,
                    unit="batch",
                    mininterval=0.5,
                    # dynamic_ncols=True,
                )
            ):
                X, y = X.to(self.device), y.to(self.device)

                pred = self.model(X)

                test_loss += self.loss_fn(pred.squeeze(), y).item()
                correct += correct_fn(pred, y)
                # correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                statistics_fn(statistics, pred, y, batch)

        test_loss /= num_batches
        correct /= size

        return correct, test_loss

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)
