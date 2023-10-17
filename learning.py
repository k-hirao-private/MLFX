import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import json
import datetime
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)


from data_set import ExchangeDataset

batch_size = 512
label_kinds = 2
epochs = 7000


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(175, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, label_kinds),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(
        tqdm(
            dataloader,
            desc=f"Epoch {epoch}",
            disable=False,
            leave=False,
            ncols=80,
            unit="batch",
            # dynamic_ncols=True,
        )
    ):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= num_batches
    correct /= size
    return correct, train_loss.item()


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    result_mat = np.zeros((label_kinds, label_kinds))
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for p in range(label_kinds):
                for a in range(label_kinds):
                    result_mat[p][a] += (
                        torch.logical_and((pred.argmax(1) == p), (y == a))
                        .type(torch.float)
                        .sum()
                        .item()
                    )
    test_loss /= num_batches
    correct /= size
    print(result_mat / size)

    return correct, test_loss, result_mat


if __name__ == "__main__":
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    model = NeuralNetwork().to(device)

    args = sys.argv
    if len(args) == 2:
        print(args[1])
        with open(args[1].replace(".pth", ".json")) as f:
            log = json.load(f)
        init_epoch = log[-1]["epoch"]

        model.load_state_dict(torch.load(args[1]))
    else:
        log = []
        init_epoch = 0

    training_data = ExchangeDataset("formatted_data/train_data.npz")
    test_data = ExchangeDataset("formatted_data/test_data.npz")

    # Create data loaders.
    train_dataloader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    try:
        for t in range(init_epoch, init_epoch + epochs):
            print(
                f"Epoch {t+1} / {init_epoch + epochs}\n-------------------------------"
            )
            train_correct, train_loss = train(
                train_dataloader, model, loss_fn, optimizer, t + 1
            )
            test_correct, test_loss, result_mat = test(test_dataloader, model, loss_fn)
            print(
                f"train_correct: {(100*train_correct):>0.1f}%, train_loss:{train_loss:>8f}",
                f"test_correct: {(100*test_correct):>0.1f}%, test_loss:{test_loss:>8f} \n",
            )
            log.append(
                {
                    "epoch": t + 1,
                    "train_correct": train_correct,
                    "train_loss": train_loss,
                    "test_correct": test_correct,
                    "test_loss": test_loss,
                    "result_mat": result_mat.tolist(),
                }
            )

            plt.cla()
            x = np.linspace(1, t + 1, t + 1)
            plt.plot(x, [l["train_correct"] for l in log], label="train_correct")
            plt.plot(x, [l["train_loss"] for l in log], label="train_loss")
            plt.plot(x, [l["test_correct"] for l in log], label="test_correct")
            plt.plot(x, [l["test_loss"] for l in log], label="test_loss")
            plt.grid()
            plt.legend()
            plt.pause(0.1)
    except KeyboardInterrupt:
        pass

    print("Done!")

    os.makedirs("model/", exist_ok=True)
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), f"model/{file_name}.pth")
    open(f"model/{file_name}.json", "w").write(json.dumps(log, indent=4))
    plt.savefig(f"model/{file_name}.png")
    print(f"Saved PyTorch Model State to {file_name}.pth")
