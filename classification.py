import datetime
import json
import os
from matplotlib import pyplot as plt
import torch
import sys
from torch.utils.data import DataLoader
from data_set import ExchangeDataset
from torch import nn
import numpy as np

from learning import LearningBase
from plot_thread import PlotThread

np.set_printoptions(precision=3, suppress=True)

batch_size = 64
epochs = 20000
label_kinds = 2


class NeuralNetwork(nn.Module):
    def __init__(self):
        # data_size=5 * (7 * 10 + 10)
        data_size = 720
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(data_size, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, label_kinds),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def statistics_fn(statistics, label_kinds, pred, ans):
    for p in range(label_kinds):
        for a in range(label_kinds):
            statistics[p][a] += (
                torch.logical_and((pred.argmax(1) == p), (ans == a))
                .type(torch.float)
                .sum()
                .item()
            )


if __name__ == "__main__":
    args = sys.argv
    lb = LearningBase()
    lb.setModel(NeuralNetwork)

    args = sys.argv
    if len(args) == 2:
        lb.model_init(args[1])
        with open(args[1].replace(".pth", ".json")) as f:
            log = json.load(f)
        init_epoch = log[-1]["epoch"]
    else:
        log = []
        init_epoch = 0

    training_data = ExchangeDataset("formatted_data/train_data.npz", noise=True)
    test_data = ExchangeDataset("formatted_data/test_data.npz", noise=False)

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

    lb.setDataLoader(train_dataloader, test_dataloader)

    weight = torch.reciprocal(training_data.label_distribution())
    loss_fn = nn.CrossEntropyLoss(weight=weight.to(lb.device))
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.BCEWithLogitsLoss()
    lb.setLossFn(loss_fn)
    # optimizer = torch.optim.SGD(lb.model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(lb.model.parameters())
    lb.set_optimizer(optimizer)

    plot_thread = PlotThread()
    plot_thread.start()

    try:
        for t in range(init_epoch, init_epoch + epochs):
            print(
                f"Epoch {t+1} / {init_epoch + epochs}\n-------------------------------"
            )
            train_correct, train_loss = lb.train()
            test_correct, test_loss, result_mat = lb.test(statistics_fn)
            print(result_mat / len(test_dataloader.dataset))
            print(
                f"train_correct: {(100*train_correct):>0.2f}%, train_loss:{train_loss:>8f}",
                f"test_correct: {(100*test_correct):>0.2f}%, test_loss:{test_loss:>8f} \n",
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
            plot_thread.update(log)
    except KeyboardInterrupt:
        # raise KeyboardInterrupt
        pass

    print("Done!")

    os.makedirs("model/", exist_ok=True)
    file_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lb.saveModel(f"model/{file_name}.pth")
    open(f"model/{file_name}.json", "w").write(json.dumps(log, indent=4))
    plt.savefig(f"model/{file_name}.png")
    print(f"Saved PyTorch Model State to {file_name}.pth")
    plot_thread.kill()
