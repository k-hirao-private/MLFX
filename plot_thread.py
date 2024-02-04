import warnings
import matplotlib.pyplot as plt
from threading import Thread
import numpy as np
import time
import json

warnings.filterwarnings("ignore", category=UserWarning, module="plot_thread", lineno=49)


class PlotThread(Thread):
    killed = False
    updated = False
    log = []
    interval = 1

    axs = None

    def kill(self):
        self.killed = True
        self.join()

    def update(self, data):
        self.data = data
        self.updated = True

    def plot(self, t):
        data = self.data
        x = np.linspace(1, t, t)
        self.axs[0].cla()
        self.axs[0].plot(x, [l["train_correct"] for l in data], label="train_correct")
        self.axs[0].plot(x, [l["test_correct"] for l in data], label="test_correct")
        self.axs[0].grid()
        self.axs[0].legend()

        self.axs[1].cla()
        self.axs[1].plot(x, [l["train_loss"] for l in data], label="train_loss")
        self.axs[1].plot(x, [l["test_loss"] for l in data], label="test_loss")
        self.axs[1].grid()
        self.axs[1].legend()

    def run(self):
        if self.axs is None:
            fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        while not self.killed:
            if self.updated:
                self.plot(self.data_size())
                self.updated = False
            plt.pause(self.interval)
        plt.close()

    def data_size(self):
        return len(self.data)


if __name__ == "__main__":
    with open("model/2023-10-26_22-24-57.json") as f:
        data = json.load(f)

    plot_thread = PlotThread()
    plot_thread.start()
    for t in range(len(data)):
        plot_thread.update(data[: t + 1])
        time.sleep(0.1)
    plot_thread.kill()
