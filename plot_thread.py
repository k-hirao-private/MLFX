import matplotlib.pyplot as plt
from threading import Thread
import numpy as np
import time
import json


class PlotThread(Thread):
    killed = False
    updated = False
    log = []
    interval = 1

    axs = None

    def kill(self):
        self.killed = True

    def update(self, log):
        self.log = log
        self.updated = True

    def plot(self, t):
        log = self.log
        x = np.linspace(1, t, t)
        self.axs[0].cla()
        self.axs[0].plot(x, [l["train_correct"] for l in log], label="train_correct")
        self.axs[0].plot(x, [l["test_correct"] for l in log], label="test_correct")
        self.axs[0].grid()
        self.axs[0].legend()

        self.axs[1].cla()
        self.axs[1].plot(x, [l["train_loss"] for l in log], label="train_loss")
        self.axs[1].plot(x, [l["test_loss"] for l in log], label="test_loss")
        self.axs[1].grid()
        self.axs[1].legend()

    def run(self):
        fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        while not self.killed:
            if self.updated:
                self.plot(len(self.log))
                self.updated = False
            plt.pause(self.interval)
        plt.close()


if __name__ == "__main__":
    with open("model/2023-10-26_22-24-57.json") as f:
        log = json.load(f)

    plot_thread = PlotThread()
    plot_thread.start()
    for t in range(len(log)):
        plot_thread.update(log[: t + 1])
        time.sleep(0.1)
    plot_thread.kill()
    plot_thread.join()
