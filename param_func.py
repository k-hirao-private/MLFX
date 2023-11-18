import numpy as np


def candle(partial_chart):
    candle = np.array(
        [
            partial_chart[i]["close"] - partial_chart[i]["open"]
            for i in range(len(partial_chart))
        ]
    )
    return candle
