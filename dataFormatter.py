import json
import datetime
import os
import numpy as np
from scipy.optimize import curve_fit
import time
from multiprocessing import Pool, freeze_support, RLock
from enum import IntEnum
from tqdm import tqdm
import pickle

from settings import data_formatter as settings


class ChartRangeError(Exception):
    pass


def getLatestDataIndex(chart, t, max_index, min_index):
    center_index = int((max_index + min_index) / 2)

    if chart[center_index]["time"] <= t:
        max_index = center_index
    else:
        min_index = center_index

    if max_index - min_index <= 1:
        return max_index + 1
    else:
        return getLatestDataIndex(chart, t, max_index, min_index)


class Label(IntEnum):
    DOWN = 1  # 減少
    CONVEX_DOWN = 2  # 凸型減少
    CONCAVE_UP = 3  # 凹型増加
    UP = 4  # 増加


def getLabel(chart, t):
    base_index = getLatestDataIndex(chart, t.timestamp(), len(chart) - 1, 0)
    base_rate = chart[base_index]
    current_index = base_index
    max_rate = base_rate["close"]
    min_rate = base_rate["close"]
    while 1:
        current_index = current_index - 1
        if current_index < 0:
            raise ChartRangeError
        current_rate = chart[current_index]

        max_rate = max([max_rate, current_rate["close"]])
        min_rate = min([min_rate, current_rate["close"]])

        if max_rate - base_rate["close"] > settings["margin_pips"] / 100:
            if min_rate - base_rate["close"] < -settings["middle_pips"] / 100:
                label = Label.CONCAVE_UP
            else:
                label = Label.UP
            break
        elif min_rate - base_rate["close"] < -settings["margin_pips"] / 100:
            if max_rate - base_rate["close"] > settings["middle_pips"] / 100:
                label = Label.CONVEX_DOWN
            else:
                label = Label.DOWN
            break

    return label


def getApproximateParams(origin_chart, t, num):
    base_index = getLatestDataIndex(
        origin_chart, t.timestamp(), len(origin_chart) - 1, 0
    )
    try:
        partial_chart = [origin_chart[base_index + i] for i in range(num)]
    except IndexError as e:
        raise ChartRangeError

    params = []
    x = np.linspace(0, 1 - num, num)
    for column in ["open", "low", "high", "close"]:
        rate = np.array(
            [
                partial_chart[i][column] - partial_chart[0][column]
                for i in range(len(partial_chart))
            ]
        )
        params.extend(np.polyfit(x, rate, settings["approximate_dim"]))

    return params


def makeInputData(thread_num, data, start_i, end_i):
    result = []
    for i in tqdm(
        range(end_i - start_i),
        desc=f"Core {thread_num:>2}",
        position=thread_num + 1,
        disable=False,
        leave=False,
        mininterval=settings["prog_interval"],
    ):
        t = datetime.datetime.fromtimestamp(
            data[settings["base_interval"]]["chart"][start_i + i]["time"]
        )
        params = []

        try:
            for interval in settings["intervals"]:
                params.extend(
                    getApproximateParams(
                        data[interval]["chart"], t, settings["approximate_data_num"]
                    )
                )
            label = getLabel(data[settings["base_interval"]]["chart"], t)
        except ChartRangeError as e:
            continue

        result.append(
            {
                "time": data[settings["base_interval"]]["chart"][start_i + i]["time"],
                "label": label,
                "params": params,
            }
        )
    return result


def wrapper(args):
    return makeInputData(*args)


if __name__ == "__main__":
    symbol = settings["symbol"].replace("/", "-")

    data = {}
    for interval in settings["intervals"]:
        data[interval] = json.load(
            open("chart_log/" + symbol + "_" + interval + ".json")
        )
    start = time.time()
    output = []
    if settings["multi_thread"]:
        results = []
        cores = os.cpu_count() if settings["threads"] == "auto" else settings["threads"]
        split_index = np.linspace(
            0, len(data[settings["base_interval"]]["chart"]), cores + 1, dtype="int"
        )
        split_data = [
            (thread_num, data, split_index[thread_num], split_index[thread_num + 1])
            for thread_num in range(cores)
        ]
        freeze_support()
        with Pool(cores, initializer=tqdm.set_lock, initargs=(RLock(),)) as p:
            results = p.map(wrapper, split_data)
        for result in results:
            output.extend(result)
    else:
        output = makeInputData(
            0, data, 0, len(data[settings["base_interval"]]["chart"])
        )

    print("元データ数：", len(data[settings["base_interval"]]["chart"]))
    print("処理時間：", time.time() - start)
    print("処理データ数：", len(output))

    start = time.time()
    os.makedirs(settings["save_dir"], exist_ok=True)
    json_object = json.dumps(output, indent=2)
    with open(settings["save_dir"] + "data.json", "w") as outfile:
        outfile.write(json_object)
    print("保存処理時間：", time.time() - start)
