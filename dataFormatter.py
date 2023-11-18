import json
import datetime
import os
import numpy as np
import time
from multiprocessing import Pool, freeze_support, RLock
from enum import IntEnum
from tqdm import tqdm

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
    DOWN = 0  # 減少
    FLAT = 1  # 横ばい
    UP = 2  # 増加
    CONVEX_DOWN = 3  # 凸型減少
    CONCAVE_UP = 4  # 凹型増加


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

        max_rate = max([max_rate, current_rate["high"]])
        min_rate = min([min_rate, current_rate["low"]])

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
        elif base_index - current_index >= settings["label_data_range"]:
            label = Label.FLAT
            break

    return label


def getParams(origin_chart, t, num):
    base_index = getLatestDataIndex(
        origin_chart, t.timestamp(), len(origin_chart) - 1, 0
    )
    try:
        partial_chart = [origin_chart[base_index + i] for i in range(num)]
    except IndexError as e:
        raise ChartRangeError

    params = []
    x = np.linspace(0, 1 - num, num)
    for column in settings["columns"]:
        if settings["approximation"]:
            rate = np.array(
                [
                    partial_chart[i][column] - partial_chart[0][settings["base_column"]]
                    for i in range(len(partial_chart))
                ]
            )
            params.extend(np.polyfit(x, rate, settings["approximate_dim"]))
        else:
            params.extend([d[column] - partial_chart[0][column] for d in partial_chart])

    return params


def makeInputData(thread_num, data, start_i, end_i):
    params, labels = [], []
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
        param = []

        try:
            for interval in settings["intervals"]:
                param.extend(
                    getParams(data[interval]["chart"], t, settings["data_num"])
                )
            label = getLabel(data[settings["base_interval"]]["chart"], t)
        except ChartRangeError as e:
            continue
        labels.append(label)
        params.append(param)
    return labels, params


def wrapper(args):
    return makeInputData(*args)


if __name__ == "__main__":
    symbol = settings["symbol"].replace("/", "-")

    data = {}
    for interval in settings["intervals"]:
        with open("chart_log/" + symbol + "_" + interval + ".json") as f:
            data[interval] = json.load(f)
    start = time.time()
    labels, params = [], []
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

        for i in range(len(results)):
            labels.extend(results[i][0])
            params.extend(results[i][1])
            results[i] = None
    else:
        labels, params = makeInputData(
            0, data, 0, len(data[settings["base_interval"]]["chart"])
        )
    labels = np.array(labels, dtype=np.int8)
    params = np.array(params, dtype=np.float32)

    print("元データ数：", len(data[settings["base_interval"]]["chart"]))
    print("処理時間：", time.time() - start)
    print("処理データ数：", len(params))

    start = time.time()
    os.makedirs(settings["save_dir"], exist_ok=True)
    split_index = int(len(params) * 0.2)
    test_params, train_params = params[:split_index], params[split_index:]
    test_labels, train_labels = labels[:split_index], labels[split_index:]
    np.savez(
        settings["save_dir"] + "train_data",
        params=np.array(train_params),
        labels=np.array(train_labels),
    )
    np.savez(
        settings["save_dir"] + "test_data",
        params=np.array(test_params),
        labels=np.array(test_labels),
    )
    print("保存処理時間：", time.time() - start)
