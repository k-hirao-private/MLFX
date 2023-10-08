import json
import pandas as pd
import mplfinance as mpf


def mpf_plot(chart):
    df = pd.DataFrame(chart)
    df["date"] = pd.to_datetime(df["time"], unit="s")
    df = df.drop(columns="time")
    df = df.set_index("date")
    df = df.iloc[::-1]

    mpf.plot(
        df,
        type="candle",
        # volume=True,
        # mav=[5, 12],
        figratio=(2, 1),
    )


if __name__ == "__main__":
    data = json.load(open("chart_log/USD-JPY_1h.json"))
    chart = data["chart"]
    mpf_plot(chart)
