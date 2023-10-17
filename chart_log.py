from twelvedata import TDClient
from twelvedata.exceptions import TwelveDataError
import settings
import os
import datetime
import time
import pandas as pd
import pandas.tseries.offsets as offsets
import json
import statistics


class ChartManager:
    td = TDClient(apikey=settings.chart_log["TDClient_apikey"])
    save_dir = settings.chart_log["save_dir"]
    outputsize = 5000
    bollinger_range = settings.chart_log["bollinger_range"]

    def get_data(self, symbol, interval, **kwargs):
        while 1:
            try:
                log = self.td.time_series(
                    symbol=symbol, interval=interval, **kwargs
                ).as_pandas()
                break
            except TwelveDataError as e:
                print(e)
                time.sleep(5)
        print(symbol, interval, len(log.index), log.index[0], log.index[-1])
        return log

    def download(self, symbol, interval, end_date):
        log = self.get_data(
            symbol=symbol,
            interval=interval,
            start_date=settings.chart_log["start_date"],
            end_date=end_date,
            outputsize=self.outputsize,
            timezone=settings.chart_log["timezone"],
        )

        if len(log.index) < self.outputsize:
            return pd.concat(
                [
                    log,
                    self.get_data(
                        symbol=symbol,
                        interval=interval,
                        end_date=log.index[-1] - offsets.Minute(1),
                        outputsize=self.bollinger_range,
                        timezone=settings.chart_log["timezone"],
                    ),
                ]
            )
        else:
            return pd.concat(
                [
                    log,
                    self.download(
                        symbol,
                        interval,
                        log.index[self.outputsize - 1] - offsets.Minute(1),
                    ),
                ]
            )

    def bollinger(self, i):
        data = self.df.iloc[
            i : i + self.bollinger_range, self.df.columns.get_loc("close")
        ]
        SMA = statistics.mean(data)
        sigma = statistics.pstdev(data)
        return {
            "SMA": SMA,
            "bollinger_upper": SMA + sigma,
            "bollinger_lower": SMA - sigma,
        }
        pass

    def update(self):
        os.makedirs(self.save_dir, exist_ok=True)
        updated_files = 0

        for symbol in settings.chart_log["symbols"]:
            for interval in settings.chart_log["intervals"]:
                self.df = self.download(
                    symbol, interval, end_date=datetime.date.today()
                )

                file_name = symbol.replace("/", "-") + "_" + interval + ".json"

                data = {
                    "symbol": symbol,
                    "interval": interval,
                    "created": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f"),
                    "chart": None,
                }
                data["chart"] = [
                    dict(
                        time=int(self.df.iloc[i].name.timestamp()),
                        open=self.df.iloc[i].open,
                        high=self.df.iloc[i].high,
                        low=self.df.iloc[i].low,
                        close=self.df.iloc[i].close,
                        **self.bollinger(i)
                    )
                    for i in range(0, len(self.df) - self.bollinger_range)
                ]
                json_object = json.dumps(data, indent=4)
                with open(self.save_dir + file_name, "w") as outfile:
                    outfile.write(json_object)
                updated_files = updated_files + 1
                print("updated : " + symbol + "\t" + interval)

        print("updated " + str(updated_files) + " files")


if __name__ == "__main__":
    chart = ChartManager()
    chart.update()
