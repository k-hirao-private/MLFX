from twelvedata import TDClient
from twelvedata.exceptions import TwelveDataError
import settings
import os
import datetime
import time
import pandas as pd
import pandas.tseries.offsets as offsets
import json


class ChartManager:
    td = TDClient(apikey=settings.chart_log["TDClient_apikey"])
    save_dir = settings.chart_log["save_dir"]
    outputsize = 5000

    def download(self, symbol, interval, end_date):
        while 1:
            try:
                log = self.td.time_series(
                    symbol=symbol,
                    interval=interval,
                    start_date=settings.chart_log["start_date"],
                    end_date=end_date,
                    outputsize=self.outputsize,
                    timezone=settings.chart_log["timezone"],
                ).as_pandas()
                break
            except TwelveDataError as e:
                print(e)
                # print("Waiting for API")
                time.sleep(5)

        print(symbol, interval, len(log.index), log.index[0], log.index[-1])
        if len(log.index) < self.outputsize:
            return log
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

    def update(self):
        os.makedirs(self.save_dir, exist_ok=True)
        updated_files = 0

        for symbol in settings.chart_log["symbols"]:
            for interval in settings.chart_log["intervals"]:
                df = self.download(symbol, interval, end_date=datetime.date.today())

                file_name = symbol.replace("/", "-") + "_" + interval + ".json"

                data = {
                    "symbol": symbol,
                    "interval": interval,
                    "created": datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f"),
                    "chart": None,
                }
                data["chart"] = [
                    {
                        "time": int(item.Index.timestamp()),
                        "open": item.open,
                        "high": item.high,
                        "low": item.low,
                        "close": item.close,
                    }
                    for item in df.itertuples()
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
