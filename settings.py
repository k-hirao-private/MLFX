chart_log = {
    "symbols": ["USD/JPY"],
    "intervals": [
        "1min",
        "5min",
        "30min",
        "4h",
        "1day",
    ],
    "start_date": "2023-09-01",
    "timezone": "Asia/Tokyo",
    "save_dir": "chart_log/",
    "TDClient_apikey": "c2e23b2dc9264f218c71475a3dd052b6",
}
data_formatter = {
    "symbol": "USD/JPY",
    "intervals": [
        "1min",
        "5min",
        "30min",
        "4h",
        "1day",
    ],
    "base_interval": "1min",
    "multi_thread": True,
    "threads": "auto",
    "prog_interval": 1.0,
    "margin_pips": 20,
    "middle_pips": 10,
    "approximate_data_num": 10,
    "approximate_dim": 4,
    "save_dir": "formatted_data/",
}
