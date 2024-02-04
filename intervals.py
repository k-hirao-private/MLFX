interval_m_list = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "45min": 45,
    "1h": 60,
    "2h": 60 * 2,
    "4h": 60 * 4,
    "1day": 60 * 24,
    "1week": 60 * 24 * 7,
}


def str_to_interval(str, unit):
    if unit in ["m", "min", "minute"]:
        return interval_m_list[str]
    elif unit in ["h", "hour"]:
        return interval_m_list[str] / 60
    elif unit in ["d", "day"]:
        return interval_m_list[str] / 60 / 24
