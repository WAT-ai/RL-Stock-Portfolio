from dotenv import dotenv_values
from twelvedata import TDClient
import pandas_market_calendars as mcal
import pandas as pd
from datetime import date, timedelta

env_config = dotenv_values(".env")

td = TDClient(env_config["TWELVE_DATA_API_KEY"])
nyse_cal = mcal.get_calendar("NYSE")


def load_data(symbols: list[str], start_date, end_date):
    """
    Get adjusted OHCLV data for symbols in open days across [start_date, end_date].

    Uses 1 API token per symbol per call.
    """

    # add 1 day to end_date to make the date range inclusive
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    end_date += timedelta(days=1)

    open_days = nyse_cal.schedule(start_date=start_date, end_date=end_date).index
    ts = (
        td.time_series(
            symbol=symbols, interval="1day", start_date=start_date, end_date=end_date
        )
        .as_pandas()
        .reset_index()
        .rename(
            columns={
                "level_0": "Id",
                "level_1": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        .sort_values(by=["Date", "Id"])
    )
    ts["Date"] = pd.to_datetime(ts["Date"])
    ts = ts[ts["Date"].isin(open_days)]
    ts.reset_index(drop=True, inplace=True)

    return ts
