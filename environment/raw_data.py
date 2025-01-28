from dotenv import dotenv_values
from twelvedata import TDClient
import pandas_market_calendars as mcal
import pandas as pd
from datetime import timedelta
from itertools import batched
import time

env_config = dotenv_values(".env")

td = TDClient(env_config["TWELVE_DATA_API_KEY"])
nyse_cal = mcal.get_calendar("NYSE")


def _load_data_limited(symbols: tuple[str], start_date, end_date):
    """
    Get adjusted OHCLV data for symbols in open days across [start_date, end_date].

    Uses 1 API token per symbol per call. Limited at 8 symbols per call (12data limit).
    """

    assert len(symbols) <= 8

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
    )
    ts["Date"] = pd.to_datetime(ts["Date"])
    ts = ts[ts["Date"].isin(open_days)]

    return ts


def load_data(symbols: list[str], start_date, end_date, _batch_size=8):
    """
    Get adjusted OHCLV data for symbols in open days across [start_date, end_date].

    Uses len(symbols) // _batch_size API tokens per symbol per call.

    *Do not* pass in a different value for _batch_size unless you have a good reason to.
    """
    _RATE_LIMIT = 60  # seconds between calls

    ret = pd.DataFrame()
    for i, symbol_batch in enumerate(batched(symbols, n=_batch_size)):
        if i != 0:
            # don't wait for rate limit on first call
            time.sleep(_RATE_LIMIT)
        batch_df = _load_data_limited(symbol_batch, start_date, end_date)
        ret = pd.concat([ret, batch_df])

    ret.sort_values(by=["Date", "Id"], inplace=True)
    ret.reset_index(drop=True, inplace=True)
    return ret


if __name__ == "__main__":
    data = load_data(
        ["AAPL", "MSFT", "TSLA", "VOD", "NVDA", "AMZN", "BA", "DELL", "INTC", "CRM"],
        start_date="2020-01-01",
        end_date="2020-02-01",
    )
    print(data)
