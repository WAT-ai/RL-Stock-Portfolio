import yfinance as yf
import pandas as pd

start_date = "2021-01-01"
end_date = "2021-01-31"
data = yf.download(['MSFT'], start=start_date, end=end_date, interval="1d")

