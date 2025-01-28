import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset
from twelvedata import TDClient
from twelvedata.exceptions import InvalidApiKeyError
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpdates
from matplotlib.patches import Rectangle


load_dotenv()


class StockOHLCVDataset(Dataset):
    def __init__(self, symbols: list[str], window_len: int, outputsize: int = 5000, api_key: str = ""):
        # API key is optional, will automatically check .env folder for correctly named key
        self.window_len = window_len
        self.max_data_size = outputsize
        self.symbols = symbols

        if not api_key:
            try:
                api_key = os.environ["TWELVEDATA_API_KEY"]
            except KeyError:
                error_message = "Cannot find TwelveData API key. Try one of the following: 1. Moving your .env file to project root 2. Checking the name of your API key (should be TWELVEDATA_API_KEY) 3. Manually passing in a valid API key during datasest initialization"
                raise KeyError(error_message)
        try:
            self.td_client = TDClient(apikey=api_key)
            self.timeseries_data = self.td_client.time_series(
                symbol=self.symbols,
                interval="1day",
                outputsize=self.max_data_size
            ).as_pandas()
        except InvalidApiKeyError:
            error_message = "TwelveData API key is potentially invalid, expired, or misstyped. You can get your free API key instantly following this link: https://twelvedata.com/pricing. If you believe that everything is correct, you can contact us at https://twelvedata.com/contact/customer"
            raise InvalidApiKeyError(error_message)

        # print(self.timeseries_data)
        
    def __len__(self) -> int:
        return len(self.timeseries_data)

    def __str__(self) -> str:
        ...

    def __getitem__(self, idx: int, enable_plotting: bool = False) -> tuple[torch.FloatTensor]:
        seq = self.timeseries_data.iloc[idx:idx+self.window_len]
        feat = torch.tensor(seq.values, dtype=torch.float32)
        label = torch.tensor(self.timeseries_data.iloc[idx+self.window_len], dtype=torch.float32) # for prediction only
        
        if enable_plotting:
            self.plot_ohlcv(idx)
        
        return feat, label

    def plot_ohlcv(self, idx: int) -> None:
        """Plot OHLCV data for a given index using candlestick chart"""

        window_data = self.timeseries_data.iloc[idx:idx+self.window_len].copy(deep=True)
        print(window_data)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Format dates for x-axis
        window_data['datetime'] = pd.to_datetime(window_data.index)
        window_data['date_num'] = window_data['datetime'].map(mpdates.date2num)
        
        # Prepare OHLC data for candlestick chart
        ohlc = window_data[['date_num', 'open', 'high', 'low', 'close']].values
        
        # Plot candlestick chart
        candlestick_ohlc(ax1, ohlc, width=0.6, colorup='g', colordown='r')
        
        toggle_labels = False
        if toggle_labels:
            # Add value labels for each candlestick
            for i, row in window_data.iterrows():
                # Add close price above each candlestick
                ax1.text(mpdates.date2num(row['datetime']), row['high'], 
                        f'Close: {row["close"]:.2f}', 
                        ha='center', va='bottom')
                # Add open price below each candlestick
                ax1.text(mpdates.date2num(row['datetime']), row['low'], 
                        f'Open: {row["open"]:.2f}', 
                        ha='center', va='top')
        
        ax1.xaxis.set_major_formatter(mpdates.DateFormatter('%Y-%m-%d'))
        ax1.set_title('OHLC Data')
        ax1.grid(True)
        
        # Add some padding to y-axis to make room for labels
        ax1.margins(y=0.1)
        
        # Plot volume
        ax2.bar(window_data['datetime'], window_data['volume'], color='blue', alpha=0.5)
        ax2.set_title('Volume')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


def main():
    dataset = StockOHLCVDataset(symbols=["AAPL"], outputsize=30, window_len=25)
    # random_index = np.random.randint(0, len(dataset) - 5)
    dataset.__getitem__(0, enable_plotting=True)

if __name__ == "__main__":
    main()

