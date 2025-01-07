import unittest
from environment import TradingEnv
import yfinance as yf


class TradingEnvTests(unittest.TestCase):
    def test_no_adj_close(self):
        tickers = yf.Tickers("AAPL GOOGL MSFT")
        data = tickers.download()
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        flattened_data = (
            data.stack(level=1)  # Move tickers to rows
            .reset_index()  # Reset index to convert to a flat DataFrame
            .rename(columns={"level_1": "Ticker"})  # Rename the column for tickers
        )
        flattened_data.rename(columns={"Ticker": "Id"}, inplace=True)

        my_env = TradingEnv(
            ohclv_data=flattened_data,
            num_risky_assets=3,
            window_len=4,
            initial_capital=1000,
            transaction_cost=0,
            reward_function=lambda: 1,
            batch_len=5,
            index_to_id=["AAPL", "GOOGL", "MSFT"],
            seed=123,
        )

        my_env.step([0, 0, 0, 1])

        print(my_env._capital)


if __name__ == "__main__":
    unittest.main()
