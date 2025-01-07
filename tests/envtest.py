import unittest
from environment import TradingEnv
import yfinance as yf


class TradingEnvTests(unittest.TestCase):
    """
    Objectives for further test cases (more to be added later):
    - do datatype checks on every column to ensure every column contains only data type
    - must return float reward
    - env.step must return tuple of not NaN values (tuple must be valid size >= 4)
        - each value in tuple must not be None
    - observation must be dataframe of floats
    - trf must be float
    - ensure no NaN values (can either preprocessed beforehand or keep this as an edge case - to be discussed further)
    - reset must reset values to defaults (0, None, etc. depending on the value)
    - ensure capital does not fall below min threshold
    """

    def test_no_adj_close(self):
        tickers = yf.Tickers("AAPL GOOGL MSFT")
        # data = tickers.download(end="2020-01-01")
        data = yf.download(
            tickers="AAPL GOOGL MSFT",
            start="2019-01-01",
            end="2020-01-01",
        )
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
            episode_len=5,
            index_to_id=["AAPL", "GOOGL", "MSFT"],
            seed=123,
        )

        obs, reward, terminated, truncated, info = my_env.step([0, 0, 0, 1])

        print(my_env._portfolio_value)

    # def assert_reward():
    #     return type(reward)


if __name__ == "__main__":
    unittest.main()
