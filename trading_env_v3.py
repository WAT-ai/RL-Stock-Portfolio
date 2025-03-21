import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from datetime import timedelta
from deepar import test_model  # Import DeepAR testing function
from load_12data import load_data

class PortfolioEnv(gymnasium.Env):
    def __init__(self, tickers, start_date, end_date, initial_balance=100000, window_len=20, deepar_model=None):
        """
        Initialize the Portfolio Environment.

        Parameters:
          tickers (list): List of stock tickers.
          start_date, end_date (str): Date range for historical data.
          initial_balance (float): Starting portfolio value.
          window_len (int): Number of days in the observation window.
          deepar_model: Pretrained DeepAR model (optional). If provided, its predictions
                        (one per ticker) will be appended to the observation.
        """
        super(PortfolioEnv, self).__init__()

        self.tickers = tickers
        self.num_stocks = len(tickers)
        self.window_len = window_len
        self.deepar_model = deepar_model  # Save the DeepAR model

        # Total assets include cash plus each stock.
        self.num_assets = self.num_stocks + 1

        # self.data = self._load_data(tickers, start_date, end_date)
        self.data = load_data(tickers, start_date, end_date)
        print(self.data)

        self.dates = self.data.index.get_level_values("Date").unique()
        print(self.dates)

        # Portfolio properties.
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.weights[0] = 1.0  # Start with 100% cash
        self.portfolio_value = initial_balance
        self.current_step = 0

        # Action space: portfolio weights.
        self.action_space = Box(low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)

        # Observation space: historical features (6 per ticker per day), current weights,
        # and (if provided) one extra prediction per ticker.
        obs_len_per_day_per_ticker = 6
        extra_dim = self.num_stocks if self.deepar_model is not None else 0

        self.obs_shape = (self.window_len * self.num_stocks * obs_len_per_day_per_ticker) + self.num_assets + extra_dim
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)

    def _load_data(self, tickers, start_date, end_date):
        start_dt = pd.to_datetime(start_date)
        pull_dt = start_dt - timedelta(days=30)

        data = yf.download(tickers, start=pull_dt.strftime("%Y-%m-%d"), end=end_date, interval="1d")
        data = data[["Open", "High", "Low", "Close", "Volume"]]
        data = data.stack(level=1).reset_index()
        data = data.rename(columns={"level_0": "Date", "level_1": "Ticker"})
        data = data.set_index(["Date", "Ticker"]).sort_index()

        def compute_indicators(df_):
            df_ = df_.sort_values("Date").copy()
            df_["RSI"] = ta.rsi(df_["Close"], length=14)
            return df_

        data = data.groupby("Ticker", group_keys=False).apply(compute_indicators)
        data = data.groupby(level="Ticker").ffill().fillna(0.0)
        data = data[["Open", "High", "Low", "Close", "Volume", "RSI"]]
        data = data.loc[data.index.get_level_values("Date") >= start_dt]
        return data

    def reset(self):
        """Reset the environment to the initial state."""
        self.balance = self.initial_balance
        self.weights = np.zeros(self.num_assets, dtype=np.float32)
        self.weights[0] = 1.0
        self.portfolio_value = self.initial_balance
        self.current_step = self.window_len

        return self._get_observation(), {}

    def step(self, action):
        """Execute one time step within the environment."""
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action[0] = 1.0
        action /= np.sum(action)
        self.weights = action

        current_prices = self.data.loc[self.dates[self.current_step], :]["Close"].values
        next_prices = self.data.loc[self.dates[self.current_step + 1], :]["Close"].values

        returns = (next_prices - current_prices) / current_prices
        returns = np.insert(returns, 0, 0.0)

        portfolio_return = np.dot(self.weights, returns)
        self.portfolio_value *= (1 + portfolio_return)

        risk = np.std(returns) if np.std(returns) > 0 else 1e-8
        reward = np.log(abs(portfolio_return / risk + 1))

        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        return self._get_observation(), reward, done, False, {"portfolio_value": self.portfolio_value}
    
    def _get_observation(self):
        date = self.dates[self.current_step]
        raw_data = self.data.loc[(date, slice(None)), :].reset_index().sort_values(["Date", "Ticker"])
        norm_sub_data = raw_data.copy()
        print("THE DATA IS RIGHT HERE", norm_sub_data[["Open", "High", "Low", "Close"]])
        norm_sub_data[["Open", "High", "Low", "Close"]] /= norm_sub_data["Close"]
        norm_sub_data["Volume"] /= self.data["Volume"].max()
        norm_sub_data["RSI"] /= self.data["RSI"].max()
        observation = norm_sub_data[["Open", "High", "Low", "Close", "Volume", "RSI"]].to_numpy().flatten()
        observation = np.concatenate([observation, self.weights], axis=0)
        return observation

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value}, Weights: {self.weights}")


def test():
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]
    start_date = "2010-01-01"
    end_date = "2021-01-01"
    env = PortfolioEnv(tickers, start_date, end_date)
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()

test()
