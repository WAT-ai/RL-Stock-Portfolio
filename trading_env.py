import gym
from gym import spaces
from gym.spaces.box import Box
import numpy as np
import pandas as pd
import yfinance as yf

class PortfolioEnv(gym.Env):
    def __init__(self, tickers, start_date, end_date, initial_balance=100000):
        """
        Initialize the Portfolio Environment.
        :param tickers: List of stock tickers to include in the portfolio.
        :param start_date: Start date for historical data.
        :param end_date: End date for historical data.
        :param initial_balance: Initial portfolio balance.
        """
        super(PortfolioEnv, self).__init__()

        # Load data
        self.tickers = tickers
        self.data = self._load_data(tickers, start_date, end_date)

        # -------------------------------------------------
        # 1) We'll step by date only (not date+ticker).
        #    So we get unique dates from level='Date'.
        # -------------------------------------------------
        self.dates = self.data.index.get_level_values("Date").unique()

        # Portfolio properties
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.weights = np.array([1.0 / len(tickers)] * len(tickers))  # Start with equal allocation
        self.portfolio_value = initial_balance
        self.current_step = 0

        # Define action and observation spaces
        # Action = weights for each ticker
        self.action_space = Box(low=0.0, high=1.0, shape=(len(tickers),), dtype=np.float32)

        # Observation = all tickers' (Open, High, Low, Close, Volume)
        # in a single date. That is shape = (len(tickers) * 5).
        self.observation_space = Box(
            low=0, high=np.inf, shape=(len(self.tickers) * 5,), dtype=np.float32
        )

    def _load_data(self, tickers, start_date, end_date):
        """Load historical data using Yahoo Finance."""
    
        # 1. Download data
        data = yf.download(tickers, start=start_date, end=end_date, interval="1d")

        # 2. Select relevant OHLCV columns
        data = data["Open High Low Close Volume".split()]
        print("Data after selecting OHLCV columns:")
        print(data.head())

        # 3. Stack on level=1 (the TICKER level), then reset index
        data = data.stack(level=1).reset_index()
        print("Data after stacking and resetting index:")
        print(data.head())

        # 4. Rename columns for clarity
        data = data.rename(columns={"level_0": "Date", "level_1": "Ticker"})
        print("Data after renaming columns:")
        print(data.head())

        # 5. Melt to go from wide to long
        data = data.melt(id_vars=["Date", "Ticker"],
                         var_name="Price_Type",
                         value_name="Value")
        print("Data after melting:")
        print(data.head())

        # 6. Pivot to get columns = Price_Type, index = (Date, Ticker)
        data = data.pivot(index=["Date", "Ticker"], columns="Price_Type", values="Value")
        print("Data after pivoting:")
        print(data.head())

        # 7. Return the pivoted DataFrame
        return data
    
    def reset(self):
        """Reset the environment to the initial state."""
        self.balance = self.initial_balance
        self.weights = np.array([1.0 / len(self.tickers)] * len(self.tickers))
        self.portfolio_value = self.initial_balance
        # Choose a random start index so each episode doesn't always begin at day 0
        # We subtract 1 to ensure there's at least one more step after we start.
        self.current_step = np.random.randint(0, len(self.dates) - 253)
        print("Environment reset:")
        print(f"Initial balance: {self.balance}")
        print(f"Initial portfolio value: {self.portfolio_value}")
        print(f"Initial weights: {self.weights}")

        info = {}

        return self._get_observation(), info

    def step(self, action):
        """Execute one time step within the environment.
        :param action: Vector of weights for the portfolio (must sum to 1).
        """
        print(f"Step: {self.current_step}")
        print(f"Action: {action}")

        # Normalize action to ensure it sums to 1
        action = np.clip(action, 0, 1)
        if np.sum(action) == 0:
            action = np.array([1.0 / len(action)] * len(action))
        action /= np.sum(action)
        self.weights = action
        print(f"Weights after normalization: {self.weights}")

        # Extract prices
        try:
            current_prices = self.data.loc[self.dates[self.current_step], :]["Close"].values
            next_prices = self.data.loc[self.dates[self.current_step + 1], :]["Close"].values
            print(f"Current prices: {current_prices}")
            print(f"Next prices: {next_prices}")
        except KeyError as e:
            print(f"KeyError during price extraction: {e}")
            raise

        # Calculate returns
        if len(current_prices) == 0 or len(next_prices) == 0:
            raise ValueError(f"Empty price arrays. Current prices: {current_prices}, Next prices: {next_prices}")
        returns = (next_prices - current_prices) / current_prices
        print(f"Returns: {returns}")

        # Calculate portfolio return
        portfolio_return = np.dot(self.weights, returns)
        print(f"Portfolio return: {portfolio_return}")
        self.portfolio_value *= (1 + portfolio_return)
        print(f"Updated portfolio value: {self.portfolio_value}")

        # Calculate reward (log of risk-adjusted return)
        risk = np.std(returns) if np.std(returns) > 0 else 1e-8
        val = portfolio_return / risk + 1
        val = max(val, 1e-8)  # or some positive epsilon
        reward = np.log(val)

        # Update step
        self.current_step += 1
        done = self.current_step >= len(self.dates) - 1

        return self._get_observation(), reward, done, False, {"portfolio_value": self.portfolio_value}

    def _get_observation(self):
        """Get the current observation and normalize it."""
        print(f"Current step: {self.current_step}")
        current_date = self.dates[self.current_step]
        print(f"Current date: {current_date}")

        # --------------------------------------------------
        # Select *all* tickers for current_date
        # (shape ~ (num_tickers, 5))
        # --------------------------------------------------
        current_data = self.data.loc[(current_date, slice(None)), :]
        # current_data index = MultiIndex [ (date, ticker1), (date, ticker2), ... ]
        # We want it in a DataFrame with each ticker as a row.

        # The rows are unique tickers. Let's ensure it's sorted by Ticker just in case:
        current_data = current_data.reset_index(level="Date", drop=True)
        current_data = current_data.sort_index()  # sort by Ticker

        # Now current_data index = Ticker, columns = [Open, High, Low, Close, Volume]
        print("Current data for all tickers (before normalization):")
        print(current_data)

        # Normalize: 1) prices by that row's close; 2) volume by max volume per ticker
        close_prices = current_data["Close"]
        current_data[["Open", "High", "Low", "Close"]] = current_data[["Open", "High", "Low", "Close"]].div(
            close_prices, axis=0
        )

        print("Data after normalizing (Open,High,Low,Close):")
        print(current_data)

        # ---------------------------------------------------
        # Volume normalization
        # We have one row per ticker (index = ticker).
        # We'll divide by that ticker's max volume
        # (calculated across entire dataset).
        # ---------------------------------------------------
        max_volume_per_ticker = self.data["Volume"].groupby("Ticker").max()
        # current_data.index is the list of tickers
        # So we can map each ticker's index to max_volume_per_ticker
        current_data["Volume"] = current_data.apply(
            lambda row: row["Volume"] / max_volume_per_ticker.loc[row.name],
            axis=1
        )

        print("Data after normalizing volume:")
        print(current_data)

        # Flatten the observation into a 1D array: shape = (num_tickers * 5,)
        observation = current_data[["Open", "High", "Low", "Close", "Volume"]].values.flatten()
        print("Final observation shape:", observation.shape)
        print(observation)
        return observation
    
    def render(self, mode="human"):
        """Render the environment."""
        print(f"Step: {self.current_step}, "
              f"Portfolio Value: {self.portfolio_value}, "
              f"Weights: {self.weights}")
